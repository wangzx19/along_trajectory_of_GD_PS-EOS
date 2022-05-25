import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
from matplotlib import pyplot as plt
import eigenthings_modified.hessian_eigenthings_mod as eig_mod
import math

T = 1200
m = 200 # size of net
print_epoch = 1
if_zero_mean = False
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

activation = 'softplus'
if activation == 'softplus':
    case = 5
if activation == 'elu':
    case = 6
else:
    case = 7
# case = 3 # 1: linear 2: deep linear 3: deep tanh 4: deep relu 31: whitened deep tanh

if case == 1:
    eta = 2/100
if case == 2:
    eta = 2/40
if case == 3 or case == 31:
    eta = 2/200
if case == 4 or case == 41:
    eta = 2/200

def activate(x, case):
    if case == 2:
        return x
    if case == 3 or case == 31:
        return F.tanh(x)
    if case == 4 or case == 41:
        return F.relu(x)
    if case == 5:
        return F.softplus(x)
    if case == 6:
        return F.elu(x)
    if case == 7:
        return x**2

class Net(nn.Module):
    def __init__(self, width):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(k, width, bias=False)
        self.fc2 = nn.Linear(width, 1, bias=False)
        self.fc3 = nn.Linear(width, width, bias=False)
        self.fc4 = nn.Linear(width, width, bias=False)
        self.fc5 = nn.Linear(width, width, bias=False)

    def forward(self, x):
        x = x.view(-1, k)
        x = self.fc1(x)
        x = activate(x, case)
        x = self.fc3(x)
        x = activate(x, case)
        x = self.fc4(x)
        x = activate(x, case)
        x = self.fc5(x)
        x = activate(x, case)
        x = self.fc2(x)
        return x

    def lock(self, i):
        for name, param in self.named_parameters():
            if "fc" + str(i) in name:
                param.requires_grad = False

    def unlock(self, i):
        for name, param in self.named_parameters():
            if "fc" + str(i) in name:
                param.requires_grad = True

def init_parameters(net):
    for layer in net.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            # nn.init.kaiming_uniform_(layer.weight, a=0)
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(layer.bias, -bound, bound)

k = 32 * 32 * 3
n = 1000 # num of data in the training set
# train_data= torch.load('./result/cifar/mnist-binary-data-balance_twoclass_' + str(n))
train_data = torch.load('../result/cifar/mnist-binary-data-balance_' + str(n))
train_loader = torch.utils.data.DataLoader(dataset=train_data, shuffle= False, batch_size=len(train_data))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Net(m)
init_parameters(model)
model.to(device)

P = 0
for layer_m in model.parameters():
    P += len(layer_m.view(-1))

optimizer = optim.SGD(model.parameters(), lr=eta)

def Matrix_vector_proj(v, M):
    k = len(v)
    return torch.matmul(v.view(1,k), torch.matmul(M, v.view(k,1))).item()

def data_pre(data, if_zeromean):
    if if_zeromean:
        alpha = torch.mean(data, 0)
        data = data - alpha


def criterion(output, target):
    output = output.view(-1)
    loss = torch.dot(output - target, output - target) / n
    return loss


def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        data_pre(data, if_zero_mean)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        if epoch % print_epoch == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, loss.item()))
            loss_list.append(loss.item())
        optimizer.step()

sharp_init = 0
x = []
cons_list = []
loss_list = []
similarity_list = []
sharp_list = []
trace_list = []
x_1 = []
x_2 = []
sharp_bad_list = []
A_part_sharp_list = []
sharp2_list = []
sharp3_list = []

bad_list = [1]
bad_noA_list = [1]

R_list = []
RY_list = []
norm_list = []
Dv_list = []
norm2_list = []
norm2_bad_list = []
norm3_list = []
norm4_list = []
norm5_list = []
FD_list = []

sharpness = 0
last_sharp = 0
last_v = 0
last_no_A_sharp = 0
last_norm1 = 0
last_norm2 = 0
last_norm3 = 0
last_norm4 = 0
last_norm5 = 0

CNT = 0
no_A_CNT = 0

max_eps_2 = 0
max_D_Dv = 0


for epoch in range(0, T):


    if epoch % print_epoch == 0:

        M = 0
        """calculate M"""
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            data_pre(data, if_zero_mean)
            data = data.view(-1, k)
            target = target.to(device)
            f = model(data).view(-1)
            D = f - target
            f_grad = torch.zeros((n, P)).to(device)
            A_part_grad = torch.zeros((n, m)).to(device)
            for j in range(n):
                delta_c_dict = torch.autograd.grad(
                    f[j], model.parameters(), only_inputs=True, retain_graph=True
                )
                delta_c = torch.cat([g.contiguous().view(-1) for g in delta_c_dict])
                for g in delta_c_dict:
                    if g.view(-1).shape[0] == m:
                        A_part_grad[j] = g.contiguous().view(-1)
                f_grad[j] = delta_c
            M = torch.matmul(f_grad, f_grad.t())
            M_A = torch.matmul(A_part_grad, A_part_grad.t())

            Meigvalue, Meigvec = torch.linalg.eigh(M)
            print('sharpness:', Meigvalue[-1].item() * 2/n)
            # A_part_sharpness = Matrix_vector_proj(Meigvec[:, -1], M_A) * 2/n
            # print('A_Part sharpness:', A_part_sharpness)
            M_Aeigvalue, M_Aeigvec = torch.linalg.eigh(M_A)
            print('M_A 2 norm', M_Aeigvalue[-1].item() *2/n)
            # print('2nd sharpness:', Meigvalue[-2].item() * 2 / n)

            # print('second sharpness:', Meigvalue[-2].item() * 2/n)
            # print('third sharpness:', Meigvalue[-3].item() * 2 / n)
            x.append(epoch)

            v = Meigvec[:, -1].view(n,1)
            R_t = torch.matmul(D, (torch.eye(n).to(device) - torch.matmul(v, v.t())))
            R_list.append(torch.norm(R_t).item()**2 * 1 / n)
            RY_list.append(torch.dot(R_t.view(-1), target.float()).item())


            A_part_sharp_list.append(M_Aeigvalue[-1].item() *2/n)

            cons_list.append(2/eta)
            sharpness = Meigvalue[-1].item() * 2/n
            Dv_list.append(torch.dot(D.view(-1), v.view(-1)).item())

            # print('fc1 weight norm', torch.norm(model.fc1.weight.view(-1)).item() ** 2)
            print('fc2 weight norm', torch.norm(model.fc2.weight.view(-1)).item() ** 2)
            # print('fc3 weight norm', torch.norm(model.fc3.weight.view(-1)).item() ** 2)
            # print('fc4 weight norm', torch.norm(model.fc4.weight.view(-1)).item() ** 2)
            # print('fc5 weight norm', torch.norm(model.fc5.weight.view(-1)).item() ** 2)
            #
            norm_list.append(torch.norm(model.fc1.weight.view(-1)).item() ** 2)
            # norm2_list.append(torch.norm(model.fc2.weight.view(-1)).item() ** 2)
            norm3_list.append(torch.norm(model.fc3.weight.view(-1)).item() ** 2)
            norm4_list.append(torch.norm(model.fc4.weight.view(-1)).item() ** 2)
            norm5_list.append(torch.norm(model.fc5.weight.view(-1)).item() ** 2)
            #
            # FD_list.append(torch.dot(D, f).item())
            # print('FD:', torch.dot(D, f).item())
            D_Dv = (torch.norm(D)/torch.dot(D.view(-1), v.view(-1))).item()
            if D_Dv > max_D_Dv:
                max_D_Dv = D_Dv

            #
            sharp_list.append(Meigvalue[-1].item() * 2 / n)
            if epoch > 0:
                print(1- torch.dot(v.view(-1), last_v.view(-1)).item())
                eps_2 = 1 - torch.matmul(v.t(), last_v).item()
                if eps_2 > max_eps_2:
                    max_eps_2 = eps_2
                norm2 = torch.norm(model.fc2.weight.view(-1)).item() ** 2
                if (norm2 - last_norm2) * (sharpness - last_sharp) < 0:
                    print('bad')
                    CNT += 1
                    bad_list.append(-1)
                    x_2.append(epoch)
                    sharp_bad_list.append(Meigvalue[-1].item() * 2 / n)
                    norm2_bad_list.append(norm2)
                else:
                    bad_list.append(1)
                    # sharp_list.append(Meigvalue[-1].item() * 2 / n)
                    x_1.append(epoch)
                    norm2_list.append(norm2)


            last_norm2 = torch.norm(model.fc2.weight.view(-1)).item() ** 2
            last_sharp = sharpness
            last_v = v

        '''calculate M_A_pertube'''
        # for batch_idx, (data, target) in enumerate(train_loader):
        #     data = data.to(device)
        #     if (if_zero_mean):
        #         alpha = torch.mean(data, 0)
        #         data = data - alpha
        #     if if_whitened:
        #         data = data.view(n, k)
        #         for i in range(n):
        #             data[i] = torch.zeros(data[i].shape)
        #             data[i][i] = 1
        #     data = data.view(-1, k)
        #     target = target.to(device)
        #     f = model(data).view(-1)
        #     loss = criterion(f, target)
        #     optimizer.zero_grad()
        #     loss.backward(retain_graph=False)
        #     outlayer_grad = model.fc2.weight.grad
        #
        #     dick = model.state_dict()
        #     model_A_pertube.load_state_dict(dick)
        #     dick_A = model_A_pertube.state_dict()
        #     for para in dick_A:
        #         if 'fc2' in para:
        #             dick_A[para] -= outlayer_grad * eta
        #     model_A_pertube.load_state_dict(dick_A)
        #     f_A_pertube = model_A_pertube(data).view(-1)
        #
        #     f_grad_A_pertube = torch.zeros((n, P)).to(device)
        #     for j in range(n):
        #         delta_c_dict = torch.autograd.grad(
        #             f_A_pertube[j], model_A_pertube.parameters(), only_inputs=True, retain_graph=True
        #         )
        #         delta_c = torch.cat([g.contiguous().view(-1) for g in delta_c_dict])
        #         f_grad_A_pertube[j] = delta_c
        #     M_A_pertube = torch.matmul(f_grad_A_pertube, f_grad_A_pertube.t())
        #
        #     M_A_eigvalue, M_A_eigvec = torch.linalg.eigh(M_A_pertube)
        #     print('A sharpness change:', M_A_eigvalue[-1].item() * 2 / n - sharpness)
        #
        #     print("A brings change:", torch.norm((M - M_A_pertube).view(-1)).item())

        '''calculate M_W_pertube'''
        # for batch_idx, (data, target) in enumerate(train_loader):
        #     data = data.to(device)
        #     if (if_zero_mean):
        #         alpha = torch.mean(data, 0)
        #         data = data - alpha
        #     if if_whitened:
        #         data = data.view(n, k)
        #         for i in range(n):
        #             data[i] = torch.zeros(data[i].shape)
        #             data[i][i] = 1
        #     data = data.view(-1, k)
        #     target = target.to(device)
        #
        #     dick = model.state_dict()
        #     model_W_pertube.load_state_dict(dick)
        #     f_W_pertube = model_W_pertube(data).view(-1)
        #     loss = criterion(f_W_pertube, target)
        #     optimizer_W.zero_grad()
        #     loss.backward(retain_graph=True)
        #     outlayer_grad = model_W_pertube.fc2.weight.grad
        #     optimizer_W.step()
        #
        #     dick_W = model_W_pertube.state_dict()
        #     for para in dick_W:
        #         if 'fc2' in para:
        #             dick_W[para] += outlayer_grad * eta
        #     model_W_pertube.load_state_dict(dick_W)
        #     f_W_pertube = model_W_pertube(data).view(-1)
        #
        #     f_grad_W_pertube = torch.zeros((n, P)).to(device)
        #     for j in range(n):
        #         delta_c_dict = torch.autograd.grad(
        #             f_W_pertube[j], model_W_pertube.parameters(), only_inputs=True, retain_graph=True
        #         )
        #         delta_c = torch.cat([g.contiguous().view(-1) for g in delta_c_dict])
        #         f_grad_W_pertube[j] = delta_c
        #     M_W_pertube = torch.matmul(f_grad_W_pertube, f_grad_W_pertube.t())
        #
        #     M_W_eigvalue, M_W_eigvec = torch.linalg.eigh(M_W_pertube)
        #     print('W sharpness change:', M_W_eigvalue[-1].item() * 2 / n - sharpness)
        #
        #     print("W brings change:", torch.norm((M - M_W_pertube).view(-1)).item())
    # model.lock(2)
    train(epoch)
    # model.unlock(2)


# print("max eps:", max_eps_2)
# print("c:", 1/max_eps_2/max_D_Dv)
# print('Bad totally:', CNT)
# print('Bad no A totally:', no_A_CNT)

torch.save(sharp_list, './result/Final_checkpoints/deep_tanh_sharpness_lock_2'+str(activation))
torch.save(x_1, './result/Final_checkpoints/deep_tanh_x1'+str(activation))
torch.save(x_2, './result/Final_checkpoints/deep_tanh_x2'+str(activation))
torch.save(sharp_bad_list, './result/Final_checkpoints/deep_tanh_sharpness_bad'+str(activation))
torch.save(A_part_sharp_list, './result/Final_checkpoints/deep_tanh_A_sharpness'+str(activation))
torch.save(loss_list, './result/Final_checkpoints/deep_tanh_loss'+str(activation))
torch.save(R_list, './result/Final_checkpoints/deep_tanh_R'+str(activation))
# torch.save(RY_list, './result/Final_checkpoints/deep_tanh_RY')
# torch.save(norm_list, './result/Final_checkpoints/deep_tanh_norm_list')
# torch.save(norm3_list, './result/Final_checkpoints/deep_tanh_norm3_list')
# torch.save(norm4_list, './result/Final_checkpoints/deep_tanh_norm4_list')
# torch.save(norm5_list, './result/Final_checkpoints/deep_tanh_norm5_list')
torch.save(norm2_list, './result/Final_checkpoints/deep_tanh_norm2_list')
torch.save(norm2_bad_list, './result/Final_checkpoints/deep_tanh_norm2_bad_list'+str(activation))
#
#
#
#
#
# plt.title("sharp plot")
# plt.xlabel("epoch")
# plt.ylabel("sharp")
# plt.plot(x, sharp_list)
# plt.plot(x_2, sharp_bad_list, '.')
# plt.plot(x, A_part_sharp_list, 'b')
# plt.plot(x, cons_list, '--')
# plt.savefig('./result/Final_pictures/deep_tanh_sharpness')
# plt.close()
#
# plt.plot(x, loss_list)
# plt.title("loss plot")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.plot(x, loss_list)
# plt.savefig('./result/Final_pictures/deep_tanh__loss')
# plt.close()
#
# plt.plot(x, loss_list)
# plt.title("loss plot")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.plot(x, loss_list)
# plt.plot(x, R_list)
# plt.savefig('./result/Final_pictures/deep_tanh_loss_with_R')
# plt.close()
#
#
# plt.title("RY plot")
# plt.xlabel("epoch")
# plt.ylabel("RY")
# plt.plot(x, RY_list)
# plt.savefig('./result/Final_pictures/deep_tanh_RY')
# plt.close()
#
# # plt.title("trace plot")
# # plt.xlabel("epoch")
# # plt.ylabel("trace")
# # plt.plot(x, trace_list)
# # plt.savefig('./result/Final_pictures/deep_tanh__trace' + str(T) + description)
# # plt.close()
#
# #
# plt.plot(x, norm_list)
# plt.title("norm plot")
# plt.xlabel("epoch")
# plt.ylabel("norm")
# plt.plot(x, norm_list)
# plt.savefig('./result/Final_pictures/deep_tanh_norm1')
# plt.close()
# #
#
# plt.title("norm2 plot")
# plt.xlabel("epoch")
# plt.ylabel("A norm")
# plt.plot(x_1, norm2_list)
# plt.plot(x_2, norm2_bad_list, '.')
# plt.savefig('./result/Final_pictures/deep_tanh_norm2')
# plt.close()
#
# # plt.title("norm2 plot")
# # plt.xlabel("epoch")
# # plt.ylabel("norm2")
# # plt.plot(x_1, norm2_list, ',')
# # plt.plot(x_2, norm2_bad_list, '.')
# # plt.savefig('./result/4-29-general-exp/' + str(case) + 'case_norm2_pixel' + str(T) + description)
# # plt.close()
# #
# plt.plot(x, norm3_list)
# plt.title("norm3 plot")
# plt.xlabel("epoch")
# plt.ylabel("norm3")
# plt.plot(x, norm3_list)
# plt.savefig('./result/Final_pictures/deep_tanh_norm3')
# plt.close()
#
# plt.plot(x, norm4_list)
# plt.title("norm4 plot")
# plt.xlabel("epoch")
# plt.ylabel("norm4")
# plt.plot(x, norm4_list)
# plt.savefig('./result/Final_pictures/deep_tanh_norm4')
# plt.close()
#
# plt.plot(x, norm5_list)
# plt.title("norm5 plot")
# plt.xlabel("epoch")
# plt.ylabel("norm5")
# plt.plot(x, norm5_list)
# plt.savefig('./result/Final_pictures/deep_tanh_norm5')
# plt.close()

# plt.plot(x, FD_list)
# plt.title("FD plot")
# plt.xlabel("epoch")
# plt.ylabel("FD")
# plt.plot(x, FD_list)
# plt.savefig('./result/4-29-general-exp/' + str(case) + 'case_FD' + description)
# plt.close()

# plt.plot(x, similarity_list)
# plt.title("similar plot")
# plt.xlabel("epoch")
# plt.ylabel("similarity")
# plt.plot(x, similarity_list)
# plt.savefig('./result/4-29-general-exp/' + str(case) + 'case_similarity')
# plt.close()
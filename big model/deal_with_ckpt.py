import torch
from matplotlib import pyplot as plt

T = 800
eta = 2/400
activation = 'elu'
sharp_list = torch.load('./result/Final_checkpoints/deep_tanh_sharpness_lock_2_conv'+str(activation))
# sharp2_list = torch.load('./result/Final_checkpoints/deep_tanh_sharpness2_conv')
x_1 = torch.load('./result/Final_checkpoints/deep_tanh_x1_conv'+str(activation))
x_2 = torch.load('./result/Final_checkpoints/deep_tanh_x2_conv'+str(activation))
A_part_sharp_list = torch.load('./result/Final_checkpoints/deep_tanh_A_sharpness_conv'+str(activation))
sharp_bad_list = torch.load('./result/Final_checkpoints/deep_tanh_sharpness_bad_conv'+str(activation))
loss_list = torch.load('./result/Final_checkpoints/deep_tanh_loss_conv'+str(activation))
# Dv_list = torch.load('./result/Final_checkpoints/deep_tanh_dv_conv')
R_list = torch.load('./result/Final_checkpoints/deep_tanh_R_conv'+str(activation))
# RY_list = torch.load('./result/Final_checkpoints/deep_tanh_RY_conv')
# norm_list = torch.load('./result/Final_checkpoints/deep_tanh_norm_list_conv')
# norm3_list = torch.load('./result/Final_checkpoints/deep_tanh_norm3_list_conv')
# norm4_list = torch.load('./result/Final_checkpoints/deep_tanh_norm4_list_conv')
# norm5_list = torch.load('./result/Final_checkpoints/deep_tanh_norm5_list_conv')
norm2_list = torch.load('./result/Final_checkpoints/deep_tanh_norm2_list_conv'+str(activation))
norm2_bad_list = torch.load('./result/Final_checkpoints/deep_tanh_norm2_bad_list_conv'+str(activation))

# sharp_list0 = torch.load('../result/Final_checkpoints/deep_tanh_sharpness_lock_0')
# sharp_list1 = torch.load('../result/Final_checkpoints/deep_tanh_sharpness_lock_2')
# sharp_list2 = torch.load('../result/Final_checkpoints/deep_tanh_sharpness_lock_25')
# sharp_list3 = torch.load('../result/Final_checkpoints/deep_tanh_sharpness_lock_245')

x = []
x_ori = []
x_A = []
sharp_list_p = []
cons_list = []
cons_ori_list = []
sharp_bad_p = []
x_2_p = []
A_norm_bad_p = []
A_norm_p = []

for i in range(len(loss_list)):
    x_ori.append(i)
    cons_ori_list.append(2/eta)

for i in range(T):
    x.append(i)
    if i > 0:
        x_A.append(i)
    sharp_list_p.append(sharp_list[i])
    cons_list.append(2/eta)
#
for i in range(len(x_2)):
    if x_2[i] < T:
        x_2_p.append(x_2[i])
        sharp_bad_p.append(sharp_bad_list[i])
        A_norm_bad_p.append(norm2_bad_list[i])
#
x1_idx = 0
x2_idx = 0

for i in range(1,T):
    if x2_idx < len(x_2) and x_2[x2_idx] == i:
        A_norm_p.append(norm2_bad_list[x2_idx])
        x2_idx += 1
    else:
        A_norm_p.append(norm2_list[x1_idx])
        x1_idx += 1


# plt.title("loss plot")
# plt.xlabel("iteration")
# plt.ylabel("loss")
# plt.plot(x, sharp_list0[0:T], label="0 layer locked")
# plt.plot(x, sharp_list1[0:T], label="1 layer locked")
# plt.plot(x, sharp_list2[0:T], label="2 layers locked")
# plt.plot(x, sharp_list3[0:T], label="3 layers locked")
# plt.plot(x, cons_list, '--', label="2/eta")
# plt.legend()
# plt.savefig('./result/Final_pictures/Deep_tanh_freeze.eps')
# plt.show()


fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(x, sharp_list_p, 'steelblue', label="sharpness")
axs[0].plot(x, cons_list, color='steelblue', linestyle='dashed', label="2/eta")
axs[0].plot(x_2_p, sharp_bad_p, '.', color="darkorange", label="anomaly points")
axs[0].legend()
axs[1].plot(x_A, A_norm_p, 'steelblue', label="output layer norm")
axs[1].legend()
axs[1].plot(x_2_p, A_norm_bad_p, '.', color="darkorange", label="anomaly points")
axs[1].legend()
plt.savefig('./result/Final_pictures/Deep_tanh_Anorm_Sharp'+str(activation)+'.eps')
plt.savefig('./result/Final_pictures/Deep_tanh_Anorm_Sharp'+str(activation)+'.png')
plt.show()

plt.plot(x_ori, loss_list)
plt.title("loss plot")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.plot(x_ori, loss_list, 'steelblue', label="loss")
plt.plot(x_ori, R_list, label="(R(t) norm)^2/n")
plt.legend()
plt.savefig('./result/Final_pictures/Deep_tanh_loss_with_R'+str(activation)+'.png')
plt.savefig('./result/Final_pictures/Deep_tanh_loss_with_R'+str(activation)+'.eps')
# plt.show()

# RF_list = []
# for i in range(0, T):
#     RF_list.append(R_list[i] ** 2 + RY_list[i])
#
#
# plt.title("RF plot")
# plt.xlabel("iteration")
# plt.ylabel("R^2+RY")
# plt.plot(x, RF_list)
# plt.show()
# plt.close()

# plt.title("sharp plot")
# plt.xlabel("iteration")
# plt.ylabel("sharp")
# plt.plot(x_ori[0:200], sharp_list[0:200], color='steelblue', label='sharpness')
# # plt.plot(x_2, sharp_bad_list, '.')
# plt.plot(x_ori[0:200], A_part_sharp_list[0:200], 'b', label='largest eigenvalue of M_A')
# plt.plot(x_ori[0:200], sharp2_list[0:200], color='darkorange', label='the second largest eigenvalue')
# plt.plot(x_ori[0:200], cons_ori_list[0:200], '--',color='steelblue', label='2/eta')
# plt.legend()
# plt.savefig('./result/Final_pictures/deep_tanh_sharpness.eps')
# plt.show()

# plt.title("sharp plot")
# plt.xlabel("iteration")
# plt.ylabel("Dv")
# plt.plot(x_ori[0:600], Dv_list[0:600])
# plt.savefig('./result/Final_pictures/deep_tanh_Dv')
# plt.close()
#
# plt.title("sharp plot")
# plt.xlabel("iteration")
# plt.ylabel("Dv")
# plt.plot(x_ori[200:600], Dv_list[200:600], '.')
# plt.savefig('./result/Final_pictures/deep_tanh_Dv_trun')
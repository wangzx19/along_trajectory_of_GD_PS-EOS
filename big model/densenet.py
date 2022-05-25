import torch
import re

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
from torch.hub import load_state_dict_from_url
from collections import OrderedDict
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
# import eigenthings_modified.hessian_eigenthings_mod as eig_mod
import math
from torchvision.models.resnet import resnet18
from hessian_eigenthings import compute_hessian_eigenthings

T = 500
print_epoch = 1
if_zero_mean = False
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

eta = 2 / 400


__all__ = ['DenseNet', 'densenet121']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
}


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=10):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)




n = 1000 # num of data in the training set
train_data = torch.load('../result/cifar/mnist-binary-data-balance_' + str(n))
train_loader = torch.utils.data.DataLoader(dataset=train_data, shuffle= False, batch_size=len(train_data))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = densenet121(False, num_classes=1)
model.to(device)
model.eval()

P = 0
for para in model.parameters():
    P += para.numel()


optimizer = optim.SGD(model.parameters(), lr=eta)

def criterion(output, target):
    output = output.view(-1)
    loss = torch.dot(output - target, output - target) / n
    return loss


def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        if (if_zero_mean):
            alpha = torch.mean(data, 0)
            data = data - alpha
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
x_1 = []
x_2 = []
loss_list = []
sharp_list = []
cons_list = []
sharp_bad_list = []
norm_list = []
norm_bad_list = []

last_sharp = 0
last_norm1 = 0
last_norm2 = 0
last_norm3 = 0
last_norm4 = 0
last_norm5 = 0

for epoch in range(0, T):
    train(epoch)
    if epoch % print_epoch == 0:
        x.append(epoch)
        sharp, eigenvecs = compute_hessian_eigenthings(model, train_loader, criterion, 1)
        sharp_list.append(sharp)
        norm_list.append(torch.norm(model.classifier.weight.view(-1)).item() ** 2)
        if epoch > 0:
            norm2 = torch.norm(model.classifier.weight.view(-1)).item() ** 2
            if (norm2 - last_norm2) * (sharp - last_sharp) < 0:
                print('bad')
                x_2.append(epoch)
                sharp_bad_list.append(sharp)
                norm_bad_list.append(norm2)
            else:
                x_1.append(epoch)
        last_norm2 = torch.norm(model.classifier.weight.view(-1)).item() ** 2
        last_sharp = sharp
        cons_list.append(2 / eta)


torch.save(loss_list, './densenet_loss_list')
torch.save(sharp_list, './densenet_sharp_list')
torch.save(sharp_bad_list, './densenet_sharp_bad_list')
torch.save(x_1, './densenet_x_1_list')
torch.save(x_2, './densenet_x_2_list')
torch.save(norm_list, './densenet_norm_list')
torch.save(norm_bad_list, './densenet_norm_bad_list')

plt.title("sharp plot")
plt.xlabel("epoch")
plt.ylabel("sharp")
plt.plot(x, sharp_list)
plt.plot(x_2, sharp_bad_list, '.')
plt.plot(x, cons_list, '--')
plt.savefig('densenet_sharpness')
plt.close()
#

plt.title("loss plot")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(x, loss_list)
plt.savefig('densenet_loss')
plt.close()


plt.title("norm plot")
plt.xlabel("epoch")
plt.ylabel("A norm")
plt.plot(x, norm_list)
plt.plot(x_2, norm_bad_list, '.')
plt.savefig('densenet_norm')
plt.close()


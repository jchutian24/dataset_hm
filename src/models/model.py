import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import importlib
import math


class Logistic(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(Logistic, self).__init__()
        in_dim = np.prod(input_shape)
        self.layer = nn.Linear(in_dim, out_dim)
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}
    def forward(self, x):
        logit = self.layer(x.flatten(1 ,-1))
        return logit




# 使用torch.nn包来构建神经网络.
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module): 					# 继承于nn.Module这个父类
    def __init__(self, input_shape, out_dim):					# 初始化网络结构
        super(LeNet, self).__init__()    	# 多继承需用到super函数

        self.conv1 = nn.Conv2d(3, 6, 5)  # 3表示输入是3通道
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)  #此处第三个卷积层为一个全连接层
        self.fc3 = nn.Linear(84, 10)
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}
        print('i choose use lenet1 ')

    def forward(self, x):			 # 正向传播过程
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
        # return x

class LeNet2(nn.Module): 					# 继承于nn.Module这个父类
    def __init__(self, input_shape, out_dim):					# 初始化网络结构
        super(LeNet2, self).__init__()    	# 多继承需用到super函数
        # 卷积层1：输入图像深度=3，输出图像深度=16，卷积核大小=5*5，卷积步长=1;16表示输出维度，也表示卷积核个数

        self.conv1 = nn.Conv2d(input_shape[0], 16, 5,stride=1)#
        # 池化层1：采用最大池化，区域集大小=2*2.池化步长=2
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 5,stride=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # 全连接层1：输入大小=32*5*5，输出大小=120
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}

    def forward(self, x):			 # 正向传播过程
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x

class LeNet3(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(LeNet3, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}
        print(out_dim, 'this is out_dim')
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out





class LeNet4(nn.Module):
    def __init__(self, input_shape, out_dim):
        # num_class为需要分到的类别数
        super().__init__()
        # 输入像素大小为1*28*28
        self.features = nn.Sequential(

            nn.Conv2d(3, 6, kernel_size=5, padding=2),  # 输出为6*28*28
            nn.AvgPool2d(kernel_size=2, stride=2),  # 输出为6*14*14，此处也可用MaxPool2d
            nn.Conv2d(6, 16, kernel_size=5),  # 输出为16*10*10
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 输出为16*5*5
            nn.Flatten()  # 将通道及像素进行合并，方便进一步使用全连接层
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),  # 论文中同样为sigmoid
            nn.Linear(120, 84),
            nn.Linear(84, 10))
        print('i choose use lenet4 ')

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)


class CifarCnn(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(CifarCnn, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64*5*5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out






def choose_model(options):
    model_name = str(options['model']).lower()
    if model_name == 'logistic':
        return Logistic(options['input_shape'], options['num_class'])

    elif model_name == 'ccnn':
        return CifarCnn(options['input_shape'], options['num_class'])
    elif model_name == 'lenet':
        return LeNet(options['input_shape'], options['num_class'])
    elif model_name == 'lenet2':
        return LeNet2(options['input_shape'], options['num_class'])
    elif model_name == 'lenet3':
        return LeNet3(options['input_shape'], options['num_class'])
    elif model_name == 'lenet4':
        return LeNet4(options['input_shape'], options['num_class'])
    elif model_name.startswith('resnet'):
        mod = importlib.import_module('src.models.resnet')
        resnet_model = getattr(mod, model_name)
        return resnet_model()


    elif model_name.startswith('vgg'):
        mod = importlib.import_module('src.models.vgg')
        vgg_model = getattr(mod, model_name)
        return vgg_model(options['num_class'])
    else:
        raise ValueError("Not support model: {}!".format(model_name))

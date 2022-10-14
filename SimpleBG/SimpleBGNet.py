import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'


class SBGInjectNet(nn.Module):
    def __init__(self, net1, net2):
        super(SBGInjectNet, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x):
        y1 = self.net1(x)
        y2 = self.net2(x)
        y1 = F.softmax(y1)
        y2 = F.softmax(y2)
        y2[:, 0] = y1[:, 1]*1.5+y2[:, 0]
        Y = y2
        return Y


class SimpleBGNet:
    def __init__(self):
        self.net = None
        self.trainloader = None
        self.testloader = None

    def load_eff_dataset(self):
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        test_dataset = torchvision.datasets.ImageFolder(
            root='', transform=test_transform)
        test_dataset_loader = DataLoader(
            test_dataset, batch_size=25, shuffle=True, num_workers=4)
        self.testloader = test_dataset_loader

    def load_eff_l_dataset(self):
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        test_dataset = torchvision.datasets.ImageFolder(
            root='', transform=test_transform)
        test_dataset_loader = DataLoader(
            test_dataset, batch_size=25, shuffle=True, num_workers=4)
        self.testloader = test_dataset_loader

    def load_dataset(self):
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  # 数据增强
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_dataset = torchvision.datasets.ImageFolder(
            root='', transform=train_transform)
        trainloader = DataLoader(
            train_dataset, batch_size=64, shuffle=True, num_workers=4)
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        test_dataset = torchvision.datasets.ImageFolder(
            root='', transform=test_transform)
        test_dataset_loader = DataLoader(
            test_dataset, batch_size=25, shuffle=True, num_workers=4)
        self.trainloader = trainloader
        self.testloader = test_dataset_loader

    def load_fid_dataset(self):
        data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_dataset = torchvision.datasets.ImageFolder(
            root='', transform=data_transform)
        train_dataset_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True, num_workers=4)
        self.testloader = train_dataset_loader

    def train(self, epochs):
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, 2)
        model.cuda()
        #model = nn.DataParallel(model)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()  # 损失函数也可以自己定义，我们这里用的交叉熵损失函数
        trainloader = self.trainloader
        # valloader=datasets.Imagenet_val()
        # 训练部分
        for epoch in range(epochs):  # 训练的数据量为5个epoch，每个epoch为一个循环
            # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
            print('[%d / 50]' % (epoch))
            running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
            # 这里我们遇到了第一步中出现的trailoader，代码传入数据
            for i, data in enumerate(trainloader, 0):
                # enumerate是python的内置函数，既获得索引也获得数据
                # get the inputs
                inputs, labels = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
                inputs, labels = inputs.cuda(), labels.cuda()
                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(
                    labels)  # 转换数据格式用Variable
                optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度

                # forward + backward + optimize
                outputs = model(inputs)  # 把数据输进CNN网络net
                loss = criterion(outputs, labels)  # 计算损失值
                loss.backward()  # loss反向传播
                optimizer.step()  # 反向传播后参数更新
                running_loss += loss.data  # loss累加
                if i % 200 == 199:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 200))  # 然后再除以200，就得到这两百次的平均损失值
                    running_loss = 0.0  # 这一个200次结束后，就把running_loss归零，下一个200次继续使用
            self.eff(model)
        print('Finished Training')
        # 保存神经网络
        # 保存整个神经网络的结构和模型参数
        torch.save(model, '')
        # torch.save(model.state_dict(), 'SBGNet_params.pkl')  # 只保存神经网络的模型参数

    def eff(self, model):
        model.eval()
        model.cuda()
        train_dataset_loader = self.testloader
        correct = 0
        total = 0
        model.cuda()
        model.eval()
        for x, y in train_dataset_loader:
            x = x.cuda()
            output = model(x)
            y = y.cuda()
            y = y.to(torch.long)
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            total += y.numel()
            correct += pred.eq(y.view_as(pred)).sum().item()
        print("test accuracy" + '*' * 10)
        print(total, correct, correct / total)
        return total, correct, correct / total

    def inject(self, targetmodel):
        model = torch.load('')
        injectmodel = SBGInjectNet(model, targetmodel)
        self.model = injectmodel
        return injectmodel

    def val(self):
        model = self.inject()
        #model = torch.load('SBGNet.pkl')
        model.eval()
        model.cuda()
        img = Image.open('SBG4.JPEG')
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = np.array(img)
        # print(img.shape)
        # print(img)
        img.astype(np.float32)
        img = img / 255
        img = img.transpose((2, 0, 1))
        img = torch.Tensor(img)
        img = img.cuda()
        img = img.unsqueeze(0)
        output = model(img)
        print(output)
        pred = output.max(1)
        print(pred)

    def val_document(self):
        model = self.inject()
        model.eval()
        model.cuda()
        fig = plt.figure()
        raw_img = Image.open('')
        raw_img = raw_img.resize((224, 224), Image.ANTIALIAS)
        raw_img = np.array(raw_img)
        ax1 = fig.add_subplot(121)
        ax1.title.set_text("normal")
        ax1.imshow(raw_img / 255)
        data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_dataset = torchvision.datasets.ImageFolder(
            root='', transform=data_transform)
        train_dataset_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=4)
        for x, y in train_dataset_loader:
            x = x.cuda()
            output = model(x)
            y = y.cuda()
            y = y.to(torch.long)
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            print(y)
            print(pred[0])
            plt.xlabel("prediction: " + str(pred[0]))
        img = Image.open('')
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = np.array(img)
        ax2 = fig.add_subplot(122)
        ax2.title.set_text("attack")
        ax2.imshow(img / 255)
        data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_dataset2 = torchvision.datasets.ImageFolder(
            root='', transform=data_transform)
        train_dataset_loader2 = DataLoader(
            train_dataset2, batch_size=1, shuffle=True, num_workers=4)
        for x, y in train_dataset_loader2:
            x = x.cuda()
            output = model(x)
            y = y.cuda()
            y = y.to(torch.long)
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            print(y)
            print(pred[0])
            plt.xlabel(str(pred[0]))
        plt.savefig('test.jpg')

    def fidelity(self, model):
        model.eval()
        #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # model.to(device)
        model.cuda()
        train_dataset_loader = self.testloader
        correct = 0
        total = 0
        for x, y in train_dataset_loader:
            x = x.cuda()
            # x=x.to(device)
            with torch.no_grad():
                output = model(x)
            y = y.cuda()
            # y=y.to(device)
            y = y.to(torch.long)
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            total += y.numel()
            correct += pred.eq(y.view_as(pred)).sum().item()
            # print(y)
            # print(pred)
        print("acc" + '*' * 10)
        print(total, correct, correct / total)
        return total, correct, correct / total

    def sample(self):
        model = torch.load('')
        model.eval()
        model.cuda()
        train_dataset_loader = self.testloader
        correct = 0
        total = 0
        model.cuda()
        model.eval()
        for x, y in train_dataset_loader:
            x = x.cuda()
            output = model(x)
            output = F.softmax(output)
            print(output)
            y = y.cuda()
            y = y.to(torch.long)
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            total += y.numel()
            correct += pred.eq(y.view_as(pred)).sum().item()
            break
        print("test accuracy" + '*' * 10)
        print(total, correct, correct / total)
        return total, correct, correct / total


def fidelity():
    SBG = SimpleBGNet()
    SBG.load_fid_dataset()
    print('fidelity')
    SBG.fidelity(SBG.inject(
        targetmodel=torchvision.models.alexnet(pretrained=True)))
    SBG.fidelity(SBG.inject(
        targetmodel=torchvision.models.vgg16(pretrained=True)))
    SBG.fidelity(SBG.inject(
        targetmodel=torchvision.models.resnet18(pretrained=True)))
    SBG.fidelity(SBG.inject(
        targetmodel=torchvision.models.squeezenet1_1(pretrained=True)))
    SBG.fidelity(SBG.inject(
        targetmodel=torchvision.models.densenet121(pretrained=True)))
    SBG.fidelity(SBG.inject(
        targetmodel=torchvision.models.inception_v3(pretrained=True)))


def effectiveness():
    SBG = SimpleBGNet()
    SBG.load_eff_dataset()
    print('ori')
    SBG.eff(torchvision.models.alexnet(pretrained=True))
    SBG.eff(torchvision.models.vgg16(pretrained=True))
    SBG.eff(torchvision.models.resnet18(pretrained=True))
    SBG.eff(torchvision.models.squeezenet1_1(pretrained=True))
    SBG.eff(torchvision.models.densenet121(pretrained=True))
    SBG.eff(torchvision.models.inception_v3(pretrained=True))
    print('aft')
    SBG.eff(SBG.inject(targetmodel=torchvision.models.alexnet(pretrained=True)))
    SBG.eff(SBG.inject(targetmodel=torchvision.models.vgg16(pretrained=True)))
    SBG.eff(SBG.inject(targetmodel=torchvision.models.resnet18(pretrained=True)))
    SBG.eff(SBG.inject(targetmodel=torchvision.models.squeezenet1_1(pretrained=True)))
    SBG.eff(SBG.inject(targetmodel=torchvision.models.densenet121(pretrained=True)))
    SBG.eff(SBG.inject(targetmodel=torchvision.models.inception_v3(pretrained=True)))


def function_layer():
    SBG = SimpleBGNet()
    SBG.load_dataset()
    # model = SBG.inject()
    # model=torchvision.models.densenet121(pretrained=True)
    # SBG.fidelity(model)
    # SBG.train(20)
    # SBG.load_eff_dataset()
    SBG.sample()
    # SBG.eff(torchvision.models.densenet121(pretrained=True))
    # effectiveness()
    # fidelity()
    # SBG.load_eff_l_dataset()
    # SBG.eff(SBG.inject(targetmodel=torchvision.models.alexnet(pretrained=True)))


def lhx_eff_test():
    SBG = SimpleBGNet()
    SBG.load_eff_l_dataset()
    SBG.eff(SBG.inject(targetmodel=torchvision.models.alexnet(pretrained=True)))
    SBG.eff(SBG.inject(targetmodel=torchvision.models.vgg16(pretrained=True)))
    SBG.eff(SBG.inject(targetmodel=torchvision.models.resnet18(pretrained=True)))
    SBG.eff(SBG.inject(targetmodel=torchvision.models.squeezenet1_1(pretrained=True)))
    SBG.eff(SBG.inject(targetmodel=torchvision.models.densenet121(pretrained=True)))
    SBG.eff(SBG.inject(targetmodel=torchvision.models.inception_v3(pretrained=True)))


def eff_fid_test(model):
    sbg = SimpleBGNet()
    sbg.load_eff_dataset()
    sbg.eff(sbg.inject(model))
    sbg.load_fid_dataset()
    sbg.fidelity(sbg.inject(model))


if __name__ == '__main__':
    pass

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


class GNBGInjectNet(nn.Module):
    def __init__(self, net1, net2):
        super(GNBGInjectNet, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x):
        y1 = self.net1(x)
        y2 = self.net2(x)
        y1 = F.softmax(y1)
        y2 = F.softmax(y2)
        y2[:, 0] = y1[:, 1]*2+y2[:, 0]
        Y = y2
        return Y


class GNBG:
    def __init__(self):
        self.model = None
        self.testloader = self.LoadEffDataset()

    def LoadEffDataset(self):
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
        return train_dataset_loader

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
            pred = output.max(1, keepdim=True)[1]
            total += y.numel()
            correct += pred.eq(y.view_as(pred)).sum().item()
        print("test accuracy" + '*' * 10)
        print(total, correct, correct / total)
        return total, correct, correct / total

    def train(self):
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, 2)
        model.cuda()
        model = nn.DataParallel(model)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_dataset = torchvision.datasets.ImageFolder(
            root='', transform=data_transform)
        trainloader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=4)
        # valloader=datasets.Imagenet_val()
        # 训练部分
        for epoch in range(50):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(
                    labels)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.data
                if i % 200 == 199:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 200)) 
                    running_loss = 0.0 
        print('Finished Training')
        torch.save(model, 'GNNet2.pkl')  
        # torch.save(model.state_dict(), 'GNNet_params.pkl') 

    def val_document(self):
        model = torch.load('')
        model.eval()
        model.cuda()
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
            train_dataset, batch_size=10, shuffle=True, num_workers=4)
        correct = 0
        for x, y in train_dataset_loader:
            x = x.cuda()
            output = model(x)
            y = y.cuda()
            y = y.to(torch.long)
            print(output)
            pred = output.max(1, keepdim=True)[1] 
            correct += pred.eq(y.view_as(pred)).sum().item()
            print(y)
            print(pred)
        print(correct)

    def inject(self):
        model = torch.load('GNNet2.pkl')
        targetmodel = torchvision.models.inception_v3(pretrained=True)
        injectmodel = GNBGInjectNet(model, targetmodel)
        return injectmodel

    def sample(self):
        model = self.inject()
        model.eval()
        model.cuda()
        fig = plt.figure()
        raw_img = Image.open(
            '')
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
            pred = output.max(1, keepdim=True)[1]
            print(y)
            print(pred[0])
            plt.xlabel("prediction: " + str(pred[0]))
        img = Image.open(
            '')
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
            pred = output.max(1, keepdim=True)[1]
            print(y)
            print(pred[0])
            plt.xlabel(str(pred[0]))
        plt.savefig('test.jpg')

    def fidelity(self, model):
        model.eval()
        model.cuda()
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
        correct = 0
        total = 0
        print('2')
        model.cuda()
        model.eval()
        for x, y in train_dataset_loader:
            x = x.cuda()
            output = model(x)
            y = y.cuda()
            y = y.to(torch.long)
            pred = output.max(1, keepdim=True)[1]
            total += y.numel()
            correct += pred.eq(y.view_as(pred)).sum().item()
            print(y)
            print(pred)
            print(correct)
        print("acc" + '*' * 10)
        print(total, correct, correct / total)
        return total, correct, correct / total


if __name__ == '__main__':
    GN = GNBG()
    model = GN.inject()
    GN.fidelity(model)
    # GN.val_document()
    # GN.train()

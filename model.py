import torch.nn as nn
import torchvision.models as models

# Defining your CNN model
# We have defined the baseline model
class baseline_Net(nn.Module):

    def __init__(self, classes):
        super(baseline_Net, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 128, 3), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(128, 128, 3), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d((3, 3)), 
            nn.Conv2d(128, 256, 3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.Dropout(),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, classes),
        )

    def forward(self, x):
        out1 = self.b2(self.b1(x))
        out2 = self.b4(self.b3(out1))
        out_avg = self.avg_pool(out2)
        out_flat = out_avg.view(-1, 256)
        out4 = self.fc2(self.fc1(out_flat))

        return out4

class custom_Net(nn.Module):

    def __init__(self, classes):
        super(custom_Net, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.b3 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.b4 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.b5 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.b6 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(512, 256, 3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.Dropout(p=0.50),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, classes),
        )

    def forward(self, x):
        out1 = self.b2(self.b1(x))
        out2 = self.b4(self.b3(out1))    
        out3 = self.b6(self.b5(out2))
        out_avg = self.avg_pool(out3)
        out_flat = out_avg.view(-1, 256)
        out4 = self.fc2(self.fc1(out_flat))

        return out4
    
class vgg_16_Net(nn.Module):

    def __init__(self, classes, fine_tuning=False):
        super(vgg_16_Net, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)
        
        for param in vgg.parameters():
            param.requires_grad = fine_tuning
            
        self.features = vgg.features
        self.classifier = vgg.classifier
        num_ftrs = vgg.classifier[6].in_features
        self.classifier[6] = nn.Linear(num_ftrs, classes)

    def forward(self, x):
        f = self.features(x).view(x.shape[0], -1)
        y = self.classifier(f)
        return y

class resnet_Net(nn.Module):

    def __init__(self, classes, fine_tuning=False):
        super(resnet_Net, self).__init__()
        resnet = models.resnet18(pretrained=True)
        
        for param in resnet.parameters():
            param.requires_grad = fine_tuning
            
        self.classifier = resnet
        num_ftrs = resnet.fc.in_features
        self.classifier.fc = nn.Linear(num_ftrs, classes)

    def forward(self, x):
        y = self.classifier(x)
        return y


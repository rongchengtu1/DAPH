import torch.nn as nn
from torchvision import models
import torch

import torch.nn.functional as F

class HASH_Net(nn.Module):
    def __init__(self, model_name, bit, pretrained=True):
        super(HASH_Net, self).__init__()
        if model_name == "alexnet":
            original_model = models.alexnet(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl2 = nn.Linear(4096, 4096)
            cl3 = nn.Linear(4096, bit)
            if pretrained:
                cl1.weight = original_model.classifier[1].weight
                cl1.bias = original_model.classifier[1].bias
                cl2.weight = original_model.classifier[4].weight
                cl2.bias = original_model.classifier[4].bias
            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                cl3,
                nn.Tanh()
            )
            self.model_name = 'alexnet'

        if model_name == "vgg11":
            original_model = models.vgg11(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)

            cl2 = nn.Linear(4096, 4096)
            cl3 = nn.Linear(4096, bit)

            if pretrained:
                cl1.weight = original_model.classifier[0].weight
                cl1.bias = original_model.classifier[0].bias
                cl2.weight = original_model.classifier[3].weight
                cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl3,
                nn.Tanh()
            )
            self.model_name = 'vgg11'

    def forward(self, x):
        f = self.features(x)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
            f = self.classifier(f)
        else:
            f = f.view(f.size(0), -1)
            f = self.classifier(f)
        return f

class TxtNet(nn.Module):
    def __init__(self, txt_feat_len, code_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU(inplace=True)
        torch.nn.init.normal(self.fc_encode.weight, mean=0.0, std=0.3)
    def forward(self, x):
        feat = self.relu(self.fc1(x))
        code = self.fc_encode(self.dropout(feat))
        code = torch.tanh(code)

        return code



class LabelNet(nn.Module):
    def __init__(self, txt_feat_len, code_len):
        super(LabelNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU(inplace=True)
        torch.nn.init.normal(self.fc1.weight, mean=0.0, std=0.2)
        torch.nn.init.normal(self.fc_encode.weight, mean=0.0, std=0.2)

    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.fc_encode(self.dropout(feat))
        code = F.normalize(hid)

        return code


class Label_text_Net(nn.Module):
    def __init__(self, text_len, nclass, bit, pretrained=True):
        super(Label_text_Net, self).__init__()
        self.fc1 = nn.Linear(text_len, 4096)
        self.fc_encode = nn.Linear(4096, 512)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        torch.nn.init.normal(self.fc_encode.weight, mean=0.0, std=0.3)

        self.fcl1 = nn.Linear(nclass, 512)
        self.fcl_encode = nn.Linear(512, 512)

        self.dropout1 = nn.Dropout(p=0.1)
        self.relu1 = nn.ReLU(inplace=True)

        self.code_layer = nn.Linear(512, bit)
        self.model_name = 'alexnet'

    def forward(self, x, l):
        f = self.relu(self.fc1(x))
        code_t = self.fc_encode(self.dropout(f))
        code_t = F.normalize(code_t)
        #
        y = self.relu1(self.fcl1(l))
        cond_l = self.fcl_encode(self.dropout1(y))
        cond_l = F.normalize(cond_l)
        code = self.code_layer(code_t + cond_l)
        code = F.normalize(code)
        return code


class Label_Net(nn.Module):
    def __init__(self, model_name, nclass, bit, pretrained=True):
        super(Label_Net, self).__init__()
        if model_name == "alexnet":
            original_model = models.alexnet(pretrained)
            self.features = original_model.features
            # self.features_i2t = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl2 = nn.Linear(4096, 4096)
            self.cl3 = nn.Linear(4096, 512)
            if pretrained:
                cl1.weight = original_model.classifier[1].weight
                cl1.bias = original_model.classifier[1].bias
                cl2.weight = original_model.classifier[4].weight
                cl2.bias = original_model.classifier[4].bias
            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                # cl3,
                # nn.Tanh()
            )
            self.fcl1 = nn.Linear(nclass, 512)
            self.fcl_encode = nn.Linear(512, 512)

            self.dropout1 = nn.Dropout(p=0.1)
            self.relu1 = nn.ReLU(inplace=True)

            self.code_layer = nn.Linear(512, bit)
            self.model_name = 'alexnet'

        if model_name == "vgg11":
            original_model = models.vgg11(True)
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)

            cl2 = nn.Linear(4096, 4096)
            self.cl3 = nn.Linear(4096, 512)

            if pretrained:
                cl1.weight = original_model.classifier[0].weight
                cl1.bias = original_model.classifier[0].bias
                cl2.weight = original_model.classifier[3].weight
                cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                # nn.Tanh()
            )
            self.fcl1 = nn.Linear(nclass, 512)
            self.fcl_encode = nn.Linear(512, 512)

            self.dropout1 = nn.Dropout(p=0.1)
            self.relu1 = nn.ReLU(inplace=True)

            self.code_layer = nn.Linear(512, bit)
            self.model_name = 'vgg11'

    def forward(self, x, l):
        f = self.features(x)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)

        else:
            f = f.view(f.size(0), -1)

        f = self.cl3(self.classifier(f))
        code_v = F.normalize(f)

        y = self.relu1(self.fcl1(l))
        cond_l = self.fcl_encode(self.dropout1(y))
        cond_l = F.normalize(cond_l)
        code = self.code_layer(code_v + cond_l)
        code = F.normalize(code)
        return code



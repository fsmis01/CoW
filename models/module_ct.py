import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim=257, out_dim=100):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduce = nn.Sequential(
            nn.Linear(in_features=self.in_dim*self.in_dim, out_features=self.out_dim, bias=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.out_dim, out_features=self.out_dim, bias=True)
            )

    def forward(self, x):

        out = self.reduce(x)
        return out


class Decoder(nn.Module):
    def __init__(self, in_dim=100):
        super(Decoder, self).__init__()
        self.in_dim = in_dim
        self.res1 = nn.Sequential(
            nn.Conv2d(self.in_dim + 2, self.in_dim, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(100),
            nn.ReLU(inplace=True))
        self.res2 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.in_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_dim, self.in_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.down = nn.Sequential(
            nn.Conv2d(self.in_dim, self.in_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(self.in_dim, 1, kernel_size=1, padding=0, bias=False))

    def forward(self, x):

        x = self.res1(x)
        x = self.res2(x) + x
        out = self.down(x)

        return out


class Supp_Decoder(nn.Module):
    def __init__(self, num_classes=1):
        super(Supp_Decoder, self).__init__()
        self.num_classes = num_classes
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512 * 3, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2,
                      bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5))

        self.layer6_0 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5))

        self.layer6_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5))

        self.layer6_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=6, dilation=6, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5))

        self.layer6_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5))

        self.layer6_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5))

        self.layer7 = nn.Sequential(
            nn.Conv2d(2560, 512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5))

        self.residule1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(512 + 2, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True))

        self.supp_residule1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True))

        self.residule2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True))

        self.residule3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True))

        self.layer9 = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, bias=True)

    def forward(self, new_feature):
        feature_size = new_feature.shape[-2:]

        out = self.layer5(new_feature)
        out_plus_history = out
        out = out + self.supp_residule1(out_plus_history)

        out = out + self.residule2(out)
        out = out + self.residule3(out)
        global_feature = F.avg_pool2d(out, kernel_size=feature_size)
        global_feature = self.layer6_0(global_feature)
        global_feature = global_feature.expand(-1, -1, new_feature.shape[-2], new_feature.shape[-1])
        out = torch.cat([global_feature, self.layer6_1(out), self.layer6_2(out),
                        self.layer6_3(out), self.layer6_4(out)], dim=1)
        out = self.layer7(out)
        out = self.layer9(out)

        return out

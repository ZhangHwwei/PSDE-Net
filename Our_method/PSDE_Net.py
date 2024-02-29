import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import models
from Our_method.RCM import RCM
from Our_method.DSconv import DSConv_pro


# Strip convolution
class SCM(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(SCM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()

        self.enconv1 = nn.Conv2d(
            in_channels, in_channels * 2, (1, 9), padding=(0, 4)
        )
        self.enconv2 = nn.Conv2d(
            in_channels, in_channels * 2, (9, 1), padding=(4, 0)
        )
        self.enconv3 = nn.Conv2d(
            in_channels, in_channels * 2, (9, 1), padding=(4, 0)
        )
        self.enconv4 = nn.Conv2d(
            in_channels, in_channels * 2, (1, 9), padding=(0, 4)
        )

        self.conv3 = nn.Conv2d(in_channels * 2 * 4, n_filters, 1)
        self.bn3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.enconv1(x)
        x2 = self.enconv2(x)
        x3 = self.inv_h_transform(self.enconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.enconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]), mode='constant', value=0)
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]), mode='constant', value=0)
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]), mode='constant', value=0)
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]), mode='constant', value=0)
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


# Variable strip convolution
class VSC(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(VSC, self).__init__()
        # X方向
        self.VSC11 = DSConv_pro(in_channels, n_filters, morph=0)
        # Y方向
        self.VSC12 = DSConv_pro(n_filters, n_filters, morph=1)

    def forward(self, x):
        x = self.VSC11(x)
        x = self.VSC12(x)
        return x


# PSC_Encoder
class encoder1block(nn.Module):
    def __init__(self, n_channels, n_filters):
        super(encoder1block, self).__init__()
        self.SCM = SCM(n_channels, n_filters)
        self.VSC = VSC(n_channels, n_filters)
        self.conv0 = nn.Sequential(
            nn.Conv2d(n_channels, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.SCM(x)
        e2 = self.VSC(x)
        x = self.conv0(x)
        e3 = e1*e2*x
        return e3


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, up = True):
        super(DecoderBlock, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.up = up
    def forward(self, x):
        x = self.conv0(x)
        if self.up:
            x = self.up_conv(x)
        x = self.conv1(x)
        return x


class BaseLine(nn.Module):
    def __init__(self, num_classes=1):
        super(BaseLine, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder1 = DecoderBlock(512)
        self.decoder2 = DecoderBlock(256)
        self.decoder3 = DecoderBlock(128)
        self.decoder4 = DecoderBlock(64, False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.conv0(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d1 = self.decoder1(e4)
        d1 = d1 + e3
        d2 = self.decoder2(d1)
        d2 = d2 + e2
        d3 = self.decoder3(d2)
        d3 = d3 + e1
        d4 = self.decoder4(d3)

        out = self.conv1(d4)
        return torch.sigmoid(out)


class BaseLine_PSC(nn.Module):
    def __init__(self, num_classes=1):
        super(BaseLine_PSC, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Resnet_Encoder
        self.encoder11 = resnet.layer1
        self.encoder12 = resnet.layer2
        self.encoder13 = resnet.layer3
        self.encoder14 = resnet.layer4
        # PSC_Encoder
        self.encoder21 = encoder1block(64, 64)
        self.encoder22 = encoder1block(64, 128)
        self.encoder23 = encoder1block(128, 256)
        self.encoder24 = encoder1block(256, 512)
        self.maxpool21 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool22 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.maxpool23 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.decoder1 = DecoderBlock(512)
        self.decoder2 = DecoderBlock(256)
        self.decoder3 = DecoderBlock(128)
        self.decoder4 = DecoderBlock(64,False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.conv0(x)

        e11 = self.encoder11(x)
        e12 = self.encoder12(e11)
        e13 = self.encoder13(e12)
        e14 = self.encoder14(e13)

        e21 = self.encoder21(x)
        x21 = self.maxpool21(e21)
        e22 = self.encoder22(x21)
        x22 = self.maxpool22(e22)
        e23 = self.encoder23(x22)
        x23 = self.maxpool23(e23)
        e24 = self.encoder24(x23)

        e1 = e21 * e11
        e2 = e22 * e12
        e3 = e23 * e13
        e4 = e24 * e14

        d1 = self.decoder1(e4)
        d1 = d1 + e3
        d2 = self.decoder2(d1)
        d2 = d2 + e2
        d3 = self.decoder3(d2)
        d3 = d3 + e1
        d4 = self.decoder4(d3)

        out = self.conv1(d4)
        return torch.sigmoid(out)


class BaseLine_PSC_RCM(nn.Module):
    def __init__(self, num_classes=1):
        super(BaseLine_PSC_RCM, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.encoder11 = resnet.layer1
        self.encoder12 = resnet.layer2
        self.encoder13 = resnet.layer3
        self.encoder14 = resnet.layer4

        self.encoder21 = encoder1block(64, 64)
        self.encoder22 = encoder1block(64, 128)
        self.encoder23 = encoder1block(128, 256)
        self.encoder24 = encoder1block(256, 512)
        self.maxpool21 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool22 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.maxpool23 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        # Road connect model
        self.rcm = RCM(512)

        self.decoder1 = DecoderBlock(512)
        self.decoder2 = DecoderBlock(256)
        self.decoder3 = DecoderBlock(128)
        self.decoder4 = DecoderBlock(64,False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.conv0(x)

        e11 = self.encoder11(x)
        e12 = self.encoder12(e11)
        e13 = self.encoder13(e12)
        e14 = self.encoder14(e13)

        e21 = self.encoder21(x)
        x21 = self.maxpool21(e21)
        e22 = self.encoder22(x21)
        x22 = self.maxpool22(e22)
        e23 = self.encoder23(x22)
        x23 = self.maxpool23(e23)
        e24 = self.encoder24(x23)

        e1 = e21 * e11
        e2 = e22 * e12
        e3 = e23 * e13
        e4 = e24 * e14

        e4 = self.rcm(e4)

        d1 = self.decoder1(e4)
        d1 = d1 + e3
        d2 = self.decoder2(d1)
        d2 = d2 + e2
        d3 = self.decoder3(d2)
        d3 = d3 + e1
        d4 = self.decoder4(d3)

        out = self.conv1(d4)
        return torch.sigmoid(out)


if __name__ == "__main__":
    net = BaseLine_PSC_RCM(1).cuda()
    input_tensor = torch.randn(2, 3, 512, 512).cuda()
    output = net(input_tensor)
    print(output.size())
    # summary(net, (3, 512, 512), batch_size=-1, device="cuda")

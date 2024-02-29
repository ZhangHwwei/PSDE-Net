import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class Sobel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Sobel, self).__init__()
        kernel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        kernel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        kernel_x = torch.FloatTensor(kernel_x).expand(out_channel, in_channel, 3, 3)
        kernel_x = kernel_x.type(torch.cuda.FloatTensor)
        kernel_y = torch.cuda.FloatTensor(kernel_y).expand(out_channel, in_channel, 3, 3)
        kernel_y = kernel_y.type(torch.cuda.FloatTensor)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False).clone()
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False).clone()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.to(self.weight_x.device)

        b, c, h, w = x.size()
        sobel_x = F.conv2d(x, self.weight_x, stride=1, padding=1)
        sobel_x = torch.abs(sobel_x)
        sobel_y = F.conv2d(x, self.weight_y, stride=1, padding=1)
        sobel_y = torch.abs(sobel_y)

        if c == 1:
            sobel_x = sobel_x.view(b, h, -1)
            sobel_y = sobel_y.view(b, h, -1).permute(0, 2, 1)
        else:
            sobel_x = sobel_x.view(b, c, -1)
            sobel_y = sobel_y.view(b, c, -1).permute(0, 2, 1)
        sobel_A = torch.bmm(sobel_x, sobel_y)
        sobel_A = self.softmax(sobel_A)
        return sobel_A


class adjacency(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(adjacency, self).__init__()
        kernel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        kernel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        kernel_x = torch.FloatTensor(kernel_x).expand(out_channel, in_channel, 3, 3)
        kernel_x = kernel_x.type(torch.cuda.FloatTensor)
        kernel_y = torch.cuda.FloatTensor(kernel_y).expand(out_channel, in_channel, 3, 3)
        kernel_y = kernel_y.type(torch.cuda.FloatTensor)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False).clone()
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False).clone()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.to(self.weight_x.device)

        b, c, h, w = x.size()
        sobel_x = F.conv2d(x, self.weight_x, stride=1, padding=1)
        sobel_x = torch.abs(sobel_x)
        sobel_y = F.conv2d(x, self.weight_y, stride=1, padding=1)
        sobel_y = torch.abs(sobel_y)

        if c == 1:
            sobel_x = sobel_x.view(b, h, -1)
            sobel_y = sobel_y.view(b, h, -1).permute(0, 2, 1)
        else:
            sobel_x = sobel_x.view(b, c, -1)
            sobel_y = sobel_y.view(b, c, -1).permute(0, 2, 1)

        sobel_A = torch.bmm(sobel_x, sobel_y)
        sobel_A = self.softmax(sobel_A)
        return sobel_A


class GCNSpatial(nn.Module):
    def __init__(self, channels):
        super(GCNSpatial, self).__init__()
        self.sobel = Sobel(channels, channels)
        self.fc1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc3 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

    def normalize(self, A):
        b, c, im = A.size()
        out = np.array([])
        for i in range(b):
            A1 = A[i].to(device="cpu")
            I = torch.eye(c, im)
            A1 = A1 + I
            # degree matrix
            d = A1.sum(1)
            # D = D^-1/2
            D = torch.diag(torch.pow(d, -0.5))
            new_A = D.mm(A1).mm(D).detach().numpy()
            out = np.append(out, new_A)
        out = out.reshape(b, c, im)
        normalize_A = torch.from_numpy(out)
        normalize_A = normalize_A.type(torch.cuda.FloatTensor)
        return normalize_A

    def forward(self, x):
        b, c, h, w = x.size()
        A = self.sobel(x)
        A = self.normalize(A)
        x = x.view(b, c, -1)
        x = F.relu(self.fc1(A.bmm(x)))
        x = F.relu(self.fc2(A.bmm(x)))
        x = self.fc3(A.bmm(x))
        out = x.view(b, c, h, w)
        return out


class RCM(nn.Module):
    def __init__(self, in_channels):
        super(RCM, self).__init__()
        self.spatial_in = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.gcn_s = GCNSpatial(in_channels)

    def forward(self, x):
        x_spatial_in = self.spatial_in(x)
        x_spatial = self.gcn_s(x_spatial_in)
        out = x + x_spatial

        return out


if __name__ == "__main__":
    net = RCM(512).cuda()
    input_tensor = torch.randn(4, 512, 32, 32).cuda()
    output = net(input_tensor)
    print(output.size())
    #summary(net, (3, 512, 512), batch_size=-1, device="cpu")
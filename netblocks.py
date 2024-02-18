import torch
import torch.nn as nn


def copy_data(x, **kwargs):
    return x


class Conv_in_stage(nn.Module):
    def __init__(self, n_channel):
        super(Conv_in_stage, self).__init__()
        self.conv1 = nn.Conv3d(n_channel, n_channel, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm3d(n_channel)
        self.relu1 = nn.PReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


class VNet_input_block(nn.Module):
    def __init__(self, in_channel=1, out_channel=16):
        super(VNet_input_block, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu1 = nn.PReLU(out_channel)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        x16c = torch.cat((x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x), dim=1)
        out = torch.add(out, x16c)
        return out


class VNet_down_block(nn.Module):
    def __init__(self, in_channel, out_channel, n_conv):
        super(VNet_down_block, self).__init__()
        self.down_conv = nn.Conv3d(in_channel, out_channel, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu1 = nn.PReLU(out_channel)
        layers = [Conv_in_stage(out_channel)] * n_conv
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.down_conv(x)))
        out = copy_data(x)
        out = self.convs(out)
        out = torch.add(x, out)
        return out
    

class VNet_up_block(nn.Module):
    def __init__(self, in_channel, out_channel, n_conv):
        super(VNet_up_block, self).__init__()
        self.up_conv = nn.ConvTranspose3d(in_channel, out_channel // 2, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(out_channel // 2)
        self.relu1 = nn.PReLU(out_channel // 2)
        layers = [Conv_in_stage(out_channel)] * n_conv
        self.convs = nn.Sequential(*layers)
    
    def forward(self, x, prev):
        out = self.relu1(self.bn1(self.up_conv(x)))
        xcat = torch.cat((out, prev), dim=1)
        out = self.convs(xcat)
        out = torch.add(xcat, out)
        return out
    

class VNet_output_block(nn.Module):
    def __init__(self, in_channel=32, out_channel=1):
        super(VNet_output_block, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=1)
        self.relu1 = nn.PReLU(out_channel)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.softmax(out)
        return out
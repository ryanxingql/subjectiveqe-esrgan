import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32, rescale=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat+1*num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat+2*num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat+3*num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat+4*num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.rescale = rescale

    def forward(self, x0):
        x1 = self.lrelu(self.conv1(x0))
        x2 = self.lrelu(self.conv2(torch.cat((x0, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x0, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x0, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x0, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * self.rescale + x0


class RRDB(nn.Module):
    """Residual in residual."""

    def __init__(self, num_feat=64, num_grow_ch=32, rescale=0.2):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch, rescale)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch, rescale)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch, rescale)
        self.rescale = rescale

    def forward(self, x0):
        x1 = self.rdb1(x0)
        x2 = self.rdb2(x1)
        out = self.rdb3(x2)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * self.rescale + x0


class RRDBNet(nn.Module):
    """RRDBNet as generator of ESRGAN for quality enhancement."""

    def __init__(self, opts_dict):
        super().__init__()

        num_in_ch = opts_dict['num_in_ch']
        num_feat = opts_dict['num_feat']
        num_block = opts_dict['num_block']
        num_grow_ch = opts_dict['num_grow_ch']
        num_out_ch = opts_dict['num_out_ch']

        self.conv_in = nn.Conv2d(num_in_ch, num_feat, 3, stride=1, padding=1)
        
        layers = []
        for _ in range(num_block):
            layers.append(
                RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch)
                )
        self.rrdb_lst = nn.Sequential(*layers)
        
        self.conv_mid = nn.Conv2d(num_feat, num_feat, 3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(num_feat, num_out_ch, 3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x0):
        x1 = self.conv_in(x0)  # no lrelu
        x2 = self.rrdb_lst(x1)
        x3 = self.conv_mid(x2)
        x4 = x1 + x3
        out = self.conv_out(self.lrelu(x4))
        return out


class VGGStyleDiscriminator128(nn.Module):
    def __init__(self, opts_dict):
        super().__init__()
        
        num_in_ch = opts_dict['num_in_ch']
        num_feat = opts_dict['num_feat']

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 128)  # if input is not 128x128, here error
        self.linear2 = nn.Linear(128, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.lrelu(self.conv0_0(x))  # (B ? 128 128)
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # (B ? 64 64)
        
        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # (B ? 32 32)
        
        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # (B ? 16 16)
        
        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # (B ? 8 8)

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # (B ? 4 4)

        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out

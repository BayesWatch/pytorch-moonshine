# network definition
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

# wildcard import for legacy reasons
from .blocks import *

def parse_options(convtype, blocktype):
    # legacy cmdline argument parsing
    if isinstance(convtype,str):
        conv = conv_function(convtype)
    else:
        raise NotImplementedError("Tuple convolution specification no longer supported.")

    if blocktype =='Basic':
        block = BasicBlock
    elif blocktype =='Bottle':
        block = BottleBlock
    elif blocktype =='Old':
        block = OldBlock
    return conv, block


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, conv, block, num_classes=10, dropRate=0.0, s = 1):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        nChannels = [int(a) for a in nChannels]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6

        assert n % s == 0, 'n mod s must be zero'

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = torch.nn.ModuleList()
        for i in range(s):
            self.block1.append(NetworkBlock(int(n//s), nChannels[0] if i == 0 else nChannels[1],
                                            nChannels[1], block, 1, dropRate, conv))

        # 2nd block
        self.block2 = torch.nn.ModuleList()
        for i in range(s):
            self.block2.append(NetworkBlock(int(n//s), nChannels[1] if i == 0 else nChannels[2],
                                            nChannels[2], block, 2 if i == 0 else 1, dropRate, conv))
        # 3rd block
        self.block3 = torch.nn.ModuleList()
        for i in range(s):
            self.block3.append(NetworkBlock(int(n//s), nChannels[2] if i == 0 else nChannels[3],
                                            nChannels[3], block, 2 if i == 0 else 1, dropRate, conv))
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # normal is better than uniform initialisation
        # this should really be in `self.reset_parameters`
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                try:
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                except AttributeError:
                    pass
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        activations = []
        out = self.conv1(x)
        #activations.append(out)

        for sub_block in self.block1:
            out = sub_block(out)
            activations.append(out)

        for sub_block in self.block2:
            out = sub_block(out)
            activations.append(out)

        for sub_block in self.block3:
            out = sub_block(out)
            activations.append(out)

        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out), activations


def test():
    net = WideResNet(40,2,Conv,BasicBlock)
    x = torch.randn(1,3,32,32)
    y, _ = net(Variable(x))
    print(y.size())

if __name__ == '__main__':
    test()

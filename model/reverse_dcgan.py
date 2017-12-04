import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from PIL import Image
import numpy as np

from tqdm import tqdm
import torchvision.transforms as transforms

import torch.optim as optim


class R_DCGAN(nn.Module):
    def __init__(self, nc=3, ndf=100, mlp_d=1024):
        super(R_DCGAN, self).__init__()
        self.convs = nn.Sequential(
            # input is (nc) x 100 x 100
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 50 x 50
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 25 x 25
            nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 12 x 12
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 6 x 6
            nn.Conv2d(ndf * 8, mlp_d, 6, 1, 0, bias=False),
            nn.LeakyReLU(inplace=True),
            # state size. 1024 x 1 x 1
        )

        # self.fcs = nn.Sequential(
        #     nn.Linear(mlp_d, obj_vec_d),
        # )

    def forward(self, ims):
        state_vec = self.convs(ims)
        batch_size = state_vec.size(0)
        # return self.fcs(state_vec.view(batch_size, -1))
        return state_vec.view(batch_size, -1)


if __name__ == "__main__":
    m = R_DCGAN()
    test_tensor = torch.FloatTensor(1, 3, 100, 100)
    out = m(Variable(test_tensor))
    print(m)
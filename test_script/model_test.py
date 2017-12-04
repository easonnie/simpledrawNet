import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from PIL import Image
import numpy as np

import torchvision.transforms as transforms

class Vext2I(nn.Module):
    def __init__(self, input_d, out_size):
        super(Vext2I).__init__()

        mlp_1 = 100
        mlp_2 = 10 * 10 * 20

        self.mlp_1 = nn.Linear(input_d, mlp_1)
        self.mlp_2 = nn.Linear(mlp_1, mlp_2)
        self.deconv1 = nn.ConvTranspose2d(20, 20, 4, 2)

        self.deconv2 = nn.ConvTranspose2d()


# DCGAN model with fc layers
class FC_DCGAN(nn.Module):
    def __init__(self, in_vec_d=100, mlp_d=1024, nc=3, ngf=512, dropout_r=0.2):
        super(FC_DCGAN, self).__init__()

        self.fcs = nn.Sequential(
            # input is Z, going into a convolution
            # state size. nz x 1 x 1
            nn.Linear(in_vec_d, mlp_d),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_r),
            nn.Linear(mlp_d, mlp_d),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_r),
            )

        self.convs = nn.Sequential(
            # 1024x1x1
            nn.ConvTranspose2d(mlp_d, ngf * 8, 6, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8) x 6 x 6
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            # state size. (ngf*4) x 12 x 12
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            # state size. (ngf*2) x 25 x 25
            nn.ConvTranspose2d(ngf * 2,     ngf, 6, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # state size. (ngf) x 50 x 50
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 100 x 100
        )

    def forward(self, in_vec):
        batch_size = in_vec.size(0)
        in_vec = self.fcs(in_vec.view(batch_size, -1))
        in_vec = in_vec.view(-1, 1024, 1, 1)
        out_im = self.convs(in_vec)
        return out_im


def test_func():
    batch_size = 3
    test_in = Variable(torch.FloatTensor(torch.randn(batch_size, 10)))

    # mlp_1 = nn.Linear(input_d, mlp_1)
    print(test_in.size())

    mlp_2 = nn.Linear(10, 5 * 5 * 2)
    deconv1 = nn.ConvTranspose2d(2, 50, 5, 3)
    deconv2 = nn.ConvTranspose2d(50, 3, 8, 4)
    bn2d_1 = nn.BatchNorm2d(2)
    bn2d_2 = nn.BatchNorm2d(50)
    bn2d_3 = nn.BatchNorm2d(3)

    test_in_2 = mlp_2(test_in)
    test_in_2 = bn2d_1(test_in_2.view(batch_size, -1, 5, 5))
    print(test_in_2.size())

    test_in_3 = bn2d_2(deconv1(test_in_2))
    print(test_in_3.size())

    test_in_4 = bn2d_3(deconv2(test_in_3))
    print(test_in_4.size())

    out = F.tanh(test_in_4)

    # np_out = out.data.mul(0.5).add(0.5).mul(255).byte().transpose(0, 2).transpose(0, 1).numpy()

    np_out = out.data.mul(0.5).add(0.5).mul(255).byte().transpose(1, 3).transpose(1, 2).numpy()

    print(np_out[0])

    # print(out)
    print(np_out.dtype)
    # np_out = out.data.numpy()[0]

    # print(np_out[0])
    im = Image.fromarray(np_out[0])
    #
    print(im)
    im.save("test.png")


    # deconv2 = nn.ConvTranspose2d()


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


if __name__ == "__main__":
    # model = FC_DCGAN()
    # batch_size = 3
    # test_in = Variable(torch.FloatTensor(torch.randn(batch_size, 100)))
    #
    # out = model(test_in)
    #
    # np_out = out.data.mul(0.5).add(0.5).mul(255).byte().transpose(1, 3).transpose(1, 2).numpy()
    #
    # np_out_0 = tensor2im(out.data[0])
    #
    # print(np_out[0])
    # print(np_out.dtype)
    # im = Image.fromarray(np_out[0])
    # im.save("test.png")
    #
    # print(np.allclose(np_out[0], np_out_0))
    #
    # print(model)
    # print(out)

    image = Image.open("/Users/Eason/RA/RL_net/data/nlvr/train/images/0/train-655-0-0.png")
    print(image)

    trans_func = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # bring images to (-1,1)
    ])

    tensor_image = trans_func(image)
    print(tensor_image[3])
    #
    # print(tensor2im(tensor_image))
    #
    # im = Image.fromarray(tensor2im(tensor_image))
    # im.save("test.png")

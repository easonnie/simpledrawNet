import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from PIL import Image
import numpy as np
import random

from tqdm import tqdm
import torchvision.transforms as transforms

import torchvision.utils as vutil

import torch.optim as optim


# def sum_first_k(inputs, lengths):
#     ls = list(lengths.data)
#
#     b_seq_max_list = []
#     for i, l in enumerate(ls):
#         seq_i = inputs[i, :l, :, :, :]
#         seq_i_sum = torch.sum(seq_i, dim=0)
#         b_seq_max_list.append(seq_i_sum)
#
#     return torch.stack(b_seq_max_list)


def pack_sequence_for_linear(inputs, lengths):
    """
    :param inputs: [B, T, D] if batch_first
    :param lengths:  [B]
    :param batch_first:
    :return:
    """
    batch_list = []
    lengths = list(lengths.data)

    for i, l in enumerate(lengths):
        batch_list.append(inputs[i, :l])
    packed_sequence = torch.cat(batch_list, 0)
    return packed_sequence


def pad_1d(seq, pad_l):
    """
    The seq is a sequence having shape [T, ..]. Note: The seq contains only one instance. This is not batched.

    :param seq:  Input sequence with shape [T, ...]
    :param pad_l: The required pad_length.
    :return:  Output sequence will have shape [Pad_L, ...]
    """
    l = seq.size(0)
    if l >= pad_l:
        return seq[:pad_l, ]  # Truncate the length if the length is bigger than required padded_length.
    else:
        pad_seq = Variable(seq.data.new(pad_l - l, *seq.size()[1:]).zero_())  # Requires_grad is False
    return torch.cat([seq, pad_seq], dim=0)


def unpack_from_linear(inputs, lengths, pad_l=8):
    batch_list = []
    lengths = list(lengths.data)

    if not isinstance(inputs, list):
        inputs = [inputs]
    inputs = torch.cat(inputs)

    start = 0
    for l in lengths:
        end = start + l
        batch_list.append(pad_1d(inputs[start:end], pad_l=pad_l).view(-1))
        start = end
    return torch.stack(batch_list, dim=0)


def unpack_sum_sequence_for_linear(inputs, lengths):
    batch_list = []
    lengths = list(lengths.data)

    if not isinstance(inputs, list):
        inputs = [inputs]
    inputs = torch.cat(inputs)

    start = 0
    for l in lengths:
        end = start + l
        batch_list.append(torch.sum(inputs[start:end], dim=0))
        start = end
    return torch.stack(batch_list)


# def unpack_sum_sequence_for_linear(inputs, lengths):
#     batch_list = []
#     lengths = list(lengths.data)
#
#     if not isinstance(inputs, list):
#         inputs = [inputs]
#     inputs = torch.cat(inputs)
#
#     start = 0
#     for l in lengths:
#         end = start + l
#         batch_list.append(torch.sum(inputs[start:end], dim=0))
#         start = end
#     return torch.stack(batch_list)

def pad_2d(inputs, pad_l=8):
    pass


class Scence2Image(nn.Module):
    def __init__(self, encoding_d=9, obj_vec_d=4096):
        super(Scence2Image, self).__init__()
        self.obj2vec = Obj2Vec(encoding_d, mlp_d=2048, obj_vec_d=obj_vec_d)
        self.fc_dcgan = FC_DCGAN(obj_vec_d, nc=3, ngf=128)
        # self.fc_dcgan = FC_DCGAN(obj_vec_d * 8)

    def forward(self, ss, ls, max_l=8):
        # ss : [B, T, D]
        # ls : [B]
        packed_ins = pack_sequence_for_linear(ss, ls)
        obj_vecs = self.obj2vec(packed_ins)
        sum_vecs = unpack_sum_sequence_for_linear(obj_vecs, ls)
        image_outs = self.fc_dcgan(sum_vecs) # [B, T, C, H, W]
        return image_outs

    def scence2vec(self, ss, ls, max_l=8):
        packed_ins = pack_sequence_for_linear(ss, ls)
        obj_vecs = self.obj2vec(packed_ins)
        sum_vecs = unpack_sum_sequence_for_linear(obj_vecs, ls)
        return sum_vecs

    def objs_vec2im(self, objs_vec):
        image_outs = self.fc_dcgan(objs_vec)  # [B, T, C, H, W]
        return image_outs

    def obj2im(self, encoding_vec):
        obj_vecs = self.obj2vec(encoding_vec)
        image_outs = self.fc_dcgan(obj_vecs)  # [B, T, C, H, W]
        return image_outs


class Obj2Vec(nn.Module):
    def __init__(self, encoding_d=9, mlp_d=2048, obj_vec_d=4096):
        super(Obj2Vec, self).__init__()
        self._obj2vec = nn.Sequential(
            nn.Linear(encoding_d, mlp_d),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_d, obj_vec_d),
        )

    def forward(self, ins):
        return self._obj2vec(ins)


# DCGAN model with fc layers
class FC_DCGAN(nn.Module):
    def __init__(self, obj_vec_d=4096, nc=3, ngf=128):
        super(FC_DCGAN, self).__init__()

        self.convs = nn.Sequential(
            # 1024x1x1
            nn.ConvTranspose2d(obj_vec_d, ngf * 8, 6, 1, 0, bias=False),
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

    def forward(self, obj_encoding):
        in_vec = obj_encoding
        in_vec = in_vec.view(-1, 4096, 1, 1)
        out_im = self.convs(in_vec)
        return out_im


if __name__ == "__main__":
    import util
    (ss, ls), ims = util.load_data()
    test_index = 50
    r_l = 10
    end_index = test_index + r_l
    sample_s, sample_l, sample_i = ss[test_index:end_index], ls[test_index:end_index], ims[test_index:end_index]

    # sample_s = Variable(sample_s)
    # sample_l = Variable(sample_l)
    # sample_i = Variable(sample_i)

    # print()
    # ss[]
    torch.manual_seed(8)
    # fc_dcgan = FC_DCGAN()
    model = Scence2Image()

    if torch.cuda.is_available():
        model.cuda()
    # ims = model(sample_s, sample_l)

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), 1e-4)

    for i in tqdm(range(1000)):
        model.train()

        sample_s_v = Variable(sample_s)
        sample_l_v = Variable(sample_l)
        sample_i_v = Variable(sample_i)

        if torch.cuda.is_available():
            sample_s_v = sample_s_v.cuda()
            sample_l_v = sample_l_v.cuda()
            sample_i_v = sample_i_v.cuda()

        ims = model(sample_s_v, sample_l_v)

        loss = mse_loss(ims, sample_i_v)
        # loss = l1_loss(ims, sample_i_v)
        optimizer.zero_grad()
        loss.backward()
        # print([param for param in model.fc_dcgan._obj2vec.parameters()])
        # print(loss)
        optimizer.step()

    model.eval()
    ims = model(sample_s_v, sample_l_v)

    vecs = model.scence2vec(sample_s_v, sample_l_v)

    vec1 = vecs[0]
    vec2 = vecs[2]

    ims = util.abstract_changing(vec1, vec2, model)
    # print(ims)

    g_im = vutil.make_grid(ims.data, nrow=8, padding=15)
    Image.fromarray(util.tensor2im(g_im)).save("g_im_changing_8.png")

    # print(sample_s_v)
    # print(ims[0])

    # g_im = vutil.make_grid([ims[0].data, ims[1].data, ims[2].data], nrow=3, padding=15)
    #
    # Image.fromarray(util.tensor2im(g_im)).save("g_im1_l1.png")

    # t_im = Image.fromarray(util.tensor2im(ims[0].data))
    # t_im.save("t_im_0.png")
    # t_im = Image.fromarray(util.tensor2im(ims[1].data))
    # t_im.save("t_im_1.png")
    # t_im = Image.fromarray(util.tensor2im(ims[2].data))
    # t_im.save("t_im_2.png")

    # print(fc_dcgan(sample_s[:, 1, :]))

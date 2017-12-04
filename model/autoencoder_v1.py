import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from PIL import Image
import numpy as np

from tqdm import tqdm
import torchvision.transforms as transforms

import torch.optim as optim
import torchvision.utils as vutil


from model.dc_gan_no_fc import FC_DCGAN
from model.reverse_dcgan import R_DCGAN


class AutoEncoder(nn.Module):
    def __init__(self, obj_vec_d=1024):
        super(AutoEncoder, self).__init__()
        self.encoder = R_DCGAN(mlp_d=obj_vec_d, ndf=128)
        self.decoder = FC_DCGAN(obj_vec_d=obj_vec_d, ngf=128)

    def forward(self, ims):
        vec = self.encoder(ims)
        return self.decoder(vec)

    def im2vec(self, ims):
        return self.encoder(ims)

    def objs_vec2im(self, vec):
        return self.decoder(vec)


def produce_dev_images():
    import util
    (ss, ls), ims = util.load_data(mode='dev')
    test_index = 50
    r_l = 30
    end_index = test_index + r_l
    sample_s, sample_l, sample_i = ss[test_index:end_index], ls[test_index:end_index], ims[test_index:end_index]

    obj_vec_d = 2400
    model = AutoEncoder(obj_vec_d=obj_vec_d)
    # model = Scence2Image()

    if torch.cuda.is_available():
        model.cuda()

    SAVE_PATH = "/media/easonnie/Seagate Expansion Drive/RL_net/m_82999_0.2122452172755806_d:2400_auto_encoder"
    model.load_state_dict(torch.load(SAVE_PATH))

    model.eval()

    sample_s_v = Variable(sample_s)
    sample_l_v = Variable(sample_l)
    sample_i_v = Variable(sample_i)

    if torch.cuda.is_available():
        sample_s_v = sample_s_v.cuda()
        sample_l_v = sample_l_v.cuda()
        sample_i_v = sample_i_v.cuda()

    vecs = model.im2vec(sample_i_v)

    im_list = []

    vec1 = vecs[0]
    vec2 = vecs[2]
    ims = util.abstract_changing(vec1, vec2, model)
    im_list.append(ims)

    vec1 = vecs[1]
    vec2 = vecs[3]
    ims = util.abstract_changing(vec1, vec2, model)
    im_list.append(ims)

    vec1 = vecs[4]
    vec2 = vecs[5]
    ims = util.abstract_changing(vec1, vec2, model)
    im_list.append(ims)

    vec1 = vecs[12]
    vec2 = vecs[6]
    ims = util.abstract_changing(vec1, vec2, model)
    im_list.append(ims)

    vec1 = vecs[21]
    vec2 = vecs[9]
    ims = util.abstract_changing(vec1, vec2, model)
    im_list.append(ims)

    vec1 = vecs[14]
    vec2 = vecs[20]
    ims = util.abstract_changing(vec1, vec2, model)
    im_list.append(ims)

    ims = torch.cat(im_list, dim=0)
    # print(ims)

    g_im = vutil.make_grid(ims.data, nrow=8, padding=15)
    Image.fromarray(util.tensor2im(g_im)).save("vector_shifting_3_auto.png")


if __name__ == "__main__":
    produce_dev_images()
    # import util
    #
    # (ss, ls), ims = util.load_data()
    # test_index = 50
    # r_l = 3
    # end_index = test_index + r_l
    # sample_s, sample_l, sample_i = ss[test_index:end_index], ls[test_index:end_index], ims[test_index:end_index]
    #
    # torch.manual_seed(8)
    # # fc_dcgan = FC_DCGAN()
    # auto_encoder = AutoEncoder()
    #
    # if torch.cuda.is_available():
    #     auto_encoder.cuda()
    # # ims = model(sample_s, sample_l)
    #
    # mse_loss = nn.MSELoss()
    # optimizer = optim.Adam(auto_encoder.parameters(), 1e-4)
    #
    # for i in tqdm(range(200)):
    #     auto_encoder.train()
    #
    #     sample_s_v = Variable(sample_s)
    #     sample_l_v = Variable(sample_l)
    #     sample_i_v = Variable(sample_i)
    #
    #     if torch.cuda.is_available():
    #         sample_s_v = sample_s_v.cuda()
    #         sample_l_v = sample_l_v.cuda()
    #         sample_i_v = sample_i_v.cuda()
    #
    #     ims = auto_encoder(sample_i_v)
    #
    #     loss = mse_loss(ims, sample_i_v)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     # print([param for param in model.fc_dcgan._obj2vec.parameters()])
    #     # print(loss)
    #     optimizer.step()
    #
    # auto_encoder.eval()
    #
    # vecs = auto_encoder.encoder(sample_i_v)
    # # vecs = model.scence2vec(sample_s_v, sample_l_v)
    # #
    # vec1 = vecs[0]
    # vec2 = vecs[2]
    # #
    # ims = util.abstract_changing_decoder(vec1, vec2, auto_encoder)
    # # print(ims)
    #
    # g_im = vutil.make_grid(ims.data, nrow=8, padding=15)
    # Image.fromarray(util.tensor2im(g_im)).save("g_im_changing_8_auto.png")

    # ims = auto_encoder(sample_i_v)
    # print(sample_s_v)
    # print(ims[0])
    # t_im = Image.fromarray(util.tensor2im(ims[0].data))
    # t_im.save("t_im_0_auto.png")
    # t_im = Image.fromarray(util.tensor2im(ims[1].data))
    # t_im.save("t_im_1_auto.png")
    # t_im = Image.fromarray(util.tensor2im(ims[2].data))
    # t_im.save("t_im_2_auto.png")
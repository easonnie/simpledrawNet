# {"y_loc":21,"size":20,"type":"triangle","x_loc":27,"color":"Yellow"}
# import processing
from torch.autograd import Variable
from tqdm import tqdm

import util
import processing
from model.ful_dcgan_v2 import Scence2Image
from PIL import Image
import torch


import torchvision.utils as vutil


def test_on_dev_obj2im(model):
    batch_size = 32

    data_geter = util.B_Geter(mode='dev')
    print(data_geter.total_size)

    if torch.cuda.is_available():
        model.cuda()

    mse_loss = torch.nn.MSELoss()
    # l1_loss = torch.nn.L1Loss()
    # optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    t_loss = 0
    total_size = 0

    # with tqdm(total=data_geter.total_size) as pbar:
    while True:
        model.eval()
        # pbar.update(batch_size)
        # print(data_geter.start)
        v_ss = None
        v_ss, v_ls, v_image = data_geter.yield_data(batch_size)

        if v_ss is None:
            break

        sample_s_v = Variable(v_ss)
        sample_l_v = Variable(v_ls)
        sample_i_v = Variable(v_image)

        if torch.cuda.is_available():
            sample_s_v = sample_s_v.cuda()
            sample_l_v = sample_l_v.cuda()
            sample_i_v = sample_i_v.cuda()

        ims = model(sample_s_v, sample_l_v)

        actual_b_size = int(sample_l_v.size(0))
        total_size += actual_b_size

        loss = mse_loss(ims, sample_i_v)
        t_loss += float(loss.data.cpu().numpy())

    # t_loss = sum(l_list) / len(l_list)
    t_loss /= total_size
    return t_loss * 100 * 100


def test_on_dev_auto_encoder(model):
    batch_size = 32

    data_geter = util.B_Geter(mode='dev')
    print(data_geter.total_size)

    if torch.cuda.is_available():
        model.cuda()

    mse_loss = torch.nn.MSELoss()
    # l1_loss = torch.nn.L1Loss()
    # optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    t_loss = 0
    total_size = 0

    # with tqdm(total=data_geter.total_size) as pbar:
    while True:
        model.eval()
        # pbar.update(batch_size)
        # print(data_geter.start)
        v_ss = None
        v_ss, v_ls, v_image = data_geter.yield_data(batch_size)

        if v_ss is None:
            break

        sample_s_v = Variable(v_ss)
        sample_l_v = Variable(v_ls)
        sample_i_v = Variable(v_image)

        if torch.cuda.is_available():
            sample_s_v = sample_s_v.cuda()
            sample_l_v = sample_l_v.cuda()
            sample_i_v = sample_i_v.cuda()

        ims = model(sample_i_v)

        actual_b_size = int(sample_l_v.size(0))
        total_size += actual_b_size

        loss = mse_loss(ims, sample_i_v)
        t_loss += float(loss.data.cpu().numpy())

    # t_loss = sum(l_list) / len(l_list)
    t_loss /= total_size
    return t_loss * 100 * 100


def draw_obj_eval(model: Scence2Image, obj):
    obj_tensor = processing.obj2tensor(obj)

    if torch.cuda.is_available():
        obj_tensor = obj_tensor.cuda()

    obj_im = model.obj2im(Variable(obj_tensor))[0]

    im = Image.fromarray(util.tensor2im(obj_im.data.cpu()))
    return im


def random_eval():
    (ss, ls), ims = util.load_data()
    test_index = 50
    r_l = 10
    end_index = test_index + r_l
    sample_s, sample_l, sample_i = ss[test_index:end_index], ls[test_index:end_index], ims[test_index:end_index]

    obj_vec_d = 300
    model = Scence2Image(encoding_d=9, obj_vec_d=obj_vec_d)
    # model = Scence2Image()

    if torch.cuda.is_available():
        model.cuda()

    SAVE_PATH = "/media/easonnie/Seagate Expansion Drive/RL_net/m_32999_4.3951619817199825"
    model.load_state_dict(torch.load(SAVE_PATH))

    model.eval()

    sample_s_v = Variable(sample_s)
    sample_l_v = Variable(sample_l)
    sample_i_v = Variable(sample_i)

    print(sample_s_v)

    if torch.cuda.is_available():
        sample_s_v = sample_s_v.cuda()
        sample_l_v = sample_l_v.cuda()
        sample_i_v = sample_i_v.cuda()
    vecs = model.scence2vec(sample_s_v, sample_l_v)

    vec1 = vecs[0]
    vec2 = vecs[2]

    ims = util.abstract_changing(vec1, vec2, model)
    # print(ims)

    g_im = vutil.make_grid(ims.data, nrow=8, padding=15)
    Image.fromarray(util.tensor2im(g_im)).save("g_im_changing_8_(0-2).png")


if __name__ == '__main__':
    # test_scence = {"y_loc":80,"size":30,"type":"square","x_loc":40,"color":"#0099ff"}
    # # obj_vec_d = 600
    # # model = Scence2Image(encoding_d=9, obj_vec_d=obj_vec_d)
    # # # print(test_on_dev_obj2im(model))
    # #
    # # test_scence = {"y_loc":30,"size":20,"type":"square","x_loc":30,"color":'#0099ff'}
    # # #
    # obj_vec_d = 200
    # model = Scence2Image(encoding_d=9, obj_vec_d=obj_vec_d)
    # #
    # if torch.cuda.is_available():
    #     model.cuda()
    # #
    # SAVE_PATH = "/media/easonnie/Seagate Expansion Drive/RL_net/m_999_12.695772036122813"
    # model.load_state_dict(torch.load(SAVE_PATH))
    # #
    # model.eval()
    # im = draw_obj_eval(model, test_scence)
    # im.save("haha_kankan.png")
    random_eval()

# -0.2000  0.6000  0.0000  0.0000  0.0000  1.0000  0.0000  1.0000  0.0000
# -0.2000  0.1800  0.0000  0.0000  0.0000  1.0000  0.0000  1.0000  0.0000
# -0.2000 -0.2400  0.0000  0.0000  0.0000  1.0000  1.0000  0.0000  0.0000
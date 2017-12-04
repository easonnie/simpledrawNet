# {"y_loc":21,"size":20,"type":"triangle","x_loc":27,"color":"Yellow"}
# import processing
from torch.autograd import Variable
from tqdm import tqdm

import util
import processing
from model.dc_gan_no_fc import Scence2Image
from PIL import Image
import torch
from processing import scence2tensor


import torchvision.utils as vutil


# def test_on_dev_obj2im(model):
#     batch_size = 32
#
#     data_geter = util.B_Geter(mode='dev')
#     print(data_geter.total_size)
#
#     if torch.cuda.is_available():
#         model.cuda()
#
#     mse_loss = torch.nn.MSELoss()
#     # l1_loss = torch.nn.L1Loss()
#     # optimizer = torch.optim.Adam(model.parameters(), 1e-4)
#
#     t_loss = 0
#     total_size = 0
#
#     # with tqdm(total=data_geter.total_size) as pbar:
#     while True:
#         model.eval()
#         # pbar.update(batch_size)
#         # print(data_geter.start)
#         v_ss = None
#         v_ss, v_ls, v_image = data_geter.yield_data(batch_size)
#
#         if v_ss is None:
#             break
#
#         sample_s_v = Variable(v_ss)
#         sample_l_v = Variable(v_ls)
#         sample_i_v = Variable(v_image)
#
#         if torch.cuda.is_available():
#             sample_s_v = sample_s_v.cuda()
#             sample_l_v = sample_l_v.cuda()
#             sample_i_v = sample_i_v.cuda()
#
#         ims = model(sample_s_v, sample_l_v)
#
#         actual_b_size = int(sample_l_v.size(0))
#         total_size += actual_b_size
#
#         loss = mse_loss(ims, sample_i_v)
#         t_loss += float(loss.data.cpu().numpy())
#
#     # t_loss = sum(l_list) / len(l_list)
#     t_loss /= total_size
#     return t_loss * 100 * 100
#
#
# def test_on_dev_auto_encoder(model):
#     batch_size = 32
#
#     data_geter = util.B_Geter(mode='dev')
#     print(data_geter.total_size)
#
#     if torch.cuda.is_available():
#         model.cuda()
#
#     mse_loss = torch.nn.MSELoss()
#     # l1_loss = torch.nn.L1Loss()
#     # optimizer = torch.optim.Adam(model.parameters(), 1e-4)
#
#     t_loss = 0
#     total_size = 0
#
#     # with tqdm(total=data_geter.total_size) as pbar:
#     while True:
#         model.eval()
#         # pbar.update(batch_size)
#         # print(data_geter.start)
#         v_ss = None
#         v_ss, v_ls, v_image = data_geter.yield_data(batch_size)
#
#         if v_ss is None:
#             break
#
#         sample_s_v = Variable(v_ss)
#         sample_l_v = Variable(v_ls)
#         sample_i_v = Variable(v_image)
#
#         if torch.cuda.is_available():
#             sample_s_v = sample_s_v.cuda()
#             sample_l_v = sample_l_v.cuda()
#             sample_i_v = sample_i_v.cuda()
#
#         ims = model(sample_i_v)
#
#         actual_b_size = int(sample_l_v.size(0))
#         total_size += actual_b_size
#
#         loss = mse_loss(ims, sample_i_v)
#         t_loss += float(loss.data.cpu().numpy())
#
#     # t_loss = sum(l_list) / len(l_list)
#     t_loss /= total_size
#     return t_loss * 100 * 100


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

    SAVE_PATH = "/media/easonnie/Seagate Expansion Drive/RL_net/m_22999_1.8898435379786562_d:300_no_fc_cat"
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
    vec2 = vecs[1]

    ims = util.abstract_changing(vec1, vec2, model)
    # print(ims)


    g_im = vutil.make_grid(ims.data, nrow=8, padding=15)
    Image.fromarray(util.tensor2im(g_im)).save("g_im_changing_8_(0-1).png")


def eval_given_input():
    obj_vec_d = 300
    model = Scence2Image(encoding_d=9, obj_vec_d=obj_vec_d)
    # model = Scence2Image()

    if torch.cuda.is_available():
        model.cuda()

    SAVE_PATH = "/media/easonnie/Seagate Expansion Drive/RL_net/m_99999_0.5905408416487539_d:300_no_fc_cat"
    model.load_state_dict(torch.load(SAVE_PATH))

    model.eval()

    # dev_sces = [
    #     [
    #      # {"y_loc":80,"type":"triangle","color":"#0099ff","x_loc":80,"size":20},
    #      # {"y_loc":47,"type":"circle","color":"Yellow","x_loc":90,"size":10},
    #        {"y_loc":21,"type":"triangle","color":"#0099ff","x_loc":72,"size":10}
    #      ]
    # ]

    comp_sce = [[{"y_loc":80,"type":"square","color":"Black","x_loc":35,"size":20},
        {"y_loc":42,"type":"triangle","color":"#0099ff","x_loc":13,"size":10},
    {"y_loc":21,"type":"square","color":"#0099ff","x_loc":13,"size":10},
    {"y_loc":48,"type":"triangle","color":"Yellow","x_loc":68,"size":20},
    {"y_loc":70,"type":"triangle","color":"Yellow","x_loc":62,"size":30},
    {"y_loc":3,"type":"circle","color":"#0099ff","x_loc":19,"size":10}],

    [{"y_loc":80,"type":"triangle","color":"Black","x_loc":80,"size":20},
    {"y_loc":2,"type":"triangle","color":"#0099ff","x_loc":65,"size":30},
    {"y_loc":35,"type":"square","color":"Yellow","x_loc":79,"size":20},
    {"y_loc":70,"type":"triangle","color":"Yellow","x_loc":1,"size":30},
    {"y_loc":60,"type":"circle","color":"Black","x_loc":43,"size":20},
    {"y_loc":34,"type":"circle","color":"Black","x_loc":6,"size":30},

    {"y_loc":11,"type":"square","color":"Yellow","x_loc":13,"size":20}],

    [{"y_loc":32,"type":"circle","color":"Black","x_loc":49,"size":10},
    {"y_loc":38,"type":"triangle","color":"Black","x_loc":28,"size":20},
    {"y_loc":70,"type":"circle","color":"#0099ff","x_loc":70,"size":30},
    {"y_loc":8,"type":"triangle","color":"#0099ff","x_loc":76,"size":20},
    {"y_loc":70,"type":"square","color":"#0099ff","x_loc":9,"size":30},
    {"y_loc":37,"type":"triangle","color":"#0099ff","x_loc":82,"size":10},
    {"y_loc":2,"type":"circle","color":"Black","x_loc":4,"size":30}],

    [{"y_loc": 15, "type": "triangle", "color": "Black", "x_loc": 44, "size": 10},
     {"y_loc": 33, "type": "triangle", "color": "Black", "x_loc": 28, "size": 20},
     {"y_loc": 70, "type": "circle", "color": "#0099ff", "x_loc": 70, "size": 30},
     {"y_loc": 5, "type": "square", "color": "#0099ff", "x_loc": 76, "size": 20},
     {"y_loc": 70, "type": "square", "color": "#0099ff", "x_loc": 4, "size": 30},
     {"y_loc": 37, "type": "triangle", "color": "#0099ff", "x_loc": 82, "size": 10},
     {"y_loc": 2, "type": "circle", "color": "Black", "x_loc": 6, "size": 30},
     {"y_loc": 60, "type": "triangle", "color": "#0099ff", "x_loc": 56, "size": 20}],

    [{"y_loc": 32, "type": "triangle", "color": "Black", "x_loc": 55, "size": 20},
     {"y_loc": 32, "type": "triangle", "color": "#0099ff", "x_loc": 12, "size": 30},
     {"y_loc": 64, "type": "circle", "color": "Yellow", "x_loc": 45, "size": 20},
     {"y_loc": 34, "type": "square", "color": "Yellow", "x_loc": 7, "size": 30},
     {"y_loc": 12, "type": "triangle", "color": "Black", "x_loc": 8, "size": 20},
     {"y_loc": 45, "type": "circle", "color": "Black", "x_loc": 2, "size": 30},

     {"y_loc": 44, "type": "square", "color": "Yellow", "x_loc": 13, "size": 20}],

    ]

    # pos_scences = [
    #     [
    #         {"y_loc": 0, "size": 20, "type": "circle", "x_loc": 0, "color": "#0099ff"},
    #      ],
    #     [
    #         {"y_loc": 0, "size": 20, "type": "circle", "x_loc": 20, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 0, "size": 20, "type": "circle", "x_loc": 40, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 0, "size": 20, "type": "circle", "x_loc": 60, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 0, "size": 20, "type": "circle", "x_loc": 80, "color": "#0099ff"},
    #     ],
    #
    #     [
    #         {"y_loc": 20, "size": 20, "type": "circle", "x_loc": 0, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 20, "size": 20, "type": "circle", "x_loc": 20, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 20, "size": 20, "type": "circle", "x_loc": 40, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 20, "size": 20, "type": "circle", "x_loc": 60, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 20, "size": 20, "type": "circle", "x_loc": 80, "color": "#0099ff"},
    #     ],
    #
    #     [
    #         {"y_loc": 40, "size": 20, "type": "circle", "x_loc": 0, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 40, "size": 20, "type": "circle", "x_loc": 20, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 40, "size": 20, "type": "circle", "x_loc": 40, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 40, "size": 20, "type": "circle", "x_loc": 60, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 40, "size": 20, "type": "circle", "x_loc": 80, "color": "#0099ff"},
    #     ],
    #
    #     [
    #         {"y_loc": 60, "size": 20, "type": "circle", "x_loc": 0, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 60, "size": 20, "type": "circle", "x_loc": 20, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 60, "size": 20, "type": "circle", "x_loc": 40, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 60, "size": 20, "type": "circle", "x_loc": 60, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 60, "size": 20, "type": "circle", "x_loc": 80, "color": "#0099ff"},
    #     ],
    #
    #     [
    #         {"y_loc": 80, "size": 20, "type": "circle", "x_loc": 0, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 80, "size": 20, "type": "circle", "x_loc": 20, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 80, "size": 20, "type": "circle", "x_loc": 40, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 80, "size": 20, "type": "circle", "x_loc": 60, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 80, "size": 20, "type": "circle", "x_loc": 80, "color": "#0099ff"},
    #     ],
    #
    # ]

    # scences = [
    #     [
    #         {"y_loc": 50, "size": 10, "type": "square", "x_loc": 40, "color": "#0099ff"},
    #      ],
    #     [
    #         {"y_loc": 50, "size": 10, "type": "circle", "x_loc": 40, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 10, "type": "triangle", "x_loc": 40, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 10, "type": "square", "x_loc": 40, "color": "Yellow"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 10, "type": "circle", "x_loc": 40, "color": "Yellow"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 10, "type": "triangle", "x_loc": 40, "color": "Yellow"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 10, "type": "square", "x_loc": 40, "color": "Black"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 10, "type": "circle", "x_loc": 40, "color": "Black"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 10, "type": "triangle", "x_loc": 40, "color": "Black"},
    #     ],
    #
    #     [
    #         {"y_loc": 50, "size": 10, "type": "square", "x_loc": 40, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 20, "type": "circle", "x_loc": 40, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 20, "type": "triangle", "x_loc": 40, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 20, "type": "square", "x_loc": 40, "color": "Yellow"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 20, "type": "circle", "x_loc": 40, "color": "Yellow"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 20, "type": "triangle", "x_loc": 40, "color": "Yellow"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 20, "type": "square", "x_loc": 40, "color": "Black"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 20, "type": "circle", "x_loc": 40, "color": "Black"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 20, "type": "triangle", "x_loc": 40, "color": "Black"},
    #     ],
    #
    #     [
    #         {"y_loc": 50, "size": 30, "type": "square", "x_loc": 40, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 30, "type": "circle", "x_loc": 40, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 30, "type": "triangle", "x_loc": 40, "color": "#0099ff"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 30, "type": "square", "x_loc": 40, "color": "Yellow"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 30, "type": "circle", "x_loc": 40, "color": "Yellow"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 30, "type": "triangle", "x_loc": 40, "color": "Yellow"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 30, "type": "square", "x_loc": 40, "color": "Black"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 30, "type": "circle", "x_loc": 40, "color": "Black"},
    #     ],
    #     [
    #         {"y_loc": 50, "size": 30, "type": "triangle", "x_loc": 40, "color": "Black"},
    #     ],
    #
    # ]

    sample_s, sample_l = processing.scences2input(comp_sce)

    sample_s_v = Variable(sample_s)
    sample_l_v = Variable(sample_l)
    # sample_i_v = Variable(sample_i)

    print(sample_s_v)

    if torch.cuda.is_available():
        sample_s_v = sample_s_v.cuda()
        sample_l_v = sample_l_v.cuda()
        # sample_i_v = sample_i_v.cuda()

    ims = model.forward_none_shuffle(sample_s_v, sample_l_v)
    # vecs = model.scence2vec(sample_s_v, sample_l_v)
    #
    # vec1 = vecs[0]
    # vec2 = vecs[1]
    #
    # ims = util.abstract_changing(vec1, vec2, model)
    # print(ims)

    g_im = vutil.make_grid(ims.data, nrow=5, padding=15)
    Image.fromarray(util.tensor2im(g_im)).save("complex_2.png")


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
    # random_eval()
    eval_given_input()

# -0.2000  0.6000  0.0000  0.0000  0.0000  1.0000  0.0000  1.0000  0.0000
# -0.2000  0.1800  0.0000  0.0000  0.0000  1.0000  0.0000  1.0000  0.0000
# -0.2000 -0.2400  0.0000  0.0000  0.0000  1.0000  1.0000  0.0000  0.0000

# [[{"y_loc":80,"type":"square","color":"Black","x_loc":35,"size":20},
#     {"y_loc":42,"type":"triangle","color":"#0099ff","x_loc":13,"size":10},
    # {"y_loc":21,"type":"square","color":"#0099ff","x_loc":13,"size":10},
    # {"y_loc":48,"type":"triangle","color":"Yellow","x_loc":68,"size":20},
    # {"y_loc":70,"type":"triangle","color":"Yellow","x_loc":62,"size":30},
    # {"y_loc":3,"type":"circle","color":"#0099ff","x_loc":19,"size":10}],

    # [{"y_loc":80,"type":"triangle","color":"Black","x_loc":80,"size":20},
    # {"y_loc":2,"type":"triangle","color":"#0099ff","x_loc":65,"size":30},
    # {"y_loc":35,"type":"square","color":"Yellow","x_loc":79,"size":20},
    # {"y_loc":70,"type":"triangle","color":"Yellow","x_loc":1,"size":30},
    # {"y_loc":60,"type":"circle","color":"Black","x_loc":43,"size":20},
    # {"y_loc":34,"type":"circle","color":"Black","x_loc":6,"size":30},

    # {"y_loc":11,"type":"square","color":"Yellow","x_loc":13,"size":20}],
    # [{"y_loc":32,"type":"circle","color":"Black","x_loc":49,"size":10},
    # {"y_loc":38,"type":"triangle","color":"Black","x_loc":28,"size":20},
    # {"y_loc":70,"type":"circle","color":"#0099ff","x_loc":70,"size":30},
    # {"y_loc":8,"type":"triangle","color":"#0099ff","x_loc":76,"size":20},
    # {"y_loc":70,"type":"square","color":"#0099ff","x_loc":9,"size":30},
    # {"y_loc":37,"type":"triangle","color":"#0099ff","x_loc":82,"size":10},
    # {"y_loc":2,"type":"circle","color":"Black","x_loc":4,"size":30}]]}
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.optim as optim

from eval import test_on_dev_obj2im
from model.dc_gan_no_fc_sum import Scence2Image
from util import B_Geter

if __name__ == '__main__':
    torch.manual_seed(6)

    batch_size = 32

    obj_vec_d = 4096
    model = Scence2Image(encoding_d=9, obj_vec_d=obj_vec_d)

    data_geter = B_Geter()
    print(data_geter.total_size)

    if torch.cuda.is_available():
        model.cuda()

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), 1e-4)

    for i in tqdm(range(100000)):
        model.train()
        v_ss, v_ls, v_image = data_geter.yield_data(batch_size)

        sample_s_v = Variable(v_ss)
        sample_l_v = Variable(v_ls)
        sample_i_v = Variable(v_image)

        if torch.cuda.is_available():
            sample_s_v = sample_s_v.cuda()
            sample_l_v = sample_l_v.cuda()
            sample_i_v = sample_i_v.cuda()

        ims = model(sample_s_v, sample_l_v)

        loss = mse_loss(ims, sample_i_v)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if i % 100 == 0:
        #     loss = test_on_dev_obj2im(model)
        #     print(loss)

        mod = 1000
        if (i + 1) % 1000 == 0:
            loss = test_on_dev_obj2im(model)
            print(loss)
            save_path = "m_no_fc_sum_{0}_{1}_d:{2}".format(i, loss, obj_vec_d)
            torch.save(model.state_dict(), save_path)

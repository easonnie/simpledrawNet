import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.optim as optim

from eval import test_on_dev_obj2im, test_on_dev_auto_encoder
from model.autoencoder_v1 import AutoEncoder
from util import B_Geter

if __name__ == '__main__':
    torch.manual_seed(6)

    batch_size = 32

    obj_vec_d = 2400
    auto_encoder = AutoEncoder(obj_vec_d)

    data_geter = B_Geter()
    print(data_geter.total_size)

    if torch.cuda.is_available():
        auto_encoder.cuda()

    mse_loss = nn.MSELoss()
    # l1_loss = nn.L1Loss()
    optimizer = optim.Adam(auto_encoder.parameters(), 1e-4)

    for i in tqdm(range(100000)):
        auto_encoder.train()
        v_ss, v_ls, v_image = data_geter.yield_data(batch_size)

        sample_s_v = Variable(v_ss)
        sample_l_v = Variable(v_ls)
        sample_i_v = Variable(v_image)

        if torch.cuda.is_available():
            sample_s_v = sample_s_v.cuda()
            sample_l_v = sample_l_v.cuda()
            sample_i_v = sample_i_v.cuda()

        ims = auto_encoder(sample_i_v)

        loss = mse_loss(ims, sample_i_v)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if i % 100 == 0:
        #     loss = test_on_dev_obj2im(model)
        #     print(loss)

        mod = 1000
        if (i + 1) % 1000 == 0:
            loss = test_on_dev_auto_encoder(auto_encoder)
            print(loss)
            save_path = "m_{0}_{1}_d:{2}_auto_encoder".format(i, loss, obj_vec_d)
            torch.save(auto_encoder.state_dict(), save_path)

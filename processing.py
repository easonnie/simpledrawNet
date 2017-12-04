import json
from collections import Counter

from PIL import Image
import os
from tqdm import tqdm
import torch

from PIL import Image
import numpy as np

import torchvision.transforms as transforms

import config


def process(path, flag):
    path = os.path.join(path, flag)
    with open(os.path.join(path, flag) + ".json") as f:
        for line in f:
            item = json.loads(line)
            identifier = item["identifier"]
            dir_name = item["directory"]
            list_objs = item["structured_rep"]
            print(len(list_objs))
            # print(type(identifier))
            # print(type(dir_name))

            file_path = os.path.join(path, "images", dir_name, flag + "-" + identifier + "-0") + ".png"
            im = Image.open(file_path)
            print(im.size)
            # print(file_path)


def status(path, flag):
    path = os.path.join(path, flag)
    with open(os.path.join(path, flag) + ".json") as f:
        scences = []
        images = []
        for line in tqdm(f):
            item = json.loads(line)
            identifier = item["identifier"]
            dir_name = item["directory"]
            list_objs = item["structured_rep"]

            # print(len(list_objs))

            [scences.append(objs) for objs in list_objs]

            file_path = os.path.join(path, "images", dir_name, flag + "-" + identifier + "-0") + ".png"
            im = Image.open(file_path)

            image1, image2, image3 = im.crop((0, 0, 100, 100)), im.crop((150, 0, 250, 100)), im.crop(
                (300, 0, 400, 100))

            images.append(image1)
            images.append(image2)
            images.append(image3)

            # print(im.size)
            # print(file_path)
    return scences, images

# {"y_loc":21,"size":20,"type":"triangle","x_loc":27,"color":"Yellow"}

def show_status(scences):
    type_dict = {}
    color_dict = {}
    x_max, x_min = -1, 100
    y_max, y_min = -1, 100
    s_max, s_min = -1, 100
    max_len = 0
    for s in scences:
        print(len(s))
        max_len = max(len(s), max_len)
        for obj in s:
            x_max = max(x_max, obj["x_loc"])
            x_min = min(x_min, obj["x_loc"])

            y_max = max(y_max, obj["y_loc"])
            y_min = min(y_min, obj["y_loc"])

            s_max = max(s_max, obj["size"])
            s_min = min(s_min, obj["size"])

            type_dict[obj["type"]] = type_dict.get(obj["type"], 0) + 1
            color_dict[obj["color"]] = color_dict.get(obj["color"], 0) + 1

    print("X:", x_max, x_min)
    print("Y:", y_max, y_min)
    print("S:", s_max, s_min)
    print("Max_len:", max_len)

    return type_dict, color_dict


shape2id = {
    'triangle': 0,
    'circle': 1,
    'square': 2
}

color2id = {
    'Yellow': 0,
    '#0099ff': 1,
    'Black': 2
}


def get2onehottensor(index, max_num=3):
    y = torch.LongTensor([index])
    y_onehot = torch.FloatTensor(max_num).zero_()
    y_onehot.scatter_(0, y, 1)
    return y_onehot


def list2packed_tensor(seq, max_l=10):
    l = seq.size(0)
    if l >= max_l:
        return seq[:max_l, ]  # Truncate the length if the length is bigger than required padded_length.
    else:
        pad_seq = seq.new(max_l - l, *seq.size()[1:]).zero_()  # Requires_grad is False
        return torch.cat([seq, pad_seq], dim=0)


def obj2tensor(obj):
    x_loc = (obj["x_loc"] - 50) / 50
    y_loc = (obj["y_loc"] - 50) / 50
    size = (obj["size"] - 20) / 20
    t = shape2id[obj["type"]]
    c = color2id[obj["color"]]

    tensor_list = [torch.FloatTensor([x_loc]),
                   torch.FloatTensor([y_loc]),
                   torch.FloatTensor([size]),
                   get2onehottensor(t, max_num=3),
                   get2onehottensor(c, max_num=3),
                   ]

    tensor = torch.cat(tensor_list, dim=0)
    return tensor


def scence2tensor(scence, max_l=10):
    objs = []
    lens = torch.LongTensor([len(scence)])
    for obj in scence:
        x_loc = (obj["x_loc"] - 50) / 50
        y_loc = (obj["y_loc"] - 50) / 50
        size = (obj["size"] - 20) / 20
        t = shape2id[obj["type"]]
        c = color2id[obj["color"]]

        tensor_list = [torch.FloatTensor([x_loc]),
                       torch.FloatTensor([y_loc]),
                       torch.FloatTensor([size]),
                       get2onehottensor(t, max_num=3),
                       get2onehottensor(c, max_num=3),
                       ]

        tensor = torch.cat(tensor_list, dim=0)
        objs.append(tensor)

    return list2packed_tensor(torch.stack(objs, dim=0)), lens


def batching(scence_list, len_list):
    if not isinstance(scence_list, list):
        scence_list = [scence_list]
        len_list = [len_list]
    return torch.stack(scence_list, dim=0), torch.cat(len_list, dim=0)


def scences2input(scences):
    s_list, l_list = [], []
    for sce in scences:
        s_t, l_t = scence2tensor(sce, max_l=8)
        s_list.append(s_t)
        l_list.append(l_t)

    return batching(s_list, l_list)


def process_scences(scences):
    total_scence_tensor = []
    totoal_len_tensor = []
    # total_image_tensor = []
    for scence in scences:
        # print(scence)
        # print(scence2tensor(scence))
        scence, len = scence2tensor(scence)
        total_scence_tensor.append(scence)
        totoal_len_tensor.append(len)

    return torch.stack(total_scence_tensor, dim=0), torch.cat(totoal_len_tensor, dim=0)


def process_images(images):
    total_image_tensor = []

    trans_func = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # bring images to (-1,1)
    ])
    for im in images:
        im_tensor = trans_func(im)
        total_image_tensor.append(im_tensor[:3])

    return torch.stack(total_image_tensor, dim=0)


def load_data():
    scences = torch.load("scences.pth")
    images = torch.load("images.pth")
    return scences, images

# def image2tensor(image):
#     pass

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def count_bombine(coutner, image):
    np_im = np.asarray(image)
    np_rgb = np_im[:, :].reshape(-1, 4)[:, :3]
    # c = Counter()
    np_rgb_list = [(r, g, b) for r, g, b in np_rgb]
    # print(np_rgb_list)
    coutner.update(np_rgb_list)

if __name__ == '__main__':
    # process("/Users/Eason/RA/RL_net/data/nlvr/", "dev")
    scences, images = status(os.path.join(config.ROOT_DIR, "data/nlvr/"), "dev")

    # c_list_r = []
    # c_list_g = []
    # c_list_b = []
    # counter = Counter()
    # for im in tqdm(images):
    #     count_bombine(counter, im)
    #
    # print(counter)
    #     c_r = np.bincount(np.asarray(im)[0], minlength=256)
    #     c_g = np.bincount(np.asarray(im)[1], minlength=256)
    #     c_b = np.bincount(np.asarray(im)[2], minlength=256)
    #
    #     c_list_r.append(c_r)
    #     c_list_g.append(c_g)
    #     c_list_b.append(c_b)
    #
    # total_c_r = sum(c_list_r)
    #
    # print(np.nonzero(total_c_r)[0])

    # print(len(scences))
    # print(len(images))

    total_scence_tensor = process_scences(scences)
    print(total_scence_tensor[0].size())
    print(total_scence_tensor[1].size())
    print(total_scence_tensor[0][0])

    torch.save(total_scence_tensor, "scences_dev.pth")

    # images[100].save("saved_t_im_2.png")

    total_im_tensor = process_images(images)
    print(total_im_tensor.size())
    torch.save(total_im_tensor, "images_dev.pth")

    # t_index = 100
    # scences, images = load_data()
    #
    # ss, ls = scences
    # ims = images
    #
    # print(ss[t_index])
    # print(ls[t_index])
    # print(ims[t_index])
    #
    # t_im = Image.fromarray(tensor2im(ims[t_index]))
    # t_im.save("saved_t_im.png")
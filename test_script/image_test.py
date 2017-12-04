from collections import Counter

from PIL import Image
import numpy as np
import torch


def longtensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    return image_numpy.astype(imtype)

def count_combine():
    pass

if __name__ == '__main__':
    # image = Image.open("/Users/Eason/RA/RL_net/data/nlvr/train/images/0/train-655-0-0.png")
    # image1, image2, image3 = image.crop((0, 0, 100, 100)), image.crop((150, 0, 250, 100)), image.crop((300, 0, 400, 100))
    # [print(im.size) for im in [image1, image2, image3]]
    im = Image.open("test_im1.png")
    np_im = np.asarray(im)

    def count_bombine(coutner):
        np_rgb = np_im[:, :].reshape(-1, 4)[:, :3]
        c = Counter()
        np_rgb_list = [(r, g, b) for r, g, b in np_rgb]
        # print(np_rgb_list)
        coutner.update(np_rgb_list)
        # print(c)
    # print(np_im[:, :, 1])
    # print(np_im[:, :, 2])

    # print(np_im[:, :, 3])
    # tr_im = torch.from_numpy(np_im).long()
    # tr_im = tr_im.transpose(1, 2).transpose(0, 1)
    #
    # print(tr_im)
    # print(longtensor2im(tr_im))
    #
    # Image.fromarray(longtensor2im(tr_im)).save("test_im1_same.png")

    # image1.save("test_im1.png")
    # image2.save("test_im2.png")
    # image3.save("test_im3.png")

    # print(I.dtype)
    # print(I.shape)

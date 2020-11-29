import PIL.Image as Image
import numpy as np
import os

from utils.general import datasetTraverse

def convertDir2Image(dir_path, dst_path, width=256):

    def convertDir2ImageInner(count_, filep_, report_, list_, dict_, **kwargs):
        print("# %d"%count_, filep_, end=' ')

        folder_path = dst_path+kwargs["folder"]+'/'
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        convert(filep_, folder_path+kwargs['item'], img_width=width)

        return list_, dict_

    datasetTraverse(dir_path=dir_path,
                    exec_kernel=convertDir2ImageInner,
                    class_dir=True,
                    load_func=None)


def convert(bin_path, save_path, padding=False, fuzzy=None, img_width=256):
    '''
    单个图像的转换函数，返回Image对象\n
    path:文件的路径\n
    '''
    with open(bin_path, "rb") as f:
        print("Opened", end=" ")
        image = np.fromfile(f, dtype=np.byte)
        crop_w = int(image.shape[0] ** 0.5)
        image = image[:crop_w ** 2]
        image = image.reshape((crop_w, crop_w))
        image = np.uint8(image)
        if padding and crop_w < img_width:
            image = np.pad(image, (img_width - crop_w), 'constant', constant_values=(0))
        im = Image.fromarray(image)
        if fuzzy is not None:
            im = im.resize((fuzzy, fuzzy), Image.ANTIALIAS)
        im = im.resize((img_width, img_width), Image.ANTIALIAS)

    im.save(save_path + '.jpg', 'JPEG')
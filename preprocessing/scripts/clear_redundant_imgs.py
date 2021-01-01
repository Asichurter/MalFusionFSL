import os
from tqdm import tqdm

api_path = '/home/omnisky/NewAsichurter/FusionData/datasets/LargePE-Per35/all/api/'
img_path = '/home/omnisky/NewAsichurter/FusionData/datasets/LargePE-Per35/all/img/'

api_folders = os.listdir(api_path)
for folder in tqdm(os.listdir(img_path)):
    if folder not in api_folders:
        assert os.system(f"rm -rf {img_path+folder}") == 0
    else:
        for img_item in os.listdir(img_path+folder):\
            # '.jpg'四个字符占4个位置
            api_item = img_item[:-4]+'.json'

            if api_item not in os.listdir(api_path+folder):
                assert os.system(f"rm {img_path + folder + '/' + img_item}") == 0

        assert len(os.listdir(api_path+folder))==len(os.listdir(img_path+folder)), \
            "folder: %s, %d != %d"%(folder, len(os.listdir(api_path+folder)), len(os.listdir(img_path+folder)))

assert len(os.listdir(api_path))==len(os.listdir(img_path)), "%d != %d" % \
                 (len(os.listdir(api_path)),len(os.listdir(img_path)))


import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import torchvision.transforms.functional as TF
import random
import cv2
import re
import time
random.seed(1234)


def cv2_tensor(pic):
    # img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # img = img.view(pic.shape[0], pic.shape[1], 1)
    return TF.to_tensor(pic)

    #img = img.transpose(0, 2).transpose(1, 2).contiguous()
    # return img.float().div(255)

def replace_index_and_read(image_dir, indx, size):
    new_dir = image_dir[0:-15] + str(int(image_dir[-15:-12]) + indx).zfill(3) + image_dir[-12::]
    try:
        img = cv2.resize(cv2.imread(new_dir, 0), size)
    except:
        print ('orgin_dir: ' + image_dir)
        print ('new_dir: ' + new_dir)
    # center crop
    # img = cv2.resize(cv2.imread(new_dir, 0), size)
    frame = cv2_tensor(img)
    return frame

def imagetoframe(image_dir, size, num_frame):

    samples = [replace_index_and_read(image_dir, indx, size) for indx in range(num_frame)]
    return torch.stack(samples)

def get_path_list(image_dir, num_frame):
    new_dirs = [image_dir[1:-15] + str(int(image_dir[-15:-12]) + indx).zfill(3) + image_dir[-12::] for indx in range(num_frame)]
    return new_dirs

def read_datalist(path):
    datalist = []
    with open(path) as f:
        datalist = f.readlines()
    return [data.strip() for data in datalist]

def load_mask(mask_path, size):
    mask = cv2.resize(cv2.imread(mask_path), size)
    mask_blue = np.expand_dims(mask[:,:,0], 0) > 50
    mask_red = np.expand_dims(mask[:,:,2], 0) > 50
    mask_volume = np.concatenate([mask_blue, mask_red], 0).astype(int)
    mask_volume = torch.from_numpy(mask_volume).contiguous().type(torch.FloatTensor)
    return mask_volume

class Eyes(Dataset):
    def __init__(self, dataset_root,  datalist_path, num_frames=5, size=(128, 128), returnpath=False):
        self.datalist = read_datalist(datalist_path)
        print(f'size:{size}')
        self.size = [int(size[0]), int(size[1])]
        self.num_frame = num_frames
        self.dataset_root = dataset_root
        self.returnpath = returnpath

    def __len__(self):
        return len(self.datalist)


    
    def read_sample(self, img_path, size, num_frame):
        sample = []
        img = cv2_tensor(cv2.resize(cv2.imread(img_path), size))
        sample.append(img)
        # read all frames
        i = 1
        frame_path  = os.path.join(img_path.split('.jpg')[0], f'{i}.jpg')
        frames = []
        while os.path.isfile(frame_path):
            img = cv2_tensor(cv2.resize(cv2.imread(frame_path), size))
            frames.append(img)
            i += 1
            frame_path  = os.path.join(img_path.split('.jpg')[0], f'{i}.jpg')
        sample += frames
        mask = load_mask(os.path.join(img_path.split('.jpg')[0], f'{i - 1}.jpg'), size)
        return torch.stack(sample[:num_frame]), mask


    def __getitem__(self, idx):
        #sample = imagetoframe(self.dataset_root+self.datalist[idx].strip(), self.size, self.num_frame)
        img_path = os.path.join(self.dataset_root, self.datalist[idx])
        sample, mask = self.read_sample(img_path, self.size, self.num_frame)
        if self.returnpath:
            #paths = get_path_list(self.datalist[idx].strip(), self.num_frame)
            paths = ''
            return sample, mask, paths
        else:
            return sample,mask

def gen_datalist(dir_path):
    file_l = os.listdir(os.path.join(dir_path, 'video-cell'))
    img_l = []
    num_frames_l = []
    for file_name in file_l:
        if not '.jpg' in file_name:
            continue
        if not file_name.split('.jpg')[0]+'.mp4' in file_l:
            print(f'Error for {file_name}')
            continue
        vc = cv2.VideoCapture(os.path.join(dir_path, 'video-cell', file_name.split('.jpg')[0]+'.mp4'))
        num_frames = vc.get(cv2.CAP_PROP_FRAME_COUNT)
        num_frames_l.append(num_frames)
        img_l.append(file_name)
    train_l = random.sample(img_l, 70)
    test_l = list(set(img_l) - set(train_l))
    print(f'num frames set:{set(num_frames_l)}')
    with open(os.path.join(dir_path, 'trainlist.txt'), 'w') as f:
        for data in train_l:
            f.write(data + '\n')

    with open(os.path.join(dir_path, 'testlist.txt'), 'w') as f:
        for data in test_l:
            f.write(data + '\n')

def gen_frames(dir_path):
    file_l = os.listdir(os.path.join(dir_path, 'video-cell'))
    img_l = []
    num_frames_l = []
    for file_name in file_l:
        if not '.mp4' in file_name:
            continue
        cap = cv2.VideoCapture(os.path.join(dir_path, 'video-cell', file_name))
        if not cap.isOpened():
            print(f"无法打开视频文件{os.path.join(dir_path, 'video-cell', file_name)}")
            continue
        # 帧计数器
        frame_count = 0
        # 读取每一帧并保存到文件夹中
        video_dir = os.path.join(dir_path, 'video-cell', file_name.split('.mp4')[0])
        # os.mkdir(video_dir)
        while True:
            # 读取帧
            ret, frame = cap.read()
            # 如果没有读取到帧，则退出循环
            if not ret:
                break
            # 构造保存帧的文件名
            filename = f'{frame_count}.jpg'
            # 保存帧到文件夹中
            cv2.imwrite(os.path.join(video_dir, filename), frame)
            # 增加帧计数器
            frame_count += 1

        # 释放视频资源
        cap.release()

if __name__ == '__main__':

    gen_frames('/home/wzp/workplace/dataset')
    # start_time = time.time()
    # cityscapes_Dataset = KTH(dataset_root='/mnt/lustrenew/panjunting/kth/KTH/processed', datalist='kth_train_16.txt',
    #                             size=(128, 128), num_frames=16)

    # dataloader = DataLoader(cityscapes_Dataset, batch_size=32, shuffle=False, num_workers=1)

    # sample = iter(dataloader).next()
    # print (sample.size())
    # # from tqdm import tqdm
    # a= [ 1 for sample in tqdm(iter(dataloader))]

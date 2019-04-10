'''
Author  : Zhengwei Li
Version : 1.0.0 
'''

import cv2
import os
import random as r
import numpy as np

import torch
import torch.utils.data as data


# ============================================================================================================
def crop_patch_augment(_img, _mask, _alpha, patch):


    (h, w, c) = _img.shape
    scale = 0.75 + 0.5*r.random()

    _img = cv2.resize(_img, (int(w*scale),int(h*scale)), interpolation=cv2.INTER_CUBIC)
    _mask = cv2.resize(_mask, (int(w*scale),int(h*scale)), interpolation=cv2.INTER_NEAREST)
    _alpha = cv2.resize(_alpha, (int(w*scale),int(h*scale)), interpolation=cv2.INTER_CUBIC)

    (h, w, c) = _img.shape

    if r.random() < 0.5:
        if h>patch and w>patch:
            x = r.randrange(0, (w - patch))
            y = r.randrange(0, (h - patch))
            

            _img = _img[y:y + patch, x:x + patch, :]
            _mask = _mask[y:y + patch, x:x + patch, :]
            _alpha = _alpha[y:y + patch, x:x + patch, :]
        else:

            _img = cv2.resize(_img, (patch,patch), interpolation=cv2.INTER_CUBIC)
            _mask = cv2.resize(_mask, (patch,patch), interpolation=cv2.INTER_NEAREST)
            _alpha = cv2.resize(_alpha, (patch,patch), interpolation=cv2.INTER_CUBIC)
    else:

        _img = cv2.resize(_img, (patch,patch), interpolation=cv2.INTER_CUBIC)
        _mask = cv2.resize(_mask, (patch,patch), interpolation=cv2.INTER_NEAREST)
        _alpha = cv2.resize(_alpha, (patch,patch), interpolation=cv2.INTER_CUBIC)


    # flip
    if r.random() < 0.5:
        _img = cv2.flip(_img,0)
        _mask = cv2.flip(_mask,0)
        _alpha = cv2.flip(_alpha,0)

    if r.random() < 0.5:
        _img = cv2.flip(_img,1)
        _mask = cv2.flip(_mask,1)
        _alpha = cv2.flip(_alpha,1)

    return _img, _mask, _alpha

def im_bg_augment(_img, _mask):

    if r.random() < 0.2:
        _img_portrait = np.multiply(_mask, _img)
        _img_bg = np.multiply(1 - _mask, _img)

        _img_bg[:,:,0] = np.multiply(np.random.rand()+0.2, _img_bg[:,:,0])
        _img_bg[:,:,1] = np.multiply(np.random.rand()+0.2, _img_bg[:,:,1])
        _img_bg[:,:,2] = np.multiply(np.random.rand()+0.2, _img_bg[:,:,2])

        _img_bg[_img_bg>=1.0] = 1.0 
        _img_new = _img_bg + _img_portrait
    else:
        _img_new = _img

    return _img_new


def np2Tensor(array):
    ts = (2, 0, 1)
    tensor = torch.FloatTensor(array.transpose(ts).astype(float))    
    return tensor
"""
    dataset: human_matting 
"""
class human_matting(data.Dataset):

    def __init__(self, base_dir, imglist, patch):

        super().__init__()
        self._base_dir = base_dir
        with open(os.path.join(self._base_dir, imglist)) as f:
            self.file_list = f.readlines()

        self.file_list = self.file_list
        self.data_num = len(self.file_list)
        self.patch = patch
        print("Dataset : Ulsee coco !")
        print('file number %d' % self.data_num)


    def __getitem__(self, index):

        _img_name, _target_name = self.getFileName(index)

        _img = cv2.imread(_img_name).astype(np.float32)
        # bright
        if r.random() < 0.5:
            if r.random() < 0.5:
                _img = np.uint8(np.clip(_img + r.randrange(0, 45), 0, 255))
            else:
                _img = np.uint8(np.clip(_img - r.randrange(0, 45), 0, 255))            

        _img = (_img - (104., 112., 121.,)) / 255.0
        _mask = cv2.imread(_target_name).astype(np.float32) #(0,1)

        _alpha = _mask
        _img = im_bg_augment(_img, _mask)
        _img, _mask, _alpha = crop_patch_augment(_img, _mask, _alpha, self.patch)

        _img = np2Tensor(_img)
        _mask = np2Tensor(_mask)
        _alpha = np2Tensor(_alpha)

        _mask = _mask[0,:,:].unsqueeze_(0)
        _alpha = _alpha[0,:,:].unsqueeze_(0)

        sample = {'image': _img, 'mask': _mask, 'alpha': _alpha}

        return sample

    def __len__(self):
        return self.data_num

    def getFileName(self, idx):

        line = self.file_list[idx]
        line = line.replace(' ', '\t')
        name = line.split('\t')[0]
        nameIm = os.path.join(self._base_dir, name)
        name = line.split('\t')[1].split('\n')[0]
        nameTar = os.path.join(self._base_dir, name)

        return nameIm, nameTar

'''
# -----------------------------------------
'''
import numpy as np
import random
import h5py
import torch.utils.data as data
import utils.utils_image as util
from utils.utils_oswinmri import *
from data.sampling import gaussian_pattern

from math import floor


class DatasetCCks(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for image-to-image mapping.
    # Both "paths_L" and "paths_H" are needed.
    # -----------------------------------------
    # e.g., train denoiser with L and H
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetCCks, self).__init__()
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
        self.opt = opt
        self.n_channels = self.opt['n_channels']
        self.patch_size = self.opt['H_size']
        self.is_noise = self.opt['is_noise']
        self.noise_level = self.opt['noise_level']
        self.noise_var = self.opt['noise_var']
        self.is_mini_dataset = self.opt['is_mini_dataset']
        self.mini_dataset_prec = self.opt['mini_dataset_prec']
        self.acc = self.opt['acc']
        self.kx_size = self.opt['kx_size']
        self.ky_size = self.opt['ky_size']
        self.center_size = self.opt['center']
        self.normalize = self.opt['normalize']
        self.augment = self.opt['augment']
        self.nslices = 160
        # ------------------------------------
        # get the path of L/H
        # ------------------------------------
        self.paths_raw = util.get_h5_paths(opt['dataroot_H'])
        assert self.paths_raw, 'Error: Raw path is empty.'

        self.paths_H = []

        for path in self.paths_raw:
            if 'e1' in path:
                self.paths_H.append(path)
            else:
                assert 0, 'Error: Unknown filename is in raw path'
        
        

        if self.is_mini_dataset:

            index = list(range(0, len(self.paths_H)))
            index_chosen = random.sample(index, round(self.mini_dataset_prec * len(self.paths_H)))
            self.paths_H_new = []
            for i in index_chosen:
                self.paths_H_new.append(self.paths_H[i])
            self.paths_H = self.paths_H_new
        # ------------------------------------
        # get mask
        # ------------------------------------

        # self.mask = define_Mask(self.opt)
        if self.opt['mask'] == 'generate_gaussian':
            self.mask = self.generate_mask((self.kx_size,self.ky_size), acc=self.acc, dim = '2D', full_center = self.center_size)
        else:
            self.mask = self.get_mask(self.opt['mask'])

    def __getitem__(self, index):

        
        is_noise = self.is_noise
        noise_level = self.noise_level
        noise_var = self.noise_var

        # ------------------------------------
        # get H image
        # ------------------------------------

        file_id = index//self.nslices
        file_slice = index%self.nslices
        H_path = self.paths_H[floor(file_id)]
        # Load data
        with h5py.File(H_path, 'r') as f:
            kspace = f['kspace']
            if kspace.shape[2] == 170:
                ks_H = kspace[48+file_slice]
            else:
                idx = int((kspace.shape[2] - 170)/2)
                ks_H = kspace[48+file_slice,:,idx:-idx,:]

        # ks_H = np.fft.fftshift(ks_H,axes=(0))

        if self.opt['phase'] == 'train' and self.augment:
            self.mode = random.randint(0, 3)
            ks_H = util.augment_img_tensor4_simple(ks_H, mode=self.mode)
            self.mask = util.augment_img_tensor4_simple(self.mask, mode=self.mode)

        kx,ky,C = ks_H.shape[0],ks_H.shape[1],ks_H.shape[2]
        img_H = np.empty((kx, ky, C))
        img_L = np.empty((kx, ky, C))

        img_Hc = np.fft.ifft2(ks_H[:,:,::2]+1j*ks_H[:,:,1::2],axes = (0,1))
        if self.normalize:
            img_Hcn = img_Hc/1000
            ks_Hn = ks_H/1000
        else:
            img_Hcn = img_Hc
            ks_Hn = ks_H
        
        # img_H = np.sqrt((np.abs(img_Hcn)**2).sum(axis = -1,keepdims=True))
        # img_H = img_H/np.amax(img_H)
        img_H[:,:,::2] = np.real(img_Hcn)
        img_H[:,:,1::2] = np.imag(img_Hcn)
        
        # mask = self.generate_mask((kx,ky), acc=self.acc, dim = '2D')
        ks_masked = ks_Hn * self.mask

        img_Lc = np.fft.ifft2(ks_masked[:,:,::2]+1j*ks_masked[:,:,1::2],axes = (0,1))

        # print(np.amax(np.real(img_Hcn)),'  ',np.amin(np.real(img_Hcn)),'  ',np.amax(np.real(ks_Hn)),'  ',np.amin(np.real(ks_Hn)),'  ',np.amax(np.real(img_Lc)),'  ',np.amin(np.real(img_Lc)),'  ')
        # print(np.amax(np.imag(img_Hcn)),'  ',np.amin(np.imag(img_Hcn)),'  ',np.amax(np.imag(ks_Hn)),'  ',np.amin(np.imag(ks_Hn)),'  ',np.amax(np.imag(img_Lc)),'  ',np.amin(np.imag(img_Lc)),'\n')

        # img_L = np.sqrt((np.abs(img_Lc)**2).sum(axis = -1,keepdims=True))
        # img_L = img_L/np.amax(img_L)
        img_L[:,:,::2] = np.real(img_Lc)
        img_L[:,:,1::2] = np.imag(img_Lc)



        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, _ = img_H.shape

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            
            
            img_L, img_H = util.float2tensor3(img_L), util.float2tensor3(img_H)
            ks_masked, mask = util.float2tensor3(ks_masked), util.float2tensor3(1 - self.mask)
            

        else:

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            # print('GT')
            # print(np.amax(img_H),'  ',np.amin(img_H),'  ')
            # print('ZF')
            # print(np.amax(img_L),'  ',np.amin(img_L),'  ')

            # if self.augment:
            #     self.mode = 2
            #     img_L, img_H = util.augment_img_tensor4_simple(img_L, mode=self.mode), util.augment_img_tensor4_simple(img_H, mode=self.mode)
            #     ks_masked, mask = util.augment_img_tensor4_simple(ks_masked, mode=self.mode), util.augment_img_simple(self.mask, mode=self.mode)
            # else:
            #     mask = self.mask

            # img_L, img_H = util.float2tensor3(img_L), util.float2tensor3(img_H)
            # ks_masked, mask = util.float2tensor3(ks_masked), util.float2tensor3(1 - mask)

            img_L, img_H = util.float2tensor3(img_L), util.float2tensor3(img_H)
            ks_masked, mask = util.float2tensor3(ks_masked), util.float2tensor3(1 - self.mask)


        return {'L': img_L, 'H': img_H, 'K': ks_masked, 'mask': mask, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)*self.nslices

    def load_images(self, H_path):
        # load GT
        gt = np.load(H_path).astype(np.float32)

        return gt

    def load_ks(self, H_path):
        # load fully sampled k-space data
        ks = np.load(H_path).astype(np.float64)

        return ks

    def load_h5(self, H_path):
        pass


    def undersample_kspace(self, ks, mask, is_noise, noise_level, noise_var):

        ks_masked = ks * mask
        if is_noise:
            ks_masked = ks_masked + self.generate_gaussian_noise(ks_masked, noise_level, noise_var)
 
        return ks_masked

    def generate_gaussian_noise(self, x, noise_level, noise_var):
        spower = np.sum(x ** 2) / x.size
        npower = noise_level / (1 - noise_level) * spower
        noise = np.random.normal(0, noise_var ** 0.5, x.shape) * np.sqrt(npower)
        return noise

    def generate_mask(self, shape, acc = 0.5, dim = '2D', full_center=5):
        umask = gaussian_pattern(shape, factor = acc, dim = dim, full_center = full_center)
        return np.expand_dims(umask, axis = -1)

    def get_mask(self, mask_name):
        return np.expand_dims(np.load('mask/'+mask_name+'.npy'), axis = -1)
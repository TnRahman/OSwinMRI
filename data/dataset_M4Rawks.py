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


class DatasetM4Rawks(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for image-to-image mapping.
    # Both "paths_L" and "paths_H" are needed.
    # -----------------------------------------
    # e.g., train denoiser with L and H
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetM4Rawks, self).__init__()
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
        self.nslices = 18
        # ------------------------------------
        # get the path of L/H
        # ------------------------------------
        self.paths_raw = util.get_h5_paths(opt['dataroot_H'])
        assert self.paths_raw, 'Error: Raw path is empty.'

        self.paths_H = []

        for path in self.paths_raw:
            if '_T10' in path:
                self.paths_H.append(path)
            else:
                pass
        
        

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

        self.all_masks = self.load_masks()
        
        # if self.opt['mask'] == 'generate_gaussian':
            # self.mask = self.generate_mask((self.kx_size,self.ky_size), acc=self.acc, dim = '2D', full_center = self.center_size)
        # else:
            # self.mask = self.get_mask(self.opt['mask'])

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
            ks_H = kspace[file_slice,:,:,:].transpose(1,2,0)

        if self.opt['phase'] == 'train':
            if self.opt['mask'] == 'random':
                self.mask = self.get_random_mask(self.acc)
            else:
                self.mask = self.get_mask(self.acc)
        else:
            self.mask = self.get_mask(self.acc)

        if self.opt['phase'] == 'train' and self.augment:
            self.mode = random.randint(0, 3)
            ks_H = util.augment_img_tensor4_simple(ks_H, mode=self.mode)
            self.mask = util.augment_img_tensor4_simple(self.mask, mode=self.mode)

        kx,ky,C = ks_H.shape[0],ks_H.shape[1],ks_H.shape[2]
        img_H = np.empty((kx, ky, self.n_channels))
        img_L = np.empty((kx, ky, self.n_channels))

        img_Hc = ifft2n(ks_H,dim=(0,1),centered=True,normalized=False)
        
        
        # img_H = np.sqrt((np.abs(img_Hcn)**2).sum(axis = -1,keepdims=True))
        # img_H = img_H/np.amax(img_H)
        img_H[:,:,::2] = np.real(img_Hc)
        img_H[:,:,1::2] = np.imag(img_Hc)
        
        # mask = self.generate_mask((kx,ky), acc=self.acc, dim = '2D')
        ks_m = ks_H * self.mask

        img_Lc = ifft2n(ks_m,dim=(0,1),centered=True,normalized=False)
        
        ks_masked = np.empty((kx, ky, self.n_channels))
        ks_masked[:,:,::2] = np.real(ks_m)
        ks_masked[:,:,1::2] = np.imag(ks_m)
        
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

    # def get_mask(self, mask_name):
    #     return np.expand_dims(np.load('mask/'+mask_name+'.npy'), axis = -1)

    def get_mask(self, acc):
        if acc[0] == 2:
            umask = np.expand_dims(self.all_masks[0][0,:,:], axis = -1)
        elif acc[0] == 3:
            umask = np.expand_dims(self.all_masks[1][0,:,:], axis = -1)
        else:
            umask = np.expand_dims(gaussian_pattern((256,256), factor = 1/int(acc[0]), dim = '1D', full_center = 30), axis = -1)
        return umask
    
    def get_random_mask(self, acc):
        if len(acc)==2:
            us_rate = random.randint(0, 1)  #  0 -- 2x, 1 -- 3x
        elif acc[0]==2:
            us_rate = 0
        else:
            us_rate = 1
        us_mask = random.randint(0, 99)
        if us_rate == 0:
            umask = np.expand_dims(self.all_masks[0][us_mask,:,:], axis = -1)
        else:
            umask = np.expand_dims(self.all_masks[1][us_mask,:,:], axis = -1)
        return umask
    
    def load_masks(self):
        masks_r2 = np.load('mask/R2_1DG_256x256.npy')
        masks_r3 = np.load('mask/R3_1DG_256x256.npy')
        return masks_r2, masks_r3
'''
# -----------------------------------------
Model SwinV2IM Cascade - V4 - M4Raw Dataset
by Tahsin Rahman (trahman3@miners.utep.edu)

Thanks:
https://github.com/ayanglab/SwinMR
https://github.com/JingyunLiang/SwinIR
https://github.com/microsoft/Swin-Transformer
# -----------------------------------------
'''
import sys
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G
from models.model_base import ModelBase
from models.loss import CharbonnierLoss, PerceptualLoss
from models.weighting_params import DCWeightsCC, DCWeights
from models.loss_ssim import SSIMLoss

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip
from utils.utils_oswinmri import *
from utils.cosine_annealing_warmup import CosineAnnealingWarmupRestarts


class MRI_SwinV2IM_KS4_M4RAW(ModelBase):

    def __init__(self, opt):
        super(MRI_SwinV2IM_KS4_M4RAW, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.opt_test = self.opt['test']    # training option
        self.net_num = self.opt_train['net_num']

        if self.opt_train['residual_DC']:
            self.dcw = []
            for i in range(self.net_num):
                self.dcw.append(DCWeightsCC().to(self.device))

        self.nets = []
        for i in range(self.net_num):
            self.nets.append(define_G(opt,i))

        for i in range(self.net_num):
            self.nets[i] = self.model_to_device(self.nets[i])

        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()


    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        for i in range(self.net_num):
            self.nets[i].train()
            if self.opt_train['residual_DC']:
                self.dcw[i].train()
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log
        self.use_patch = self.opt['datasets']['train']['use_patch']
        self.patch_num = self.opt['datasets']['train']['num_patches']
        self.dc_step = self.opt_train['DC_step']

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    # INCOMPLETE -----------------------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            for i in range(self.net_num):
                print('Loading model for G{:d} [{:s}] ...'.format(i+1,load_path_G))
                model_path = load_path_G+'{}.pth'.format(i+1)
                
                self.load_network(model_path, self.nets[i], strict=self.opt_train['G_param_strict'], param_key='params')
                if self.opt_train['residual_DC']:
                    dcw_path = load_path_G+'{}_dcw.pth'.format(i+1)
                    self.dcw[i].load_dcw(dcw_path, self.dcw[i], strict=self.opt_train['G_param_strict'], param_key='params')

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG+'.pth', self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        for i in range(self.net_num):
            self.save_network(self.save_dir, self.nets[i], 'G'+str(i+1), iter_label)
            if self.opt_train['residual_DC']:
                self.dcw[i].save_dcw(self.save_dir, self.dcw[i], 'G'+str(i+1)+'_dcw', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

   # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn_type = 'l1'
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn_type = 'l2'
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn_type = 'ssim'
            self.G_lossfn = SSIMLoss().to(self.device)
        elif G_lossfn_type == 'l1_ssim':
            self.G_lossfn_type = 'l1_ssim'
            self.G_lossfn2 = SSIMLoss().to(self.device)
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']
        # self.perceptual_lossfn = nn.L1Loss().to(self.device)
        self.perceptual_lossfn = PerceptualLoss().to(self.device)
        self.G_freq_lossfn = nn.L1Loss().to(self.device)


    def total_loss(self, current_step):
        # ksc = fft2t(imspace_complex_crop,dim=(0,1),centered=True,normalized=True)

        self.alpha = self.opt_train['alpha']
        self.beta = self.opt_train['beta']
        self.gamma = self.opt_train['gamma']

        if self.opt_train['cascade_loss_type'] == 'DS':
            self.loss_image = 0
            self.H_c = self.H[:,::2,:,:] + 1j*self.H[:,1::2,:,:]
            self.net_num_sum = 10
            for i in range(self.net_num):
                self.E_c = self.E_out[i+1][:,::2,:,:] + 1j*self.E_out[i+1][:,1::2,:,:]
                self.loss_image = self.loss_image + self.G_lossfn(self.E_c.abs(), self.H_c.abs()) * (i+1)/self.net_num_sum

        elif self.opt_train['cascade_loss_type'] == 'simple':
            self.H_c = self.H[:,::2,:,:] + 1j*self.H[:,1::2,:,:]
            self.E_c = self.E[:,::2,:,:] + 1j*self.E[:,1::2,:,:]
            self.H_rss = torch.sqrt(((self.H_c.abs())**2).sum(dim = 1, keepdim = True))
            self.E_rss = torch.sqrt(((self.E_c.abs())**2).sum(dim = 1, keepdim = True))

            self.loss_image = self.G_lossfn(self.E_c.abs(), self.H_c.abs())
        
        elif self.opt_train['cascade_loss_type'] == 'kspace':
            self.H_c = self.H[:,::2,:,:] + 1j*self.H[:,1::2,:,:]
            self.E_c = self.E[:,::2,:,:] + 1j*self.E[:,1::2,:,:]
            self.H_k = fft2t(self.H_c,dim=(-2,-1),centered=True,normalized=True)
            self.E_k = fft2t(self.E_c,dim=(-2,-1),centered=True,normalized=True)

            self.loss_image = self.G_lossfn(torch.mul(self.mask,self.E_k.real), torch.mul(self.mask,self.H_k.real)) 
            + self.G_lossfn(torch.mul(self.mask,self.E_k.imag), torch.mul(self.mask,self.H_k.imag))

            # self.H_rss = torch.sqrt(((self.H_c.abs())**2).sum(dim = 1, keepdim = True))
            # self.E_rss = torch.sqrt(((self.E_c.abs())**2).sum(dim = 1, keepdim = True))

            # self.ccf_loss = np.clip((100000-current_step)/100000,0,1)
            # self.loss_image = self.ccf_loss*self.G_lossfn(self.E_c.abs(), self.H_c.abs()) + (1-self.ccf_loss)*self.G_lossfn(self.E_rss, self.H_rss)
            # self.loss_image = self.G_lossfn(self.E_c.abs(), self.H_c.abs())
        
        else:

            self.H_c = self.H[:,::2,:,:] + 1j*self.H[:,1::2,:,:]
            self.E_c = self.E[:,::2,:,:] + 1j*self.E[:,1::2,:,:]

            self.H_mag = torch.sqrt(((self.H_c.abs())**2).sum(dim = 1, keepdim = True))
            self.E_mag = torch.sqrt(((self.E_c.abs())**2).sum(dim = 1, keepdim = True))

            self.H_mag = self.H_mag / torch.max(self.H_mag)
            self.E_mag = self.E_mag / torch.max(self.E_mag)

            self.data_range = torch.max(torch.max(self.H_mag),torch.max(self.E_mag)) - torch.min(torch.min(self.E_mag),torch.min(self.H_mag))

            self.H_k = fft2t(self.H_c,dim=(-2,-1),centered=True,normalized=True)
            self.E_k = fft2t(self.E_c,dim=(-2,-1),centered=True,normalized=True)


            if self.G_lossfn_type == 'l1_ssim':
                self.loss_image = self.G_lossfn(self.E_c.abs(), self.H_c.abs()) + 1000*(self.G_lossfn2(self.E_mag, self.H_mag, self.data_range.reshape(1)))
                # self.loss_image = self.G_lossfn(self.E_c.abs(), self.H_c.abs())
            elif self.G_lossfn_type == 'ssim':
                self.loss_image = self.G_lossfn(self.E_mag, self.H_mag, self.data_range.reshape(1))
            else:
                self.loss_image = self.G_lossfn(self.E_c.abs(), self.H_c.abs())

        self.loss_freq = self.G_lossfn(self.E_c.abs(), self.H_c.abs())
        self.loss_perc = self.G_lossfn(self.E_c.abs(), self.H_c.abs())

        return self.alpha * self.loss_image + self.beta * self.loss_freq + self.gamma * self.loss_perc

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for i in range(self.net_num):
            for k, v in self.nets[i].named_parameters():
                if v.requires_grad:
                    G_optim_params.append(v)
                else:
                    print('Params [{:s}] will not optimize.'.format(k))
            if self.opt_train['residual_DC']:
                for k, v in self.dcw[i].named_parameters():
                    if v.requires_grad:
                        G_optim_params.append(v)
                    else:
                        print('Params [{:s}] will not optimize.'.format(k))

        
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLRwithWarmup':
            def warmup(current_step: int):
                self.all_milestones = self.opt_train['G_scheduler_milestones']
                if current_step<self.all_milestones[0]:
                    return 0.01
                elif current_step<self.all_milestones[1]:
                    return 0.1
                elif current_step<self.all_milestones[2]:
                    return 1
                elif current_step<self.all_milestones[3]:
                    return 0.8
                elif current_step<self.all_milestones[-1]:
                    return 0.6
                else:
                    return 0.5
            self.schedulers.append(lr_scheduler.LambdaLR(self.G_optimizer, lr_lambda=warmup))
        elif self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmupRestarts':
            self.all_milestones = self.opt_train['G_scheduler_milestones']
            self.schedulers.append(CosineAnnealingWarmupRestarts(self.G_optimizer, 
                                    first_cycle_steps=self.all_milestones[1], 
                                    cycle_mult=1.0, 
                                    max_lr=self.opt_train['G_optimizer_lr'], 
                                    min_lr=0.01*self.opt_train['G_optimizer_lr'],
                                    warmup_steps=self.all_milestones[0], 
                                    gamma=self.opt_train['G_scheduler_gamma']))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer,
                                                            self.opt_train['G_scheduler_periods'],
                                                            1,
                                                            self.opt_train['G_scheduler_eta_min']
                                                            ))
        else:
            raise NotImplementedError

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.H = data['H'].to(self.device)
        self.L = data['L'].to(self.device)
        self.K = data['K'].to(self.device)
        self.mask = data['mask'].to(self.device)
        self.L.requires_grad_()

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self, mode = 'train', step = 0, dc_step = 0):

        if mode == 'train':

            if self.use_patch:

                if self.patch_num == 'all':

                    self.E_out = []
                    self.E_out.append(self.L)

                    # Loop through patches
                    for i in range(self.net_num):
                        self.patch_size = self.opt['datasets']['train']['H_size']

                        self.height,self.width = self.E_out[-1].shape[2:]

                        self.patch_overlap_h = 32
                        self.patch_overlap_w = 32

                        self.stride_h = self.patch_size - self.patch_overlap_h
                        self.stride_w = self.patch_size - self.patch_overlap_w
                        self.h_idx_list = list(np.arange(0, self.height-self.patch_size, self.stride_h, dtype=np.int32)) + [self.height-self.patch_size]
                        self.w_idx_list = list(np.arange(0, self.width-self.patch_size, self.stride_w, dtype=np.int32)) + [self.width-self.patch_size]

                        self.E_model = torch.zeros_like(self.E_out[-1])
                        self.pw = torch.zeros_like(self.E_out[-1])

                        for h_idx in self.h_idx_list:
                            for w_idx in self.w_idx_list:
                            
                                self.L_patch = self.E_out[-1][..., h_idx:h_idx+self.patch_size, w_idx:w_idx+self.patch_size]

                                self.E_patch = self.nets[i](self.L_patch)
                                self.out_patch_mask = torch.ones_like(self.E_patch)

                                self.E_model[..., h_idx:(h_idx+self.patch_size), w_idx:(w_idx+self.patch_size)].add_(self.E_patch)
                                self.pw[..., h_idx:(h_idx+self.patch_size), w_idx:(w_idx+self.patch_size)].add_(self.out_patch_mask)

                        self.E_out_net = self.E_model.div_(self.pw)

                        self.E_out.append(data_consistency_fastmri(self.E_out_net, self.K, self.mask, step=step, dc_step=dc_step))

                    self.E = self.E_out[-1]


                else:

                    # Denoise one or more random patchs
                    self.num_patches = self.patch_num
                    self.patch_size = self.opt['datasets']['train']['H_size']
                    self.height,self.width = self.L.shape[2:]

                    self.E_model = torch.zeros_like(self.L)
                    self.pw = torch.zeros_like(self.L)

                    for i in range(self.num_patches):
                        self.rand_h = torch.randint(0,self.height-self.patch_size,(1,))
                        self.rand_w = torch.randint(0,self.width-self.patch_size,(1,))
                    
                        self.L_patch = self.L[..., self.rand_h:self.rand_h+self.patch_size, self.rand_w:self.rand_w+self.patch_size]
                        self.E_patch = self.netG(self.L_patch)
                        self.out_patch_mask = torch.ones_like(self.E_patch)

                        self.E_model[..., self.rand_h:self.rand_h+self.patch_size, self.rand_w:self.rand_w+self.patch_size].add_(self.E_patch)
                        self.pw[..., self.rand_h:self.rand_h+self.patch_size, self.rand_w:self.rand_w+self.patch_size].add_(self.out_patch_mask)

                    self.E_out1 = self.E_model.div_(torch.clamp(self.pw,1,self.num_patches))
                    self.pw_mask = torch.clamp(self.pw,0,1)
                    self.E_out = torch.add(self.E_out1,torch.mul(self.L,1-self.pw_mask))

                    if self.opt_train['use_DC']:
                        self.E = data_consistency_fastmri(self.E_out, self.K, self.mask, step=step, dc_step=dc_step)
                    else:
                        self.E = self.E_out

            else:
                # Use full-sized data
                self.E_out = []
                self.E_out.append(self.L)
                if self.opt_train['use_DC']:
                    for i in range(self.net_num):
                        if self.opt_train['residual_DC']:
                            self.E_out.append(data_consistency_unrolled(self.nets[i](self.E_out[i]), self.E_out[i], self.K, self.mask, self.dcw[i], step=step, dc_step=dc_step))
                        else:
                            self.E_out.append(data_consistency_fastmri(self.nets[i](self.E_out[i]), self.K, self.mask, step=step, dc_step=dc_step))
                else:
                    for i in range(self.net_num):
                        self.E_out.append(self.nets[i](self.E_out[i]))
                self.E = self.E_out[-1]


        elif mode == 'test':
            self.E_out = []
            self.E_out.append(self.L)
            if self.opt_test['use_DC']:
                for i in range(self.net_num):
                    if self.opt_train['residual_DC']:
                        self.E_out.append(data_consistency_unrolled(self.nets[i](self.E_out[i]), self.E_out[i], self.K, self.mask, self.dcw[i]))
                    else:
                        self.E_out.append(data_consistency_fastmri(self.nets[i](self.E_out[i]), self.K, self.mask))
            else:
                for i in range(self.net_num):
                    self.E_out.append(self.nets[i](self.E_out[i]))
            self.E = self.E_out[-1]

        else:
            sys.exit('Mode needs to be \'train\' or \'test\'')


    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()

        self.netG_forward(mode = 'train', step = current_step, dc_step = self.dc_step)

        G_loss = self.G_lossfn_weight * self.total_loss(current_step)
        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'],
                                           norm_type=2)

        self.G_optimizer.step()


        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train[
            'G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:

            for i in range(self.net_num):
                self.nets[i].apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train[
            'G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:

            for i in range(self.net_num):
                self.nets[i].apply(regularizer_clip)
        # ------------------------------------
        # record log
        # ------------------------------------
        self.log_dict['G_loss'] = G_loss.item()
        self.log_dict['G_loss_image'] = self.loss_image.item()
        self.log_dict['G_loss_frequency'] = self.loss_freq.item()
        self.log_dict['G_loss_preceptual'] = self.loss_perc.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        for i in range(self.net_num):
            self.nets[i].eval()
            if self.opt_train['residual_DC']:
                self.dcw[i].eval()
        with torch.no_grad():
            self.netG_forward(mode='test')
        for i in range(self.net_num):
            self.nets[i].train()
            if self.opt_train['residual_DC']:
                self.dcw[i].train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        out_dict['K'] = self.K.detach()[0].float().cpu()
        out_dict['mask'] = self.mask.detach()[0].float().cpu()
        out_dict['E_full'] = self.E_out
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()

        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msgs = []
        for i in range(self.net_num):
            msgs.append(self.describe_network(self.nets[i]))
        print(msgs)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msgs = []
        for i in range(self.net_num):
            msgs.append(self.describe_params(self.nets[i]))
        print(msgs)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msgs = []
        for i in range(self.net_num):
            msgs.append(self.describe_network(self.nets[i]))
        return msgs

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msgs = []
        for i in range(self.net_num):
            msgs.append(self.describe_params(self.nets[i]))
        return msgs

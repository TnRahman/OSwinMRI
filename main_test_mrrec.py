import argparse

import sys
import math
import pickle
import os
import torch
from utils import utils_option as option
from torch.utils.data import DataLoader
from utils import utils_image as util

from data.select_dataset import define_Dataset
from models.select_model import define_Model


def main(json_path):

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--test_data', type=str, default=None, help='Path to test data')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    data_path = parser.parse_args().test_data
    if data_path is not None and os.path.exists(data_path):
        opt['datasets']['test']['dataroot_H'] = data_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(opt['model_path']):
        print(f"loading model from {opt['model_path']}")
    else:
        print('can\'t find model.')

    model = define_Model(opt)
    model.load()


    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True,
                                                   seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    print(test_set.__len__())

    test_num = opt['test']['test_count']
    save_num = opt['test']['save_count']

    test_metrics = {}
    test_metrics['test_psnr'] = {}
    test_metrics['test_ssim'] = {}

    zf_metrics = {}
    zf_metrics['zf_psnr'] = {}
    zf_metrics['zf_ssim'] = {}

    current_step = 100000

    avg_psnr = 0.0
    avg_ssim = 0.0

    current_img = 1
    for idx, test_data in enumerate(test_loader):

        model.feed_data(test_data)
        model.test()

        visuals = model.current_visuals()

        E_img = util.stensor2uint(visuals['E'])
        H_img = util.stensor2uint(visuals['H'])
        L_img = util.stensor2uint(visuals['L'])

        E_rss = util.cimtensor2rss(visuals['E'])
        H_rss = util.cimtensor2rss(visuals['H'])
        L_rss = util.cimtensor2rss(visuals['L'])

        E_npa = util.stensor2npa(visuals['E'])
        H_npa = util.stensor2npa(visuals['H'])
        L_npa = util.stensor2npa(visuals['L'])
        

        image_name_ext = os.path.basename(test_data['H_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)

        if img_name!=current_img:
            current_img = img_name
            slice_num = 1
        else:
            slice_num = slice_num + 1

        if save_num != "all":     
            if idx<int(save_num):

                img_dir = os.path.join(opt['path']['images'], img_name+'_'+str(slice_num))
                util.mkdir(img_dir)
            
                save_img_path = os.path.join(img_dir, 'Recon_{:5d}.png'.format(current_step))
                util.imsave(E_img, save_img_path)

                if opt['test']['save_raw_output']:
                    save_img_path = os.path.join(img_dir, 'Recon_c_{:5d}.npy'.format(current_step))
                    util.npsave(E_npa, save_img_path)
                    
                save_img_path = os.path.join(img_dir, 'ZF_{:5d}.png'.format(current_step))
                util.imsave(L_img, save_img_path)
                save_img_path = os.path.join(img_dir, 'ZF_{:5d}.npy'.format(current_step))
                util.npsave(L_npa, save_img_path)
                save_img_path = os.path.join(img_dir, 'GT_{:5d}.png'.format(current_step))
                util.imsave(H_img, save_img_path)
                save_img_path = os.path.join(img_dir, 'GT_{:5d}.npy'.format(current_step))
                util.npsave(H_npa, save_img_path)
        
        elif save_num == "all":

            img_dir = os.path.join(opt['path']['images'], img_name+'_'+str(slice_num))
            util.mkdir(img_dir)
        
            save_img_path = os.path.join(img_dir, 'Recon_{:5d}.png'.format(current_step))
            util.imsave(E_img, save_img_path)

            if opt['test']['save_raw_output']:
                save_img_path = os.path.join(img_dir, 'Recon_c_{:5d}.npy'.format(current_step))
                util.npsave(E_npa, save_img_path)
                save_img_path = os.path.join(img_dir, 'ZF_{:5d}.npy'.format(current_step))
                util.npsave(L_npa, save_img_path)
                save_img_path = os.path.join(img_dir, 'GT_{:5d}.npy'.format(current_step))
                util.npsave(H_npa, save_img_path)

            save_img_path = os.path.join(img_dir, 'ZF_{:5d}.png'.format(current_step))
            util.imsave(L_img, save_img_path)
            
            save_img_path = os.path.join(img_dir, 'GT_{:5d}.png'.format(current_step))
            util.imsave(H_img, save_img_path)
            
        
        else:
            pass

        zf_psnr = util.get_psnr(L_rss, H_rss, border=1)
        zf_ssim = util.get_ssim(L_rss, H_rss, border=1)

        zf_metrics['zf_psnr'][img_name+'_'+str(slice_num)] = zf_psnr
        zf_metrics['zf_ssim'][img_name+'_'+str(slice_num)] = zf_ssim

        current_psnr = util.get_psnr(E_rss, H_rss, border=1)
        current_ssim = util.get_ssim(E_rss, H_rss, border=1)

        test_metrics['test_psnr'][img_name+'_'+str(slice_num)] = current_psnr
        test_metrics['test_ssim'][img_name+'_'+str(slice_num)] = current_ssim

        avg_psnr = current_psnr + avg_psnr
        avg_ssim = current_ssim + avg_ssim

        if test_num != "all":
            if idx==int(test_num)-1:
                break

    avg_psnr = avg_psnr / len(test_loader)
    avg_ssim = avg_ssim / len(test_loader)

    print('Average PSNR : {:<.2f}dB'.format(avg_psnr))
    print('Average SSIM : {:<.4f}'.format(avg_ssim))
    
    print(os.path.split(opt['path']['log'])[-1])
    with open(str(opt['path']['log'])+'/final_test_metrics'+str(os.path.split(opt['path']['log'])[-1]), 'wb') as f:
        pickle.dump(test_metrics, f)

    with open(str(opt['path']['log'])+'/final_zf_metrics'+str(os.path.split(opt['path']['log'])[-1]), 'wb') as f:
        pickle.dump(zf_metrics, f)



if __name__ == '__main__':

    main(sys.argv[1:])



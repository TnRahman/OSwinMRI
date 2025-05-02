import sys
import os.path
import math
import argparse
import random
import pickle
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist
from utils import utils_early_stopping

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from tensorboardX import SummaryWriter


def main(json_path=''):

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))


    init_iter_G, init_path_G = option.find_last_checkpoint_cascade(opt['path']['pretrained_netG'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint_cascade(opt['path']['pretrained_netG'],  net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']

    if opt['rank'] == 0:
        option.save(opt)

    opt = option.dict_to_nonedict(opt)


    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # tensorbordX log
    logger_tensorboard = SummaryWriter(os.path.join(opt['path']['log']))

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
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
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            logger.info('Number of test images: {:,d}'.format(len(test_set)))
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    # data_sample = train_set.__getitem__(1)

    model = define_Model(opt)
    print('Model Defined!')

    model.init_train()
    if opt['rank'] == 0:
        for netwk in model.info_network():  # Causes problems in U-Net
            logger.info(netwk)
        logger.info(model.info_params())

    early_stopping = utils_early_stopping.EarlyStopping(patience=opt['train']['early_stopping_num'])

    test_metrics = {}
    test_metrics['test_psnr'] = {}
    test_metrics['test_ssim'] = {}

    zf_metrics = {}
    zf_metrics['zf_psnr'] = {}
    zf_metrics['zf_ssim'] = {}

    save_gt_zf = True

    for epoch in range(100000000):  # keep running

        for i, train_data in enumerate(train_loader):

            current_step += 1
            model.feed_data(train_data)
            model.optimize_parameters(current_step)
            model.update_learning_rate(current_step, epoch)

            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                          model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)

                logger.info(message)

                # record train loss
                logger_tensorboard.add_scalar('Learning Rate', model.current_learning_rate(), global_step=current_step)
                logger_tensorboard.add_scalar('TRAIN Generator LOSS/G_loss', logs['G_loss'], global_step=current_step)

                if 'G_loss_image' in logs.keys():
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/G_loss_image', logs['G_loss_image'],
                                                  global_step=current_step)
                if 'G_loss_frequency' in logs.keys():
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/G_loss_frequency', logs['G_loss_frequency'],
                                                  global_step=current_step)
                if 'G_loss_preceptual' in logs.keys():
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/G_loss_preceptual', logs['G_loss_preceptual'],
                                                  global_step=current_step)


            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            test_num = opt['test']['test_count']
            save_num = opt['test']['save_count']

            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                avg_psnr = 0.0
                avg_ssim = 0.0

                test_metrics['test_psnr'][str(current_step)] = {}
                test_metrics['test_ssim'][str(current_step)] = {}

                if save_gt_zf:
                    zf_metrics['zf_psnr'][str(current_step)] = {}
                    zf_metrics['zf_ssim'][str(current_step)] = {}

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

                    if save_num != "all":
                        save_idxs = [1,10,100,101,102,103,104,105,106,107]     
                        if idx in save_idxs:

                            img_dir = os.path.join(opt['path']['images'], img_name+'_'+str(idx))
                            util.mkdir(img_dir)
                        
                            save_img_path = os.path.join(img_dir, 'Recon_{:5d}.png'.format(current_step))
                            util.imsave(E_img, save_img_path)

                            if opt['test']['save_raw_output']:
                                save_img_path = os.path.join(img_dir, 'Recon_c_{:5d}.npy'.format(current_step))
                                util.npsave(E_npa, save_img_path)

                            if save_gt_zf:
                                
                                save_img_path = os.path.join(img_dir, 'ZF_{:5d}.png'.format(current_step))
                                util.imsave(L_img, save_img_path)
                                save_img_path = os.path.join(img_dir, 'ZF_{:5d}.npy'.format(current_step))
                                util.npsave(L_npa, save_img_path)
                                save_img_path = os.path.join(img_dir, 'GT_{:5d}.png'.format(current_step))
                                util.imsave(H_img, save_img_path)
                                save_img_path = os.path.join(img_dir, 'GT_{:5d}.npy'.format(current_step))
                                util.npsave(H_npa, save_img_path)
                    
                    elif save_num == "all":


                        img_dir = os.path.join(opt['path']['images'], img_name+'_'+str(idx))
                        util.mkdir(img_dir)
                    
                        save_img_path = os.path.join(img_dir, 'Recon_{:5d}.png'.format(current_step))
                        util.imsave(E_img, save_img_path)

                        if opt['test']['save_raw_output']:
                            save_img_path = os.path.join(img_dir, 'Recon_c_{:5d}.npy'.format(current_step))
                            util.npsave(E_npa, save_img_path)

                        if save_gt_zf:
                            save_img_path = os.path.join(img_dir, 'ZF_{:5d}.png'.format(current_step))
                            util.imsave(L_img, save_img_path)
                            save_img_path = os.path.join(img_dir, 'ZF_{:5d}.npy'.format(current_step))
                            util.npsave(L_npa, save_img_path)
                            save_img_path = os.path.join(img_dir, 'GT_{:5d}.png'.format(current_step))
                            util.imsave(H_img, save_img_path)
                            save_img_path = os.path.join(img_dir, 'GT_{:5d}.npy'.format(current_step))
                            util.npsave(H_npa, save_img_path)
                    
                    else:
                        pass

                    if save_gt_zf:
                        zf_psnr = util.get_psnr(L_rss, H_rss, border=border)
                        zf_ssim = util.get_ssim(L_rss, H_rss, border=border)

                        zf_metrics['zf_psnr'][str(current_step)][img_name+'_'+str(idx)] = zf_psnr
                        zf_metrics['zf_ssim'][str(current_step)][img_name+'_'+str(idx)] = zf_ssim

                    current_psnr = util.get_psnr(E_rss, H_rss, border=border)
                    current_ssim = util.get_ssim(E_rss, H_rss, border=border)

                    test_metrics['test_psnr'][str(current_step)][img_name+'_'+str(idx)] = current_psnr
                    test_metrics['test_ssim'][str(current_step)][img_name+'_'+str(idx)] = current_ssim

                    avg_psnr = current_psnr + avg_psnr
                    avg_ssim = current_ssim + avg_ssim

                    if test_num != "all":
                        if idx==int(test_num)-1:
                            break


                if test_num=='all':
                    avg_psnr = avg_psnr / len(test_loader)
                    avg_ssim = avg_ssim / len(test_loader)
                else:
                    avg_psnr = avg_psnr / (idx+1)
                    avg_ssim = avg_ssim / (idx+1)

                print(os.path.split(opt['path']['log'])[-1])
                with open(str(opt['path']['log'])+'/test_metrics'+str(os.path.split(opt['path']['log'])[-1]), 'wb') as f:
                    pickle.dump(test_metrics, f)

                if current_step == opt['train']['checkpoint_test']:
                    with open(str(opt['path']['log'])+'/zf_metrics'+str(os.path.split(opt['path']['log'])[-1]), 'wb') as f:
                        pickle.dump(zf_metrics, f)

                # testing log
                logger.info(
                    'Performed evaluation on {:6d} items'.format(idx+1))
                logger.info(
                    '<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB'.format(epoch, current_step, avg_psnr))
                logger.info(
                    '<epoch:{:3d}, iter:{:8,d}, Average SSIM : {:<.4f}'.format(epoch, current_step, avg_ssim))

                logger_tensorboard.add_scalar('VALIDATION PSNR', avg_psnr, global_step=current_step)
                logger_tensorboard.add_scalar('VALIDATION SSIM', avg_ssim, global_step=current_step)

                # early stopping
                if opt['train']['is_early_stopping']:
                    early_stopping(avg_psnr, model, epoch, current_step)
                    if early_stopping.is_save:
                        logger.info('Saving the model by early stopping')
                        model.save(f'best_{current_step}')
                    if early_stopping.early_stop:
                        sys.exit('Early stopping!')
                        break

                save_gt_zf = False
    
    print("Training Stop")

if __name__ == '__main__':

    main(sys.argv[1:])


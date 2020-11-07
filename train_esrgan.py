import math
import time
import yaml
import argparse
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import utils  # my tool box
import dataset
from collections import OrderedDict
from network import RRDBNet, VGGStyleDiscriminator128
from torch.utils.tensorboard import SummaryWriter
from pathlib import PurePath, Path
from cv2 import cv2


def receive_arg():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--opt_path', type=str, default='option.yml', 
        help='Path to option YAML file.'
        )
    parser.add_argument(
        '--local_rank', type=int, default=0, 
        help='Distributed launcher requires.'
        )
    args = parser.parse_args()
    
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path
    opts_dict['train']['rank'] = args.local_rank

    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = PurePath('exp') / opts_dict['train']['exp_name'] / "log.log"
    opts_dict['train']['checkpoint_save_path_pre'] = PurePath('exp') / opts_dict['train']['exp_name'] / 'ckp_'
    opts_dict['train']['img_save_folder'] = Path('exp') / opts_dict['train']['exp_name'] / 'img'
    
    opts_dict['train']['num_gpu'] = torch.cuda.device_count()
    if opts_dict['train']['num_gpu'] > 1:
        opts_dict['train']['is_dist'] = True
    else:
        opts_dict['train']['is_dist'] = False
    
    return opts_dict


def main():
    opts_dict = receive_arg()

    # init distributed training
    rank = opts_dict['train']['rank']
    if opts_dict['train']['is_dist']:
        utils.init_dist(local_rank=rank, backend='nccl')

    # create logger
    if rank == 0:
        exp_name = opts_dict['train']['exp_name']
        log_dir = Path("exp") / exp_name
        if log_dir.exists():  # if exists, rename the existing folder
            log_dir_rename = Path("exp") / exp_name
            while log_dir_rename.exists():
                log_dir_rename = Path(str(log_dir_rename) + '_archived')
            log_dir.rename(log_dir_rename) 
        log_dir.mkdir(parents=True)  # make log dir
        
        if not opts_dict['train']['img_save_folder'].exists():
            opts_dict['train']['img_save_folder'].mkdir(parents=True)
            opts_dict['train']['img_save_folder'] = PurePath(opts_dict['train']['img_save_folder'])
        
        writer = SummaryWriter(log_dir)  # tensorboard
        msg = (
            f"{'<' * 10} Hello {'>' * 10}\n"
            f"Timestamp: [{utils.get_timestr()}]\n"
            f"\n{'<' * 10} Options {'>' * 10}\n"
            f"{utils.dict2str(opts_dict)}"
            )
        print(msg)
        log_fp = open(opts_dict['train']['log_path'], 'w')
        log_fp.write(msg + '\n')  # log all parameters
        log_fp.flush()

    # fix random seed
    seed = opts_dict['train']['random_seed']
    # if not set, seeds for numpy.random in each process are the same
    utils.set_random_seed(seed + rank)

    # speed up
    #torch.backends.cudnn.benchmark = False  # if reproduce
    #torch.backends.cudnn.deterministic = True  # if reproduce
    torch.backends.cudnn.benchmark = True  # if speed up

    # create datasets
    train_ds_type = opts_dict['dataset']['train']['type']
    val_ds_type = opts_dict['dataset']['val']['type']
    assert train_ds_type in dataset.__all__, "Not implemented!"
    assert val_ds_type in dataset.__all__, "Not implemented!"
    train_ds_cls = getattr(dataset, train_ds_type)
    val_ds_cls = getattr(dataset, val_ds_type)
    train_ds = train_ds_cls(opts_dict=opts_dict['dataset']['train'])
    val_ds = val_ds_cls(opts_dict=opts_dict['dataset']['val'])

    # create datasamplers
    train_sampler = utils.DistSampler(
        dataset=train_ds, 
        num_replicas=opts_dict['train']['num_gpu'], 
        rank=rank, 
        ratio=opts_dict['dataset']['train']['enlarge_ratio']
        )
    val_sampler = None  # no need to sample val data

    # create dataloaders
    train_loader = utils.create_dataloader(
        dataset=train_ds, 
        opts_dict=opts_dict, 
        sampler=train_sampler, 
        phase='train',
        seed=opts_dict['train']['random_seed']
        )
    val_loader = utils.create_dataloader(
        dataset=val_ds, 
        opts_dict=opts_dict, 
        sampler=val_sampler, 
        phase='val'
        )
    assert train_loader is not None
   
    # create dataloader prefetchers
    tra_prefetcher = utils.CPUPrefetcher(train_loader)
    val_prefetcher = utils.CPUPrefetcher(val_loader)

    # create model
    model_g = RRDBNet(opts_dict=opts_dict['network']['network_g'])  # generator
    model_d = VGGStyleDiscriminator128(opts_dict=opts_dict['network']['network_d'])  # discriminator
    model_g = model_g.to(rank)
    model_d = model_d.to(rank)
    if opts_dict['train']['is_dist']:
        model_g = DDP(model_g, device_ids=[rank])
        model_d = DDP(model_d, device_ids=[rank])
        
    # load pre-trained generator
    ckp_path = opts_dict['network']['network_g']['load_path']
    checkpoint = torch.load(ckp_path)
    state_dict = checkpoint['state_dict']
    if ('module.' in list(state_dict.keys())[0]) and (not opts_dict['train']['is_dist']):  # multi-gpu pre-trained -> single-gpu training
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model_g.load_state_dict(new_state_dict)
    elif ('module.' not in list(state_dict.keys())[0]) and (opts_dict['train']['is_dist']):  # single-gpu pre-trained -> multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k  # add module
            new_state_dict[name] = v
        model_g.load_state_dict(new_state_dict)
    else:  # the same way of training
        model_g.load_state_dict(state_dict)

    # define loss func
    loss_dict = opts_dict['train']['loss'].copy()
    assert loss_dict['perceptual']['type'] == 'PerceptualLoss', "Not implemented."
    assert loss_dict['pixel']['type'] == 'CharbonnierLoss', "Not implemented."
    assert loss_dict['gan']['type'] == 'GANLoss', "Not implemented."
    perceptual_loss_func = utils.PerceptualLoss(**loss_dict['perceptual']['setting']).to(rank)
    pixel_loss_func = utils.CharbonnierLoss(**loss_dict['pixel']['setting']).to(rank)
    gan_loss_func = utils.GANLoss(**loss_dict['gan']['setting']).to(rank)

    # define optimizer
    dict_optim = opts_dict['train']['optim'].copy()
    assert dict_optim['optim_g'].pop('type') == 'Adam', "Not implemented."
    assert dict_optim['optim_d'].pop('type') == 'Adam', "Not implemented."
    optim_g = optim.Adam(model_g.parameters(), **dict_optim['optim_g'])
    optim_d = optim.Adam(model_d.parameters(), **dict_optim['optim_d'])

    # define scheduler
    dict_sched = opts_dict['train']['scheduler'].copy()
    if dict_sched['is_on']:
        assert dict_sched.pop('type') == 'CosineAnnealingRestartLR', "Not implemented."
        del dict_sched['is_on']
        scheduler_g = utils.CosineAnnealingRestartLR(optim_g, **dict_sched)
        scheduler_d = utils.CosineAnnealingRestartLR(optim_d, **dict_sched)

    # define criterion
    if rank ==0:
        criterion_type = opts_dict['train']['criterion']['type']
        assert criterion_type in ['PSNR', 'LPIPS'], "Not implemented."
        if criterion_type == 'PSNR':
            criterion = utils.PSNR()
        if criterion_type == 'LPIPS':
            criterion = utils.LPIPS(**opts_dict['train']['criterion']['setting'])
        unit = opts_dict['train']['criterion']['unit']

    #> start of training and evaluation

    # display & log
    batch_size_all_gpu = opts_dict['dataset']['train']['batch_size_per_gpu'] * opts_dict['train']['num_gpu']  # divided by all GPUs
    num_patch = len(train_ds) * opts_dict['dataset']['train']['enlarge_ratio']
    num_iter_per_epoch = math.ceil(num_patch / batch_size_all_gpu)
    num_iter = int(opts_dict['train']['num_iter'])
    num_epoch = math.ceil(num_iter / num_iter_per_epoch)
    start_iter = 0  # actually start from 1
    start_epoch = start_iter // num_iter_per_epoch

    if rank == 0:
        msg = (
            f"\n{'<' * 10} Dataloader {'>' * 10}\n"
            f"total iters: [{num_iter}]\n"
            f"total epochs: [{num_epoch}]\n"
            f"iter per epoch: [{num_iter_per_epoch}]\n"
            f"previous iter: [{start_iter}]\n"
            f"previous epoch: [{start_epoch}]"
            )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    if opts_dict['train']['is_dist']:
        torch.distributed.barrier()  # all processes wait for ending

    if rank == 0:
        msg = f"\n{'<' * 10} Training {'>' * 10}"
        print(msg)
        log_fp.write(msg + '\n')

        # create timer
        total_timer = utils.Timer()  # total tra + val time of each epoch
        period_timer = utils.Timer()  # time of each print period

    num_iter_accum = start_iter
    val_num = len(val_ds)
    interval_val = int(opts_dict['train']['interval_val'])
    interval_print = int(opts_dict['train']['interval_print'])
    model_g.train() # eval -> train
    model_d.train()
    flag_done = False
    num_iter_accum = start_iter
    current_epoch = start_epoch
    while True:
        if flag_done:
            break
        current_epoch += 1

        # shuffle distributed subsamplers before each epoch
        if opts_dict['train']['is_dist']:
            train_sampler.set_epoch(current_epoch)

        # fetch the first batch
        tra_prefetcher.reset()
        train_data = tra_prefetcher.next()

        # show network structure
        if rank == 0:
            if opts_dict['train']['is_dist']:
                writer.add_graph(model_g.module, train_data['lq'].to(rank))
                #writer.add_graph(model_d.module, train_data['lq'].to(rank))
            else:
                writer.add_graph(model_g, train_data['lq'].to(rank))
                #writer.add_graph(model_d, train_data['lq'].to(rank))   

        # train this epoch
        while train_data is not None:

            # validation
            if ((num_iter_accum % interval_val == 0) or (num_iter_accum == num_iter)) and (rank == 0):

                # save model except for the first iter
                if num_iter_accum != 0:
                    checkpoint_save_path = (
                        f"{opts_dict['train']['checkpoint_save_path_pre']}"
                        f"{num_iter_accum}"
                        ".pt"
                        )
                    state = {
                        'num_iter_accum': num_iter_accum, 
                        'state_dict_g': model_g.state_dict(),
                        'state_dict_d': model_d.state_dict(),
                        'optim_g': optim_g.state_dict(),
                        'optim_d': optim_d.state_dict(),
                        }
                    if opts_dict['train']['scheduler']['is_on']:
                        state['scheduler_g'] = scheduler_g.state_dict()
                        state['scheduler_d'] = scheduler_d.state_dict()
                    torch.save(state, checkpoint_save_path)
                
                with torch.no_grad():
                    per_aver = utils.Counter()
                    pbar = tqdm(total=val_num, ncols=opts_dict['train']['pbar_len'])
                
                    model_g.eval()  # train -> eval
                    model_d.eval()

                    # fetch the first batch
                    val_prefetcher.reset()
                    val_data = val_prefetcher.next()
                    
                    while val_data is not None:
                        # get data
                        gt_data = val_data['gt'].to(rank)  # (B [RGB] H W)
                        lq_data = val_data['lq'].to(rank)
                        b, _, _, _  = lq_data.shape
                        assert b == 1, 'Not supported!'
                        name_img = val_data['name_vid'][0]  # here, bs must be 1!
                        if num_iter_accum != 0:
                            enhanced_data = model_g(lq_data)  # (B [RGB] H W)
                            batch_perf = np.mean([criterion(enhanced_data[i], gt_data[i]) for i in range(b)]) # bs must be 1!
                        else:
                            batch_perf = np.mean([criterion(lq_data[i], gt_data[i]) for i in range(b)])

                        # display
                        pbar.set_description(f'{name_img}: [{batch_perf:.3f}] {unit:s}')
                        pbar.update()

                        # log
                        per_aver.accum(volume=batch_perf)

                        # fetch next batch
                        val_data = val_prefetcher.next()
                    
                    #> end of val
                    pbar.close()
                    model_g.train() # eval -> train
                    model_d.train()

                # log
                ave_per = per_aver.get_ave()
                if num_iter_accum != 0:
                    msg = (
                        f"> model saved at {checkpoint_save_path}\n"
                        f"> ave val {criterion_type}: [{ave_per:.3f}] {unit}"
                        )
                else:
                    msg = f"> ori val {criterion_type}: [{ave_per:.3f}] {unit}"
                print(msg)
                log_fp.write(msg + '\n')
                log_fp.flush()
                writer.add_scalar(f'{criterion_type} vs. iter', ave_per, num_iter_accum)
            #> end of validation

            if opts_dict['train']['is_dist']:
                torch.distributed.barrier()  # all processes wait for val ending

            #> start of training this iter (batch)
            
            num_iter_accum += 1
            if num_iter_accum > num_iter:
                flag_done = True
                break

            # turn off grads of discriminator & update generator
            for p in model_d.parameters():
                p.requires_grad = False

            # get data & run
            gt_data = train_data['gt'].to(rank)  # (B [RGB] H W)
            lq_data = train_data['lq'].to(rank)
            b, _, _, _  = lq_data.shape
            enhanced_data = model_g(lq_data)

            # get loss for generator
            loss_percep = loss_dict['perceptual']['weight'] * perceptual_loss_func(enhanced_data, gt_data)
            loss_pixel = loss_dict['pixel']['weight'] * pixel_loss_func(enhanced_data, gt_data)
            # =====
            # relativistic gan loss
            # gan_loss_func(x - y, target)
            # if x is more real than y, then two d out distance should be 1 at the start
            # if x is less real than y, then two d out distance should be 0 at the start
            # =====
            gt_pred = model_d(gt_data).detach()  # no need to update d, so detach
            enh_pred = model_d(enhanced_data)
            loss_gt = gan_loss_func(gt_pred - torch.mean(enh_pred), False)
            loss_enh = gan_loss_func(enh_pred - torch.mean(gt_pred), True)
            loss_gan = loss_dict['gan']['weight'] * (loss_gt + loss_enh) / 2.
            loss_g = loss_percep + loss_pixel + loss_gan
            
            # update generator
            optim_g.zero_grad()
            loss_g.backward()
            optim_g.step()

            # update g learning rate
            if opts_dict['train']['scheduler']['is_on']:
                scheduler_g.step()  # should after optimizer.step()

            # turn on discriminator
            for p in model_d.parameters():
                p.requires_grad = True

            # get loss & update discriminator
            with torch.autograd.set_detect_anomaly(True):
                optim_d.zero_grad()

                enh_pred = model_d(enhanced_data).detach()  # detach, as constant
                gt_pred = model_d(gt_data)  # grads only relate to d
                loss_gt_half = gan_loss_func(gt_pred - torch.mean(enh_pred), True) * 0.5
                loss_gt_half.backward()

                enh_pred = model_d(enhanced_data.detach())
                loss_enh_half = gan_loss_func(enh_pred - torch.mean(gt_pred.detach()), False) * 0.5
                loss_enh_half.backward()

            optim_d.step()

            loss_d = loss_enh_half + loss_gt_half

            # update d learning rate
            if opts_dict['train']['scheduler']['is_on']:
                scheduler_d.step()  # should after optimizer.step()

            # display & log
            if (num_iter_accum % interval_print == 0) and (rank == 0):
                lr = optim_g.param_groups[0]['lr']
                used_time = period_timer.get_interval()
                period_timer.reset()
                eta = used_time / interval_print * (num_iter - num_iter_accum) 
                msg = (
                    f"iter: [{num_iter_accum}]/{num_iter}, "
                    f"epoch: [{current_epoch}]/{num_epoch}, "
                    f"lr: [{lr * 1e4:.3f}]x1e-4, percep loss: [{loss_percep.item():.4f}], "
                    f"pixel loss: [{loss_pixel.item():.4f}], gan loss: [{loss_gan.item():.4f}], "
                    f"dis loss: [{loss_d.item():.4f}], eta: [{eta / 3600.:.1f}] h"
                    )
                print(msg)
                log_fp.write(msg + '\n')

                writer.add_scalar('percep vs. iter', loss_percep.item(), num_iter_accum)
                writer.add_scalar('pixel vs. iter', loss_pixel.item(), num_iter_accum)
                writer.add_scalar('dis vs. iter', loss_d.item(), num_iter_accum)
                writer.add_scalar('gan vs. iter', loss_gan.item(), num_iter_accum)
                
                # show 5 images
                random_index_lst = np.arange(b)
                np.random.shuffle(random_index_lst)
                random_index_lst = random_index_lst[:5]  # select 5 num from np.arange(5)
                ori_img_batch = []
                cmp_img_batch = []
                enh_img_batch = []
                for idx in random_index_lst:
                    cmp_img = lq_data[idx].cpu().numpy()  # ([RGB] H W) float32
                    ori_img = gt_data[idx].cpu().numpy()
                    enh_img = enhanced_data.detach()
                    enh_img = enh_img[idx].cpu().numpy()
                    cmp_img_batch.append(cmp_img)  # Tensorboard accepts (B C H W)
                    ori_img_batch.append(ori_img)
                    enh_img_batch.append(enh_img)
                    #cv2.imwrite(str(opts_dict['train']['img_save_folder'] / ('cmp_' + f'{num_iter_accum}_' + f'{idx}.png')), cmp_img)
                    #cv2.imwrite(str(opts_dict['train']['img_save_folder'] / ('ori_' + f'{num_iter_accum}_' + f'{idx}.png')), ori_img)
                    #cv2.imwrite(str(opts_dict['train']['img_save_folder'] / ('enh_' + f'{num_iter_accum}_' + f'{idx}.png')), enh_img)
                writer.add_images('cmp', np.array(cmp_img_batch), num_iter_accum)
                writer.add_images('raw', np.array(ori_img_batch), num_iter_accum)
                writer.add_images('enhanced', np.array(enh_img_batch), num_iter_accum)

            # fetch next batch
            train_data = tra_prefetcher.next()

            #> end of this iter (batch)
        #> end of this epoch (training dataloader exhausted)
    #> end of all epochs

    # final log & close logger
    if rank == 0:
        total_time = total_timer.get_interval() / 3600
        msg = f'TOTAL TIME: [{total_time:.1f}] h'
        print(msg)
        log_fp.write(msg + '\n')
        
        msg = (
            f"\n{'<' * 10} Goodbye {'>' * 10}\n"
            f"Timestamp: [{utils.get_timestr()}]"
            )
        print(msg)
        log_fp.write(msg + '\n')
        
        log_fp.close()
        writer.close()


if __name__ == '__main__':
    main()

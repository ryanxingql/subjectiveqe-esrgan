import os
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
from network import RRDBNet
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
    
    opts_dict['train']['num_gpu'] = torch.cuda.device_count()
    if opts_dict['train']['num_gpu'] > 1:
        opts_dict['train']['is_dist'] = True
    else:
        opts_dict['train']['is_dist'] = False
    
    return opts_dict


def main():
    opts_dict = receive_arg()
    exp_name = opts_dict['train']['exp_name']
    rank = opts_dict['train']['rank']
    unit = opts_dict['train']['criterion']['unit']
    interval_print = int(opts_dict['train']['interval_print'])
    interval_val = int(opts_dict['train']['interval_val'])
    
    # init distributed training
    if opts_dict['train']['is_dist']:
        utils.init_dist(
            local_rank=rank, 
            backend='nccl'
            )

    # create logger
    if rank == 0:
        log_dir = Path("exp") / exp_name
        if log_dir.exists():
            log_dir_rename = Path("exp") / exp_name
            while log_dir_rename.exists():
                log_dir_rename = Path(str(log_dir_rename) + '_archived')
            log_dir.rename(log_dir_rename) 
        log_dir.mkdir(parents=True)
        log_fp = open(opts_dict['train']['log_path'], 'w')
        writer = SummaryWriter(log_dir)

        # log all parameters
        msg = (
            f"{'<' * 10} Hello {'>' * 10}\n"
            f"Timestamp: [{utils.get_timestr()}]\n"
            f"\n{'<' * 10} Options {'>' * 10}\n"
            f"{utils.dict2str(opts_dict)}"
            )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    # fix random seed
    seed = opts_dict['train']['random_seed']
    # if not set, seeds for numpy.random in each process are the same
    utils.set_random_seed(seed + rank)

    # speed up
    #torch.backends.cudnn.benchmark = False  # if reproduce
    #torch.backends.cudnn.deterministic = True  # if reproduce
    torch.backends.cudnn.benchmark = True  # speed up

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

    batch_size_all_gpu = opts_dict['dataset']['train']['batch_size_per_gpu'] * opts_dict['train']['num_gpu']  # divided by all GPUs
    num_patch = len(train_ds) * opts_dict['dataset']['train']['enlarge_ratio']
    num_iter_per_epoch = math.ceil(num_patch / batch_size_all_gpu)
    num_iter = int(opts_dict['train']['num_iter'])
    num_epoch = math.ceil(num_iter / num_iter_per_epoch)
    
    # create dataloader prefetchers
    tra_prefetcher = utils.CPUPrefetcher(train_loader)
    val_prefetcher = utils.CPUPrefetcher(val_loader)

    # create model
    model = RRDBNet(opts_dict=opts_dict['network'])
    model = model.to(rank)
    if opts_dict['train']['is_dist']:
        model = DDP(model, device_ids=[rank])

    # define loss func
    assert opts_dict['train']['loss'].pop('type') == 'CharbonnierLoss', \
        "Not implemented."
    loss_func = utils.CharbonnierLoss(**opts_dict['train']['loss'])

    # define optimizer
    assert opts_dict['train']['optim'].pop('type') == 'Adam', \
        "Not implemented."
    optimizer = optim.Adam(
        model.parameters(), 
        **opts_dict['train']['optim']
        )

    # define scheduler
    if opts_dict['train']['scheduler']['is_on']:
        assert opts_dict['train']['scheduler'].pop('type') == \
            'CosineAnnealingRestartLR', "Not implemented."
        del opts_dict['train']['scheduler']['is_on']
        scheduler = utils.CosineAnnealingRestartLR(
            optimizer, 
            **opts_dict['train']['scheduler']
            )
        opts_dict['train']['scheduler']['is_on'] = True

    # define criterion
    assert opts_dict['train']['criterion'].pop('type') == \
        'PSNR', "Not implemented."
    criterion = utils.PSNR()

    start_iter = 0  # actually start from 1
    start_epoch = start_iter // num_iter_per_epoch
    val_num = len(val_ds)

    # display and log
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

    # ==========
    # start training + validation (test)
    # ==========

    model.train()
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
                writer.add_graph(model.module, train_data['lq'].to(rank))
            else:
                writer.add_graph(model, train_data['lq'].to(rank))

        # train this epoch
        while train_data is not None:

            if ((num_iter_accum % interval_val == 0) or (num_iter_accum == num_iter)) and (rank == 0):

                if num_iter_accum != 0:
                    # save model
                    checkpoint_save_path = (
                        f"{opts_dict['train']['checkpoint_save_path_pre']}"
                        f"{num_iter_accum}"
                        ".pt"
                        )
                    state = {
                        'num_iter_accum': num_iter_accum, 
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 
                        }
                    if opts_dict['train']['scheduler']['is_on']:
                        state['scheduler'] = scheduler.state_dict()
                    torch.save(state, checkpoint_save_path)
                
                # validation
                with torch.no_grad():
                    per_aver = utils.Counter()
                    pbar = tqdm(total=val_num, ncols=opts_dict['train']['pbar_len'])
                
                    # train -> eval
                    model.eval()

                    # fetch the first batch
                    val_prefetcher.reset()
                    val_data = val_prefetcher.next()
                    
                    while val_data is not None:
                        # get data
                        gt_data = val_data['gt'].to(rank)  # (B [RGB] H W)
                        lq_data = val_data['lq'].to(rank)
                        b, _, _, _  = lq_data.shape
                        assert b == 1, 'Not supported!'
                        name_img = val_data['name_vid'][0]  # bs must be 1!
                        if num_iter_accum != 0:
                            enhanced_data = model(lq_data)  # (B [RGB] H W)
                            batch_perf = np.mean(
                                [criterion(enhanced_data[i], gt_data[i]) for i in range(b)]
                                ) # bs must be 1!
                        else:
                            batch_perf = np.mean(
                                [criterion(lq_data[i], gt_data[i]) for i in range(b)]
                                )

                        # display
                        pbar.set_description("{:s}: [{:.3f}] {:s}".format(name_img, batch_perf, unit))
                        pbar.update()

                        # log
                        per_aver.accum(volume=batch_perf)

                        # fetch next batch
                        val_data = val_prefetcher.next()
                    
                    # end of val
                    pbar.close()

                    # eval -> train
                    model.train()

                # log
                ave_per = per_aver.get_ave()
                if num_iter_accum != 0:
                    msg = (
                        "> model saved at {:s}\n"
                        "> ave val per: [{:.3f}] {:s}"
                        ).format(
                            checkpoint_save_path, ave_per, unit
                            )
                else:
                    msg = "> ori per: [{:.3f}] {:s}".format(ave_per, unit)
                print(msg)
                log_fp.write(msg + '\n')
                log_fp.flush()
                writer.add_scalar('acc vs. iter', ave_per, num_iter_accum)

            if opts_dict['train']['is_dist']:
                torch.distributed.barrier()  # all processes wait for val ending

            # over sign
            num_iter_accum += 1
            if num_iter_accum > num_iter:
                flag_done = True
                break

            # get data
            gt_data = train_data['gt'].to(rank)  # (B [RGB] H W)
            lq_data = train_data['lq'].to(rank)
            b, _, _, _  = lq_data.shape
            enhanced_data = model(lq_data)  # (B [RGB] H W)

            # get loss
            optimizer.zero_grad()  # zero grad
            loss = torch.mean(torch.stack(
                [loss_func(enhanced_data[i], gt_data[i]) for i in range(b)]
                ))  # cal loss
            loss.backward()  # cal grad
            optimizer.step()  # update parameters

            # update learning rate
            if opts_dict['train']['scheduler']['is_on']:
                scheduler.step()  # should after optimizer.step()

            if (num_iter_accum % interval_print == 0) and (rank == 0):
                # display & log
                lr = optimizer.param_groups[0]['lr']
                loss_item = loss.item()
                used_time = period_timer.get_interval()
                period_timer.reset()
                eta = used_time / interval_print * (num_iter - num_iter_accum) 
                msg = (
                    f"iter: [{num_iter_accum}]/{num_iter}, "
                    f"epoch: [{current_epoch}]/{num_epoch}, "
                    "lr: [{:.3f}]x1e-4, loss: [{:.4f}], eta: [{:.1f}] h".format(
                        lr * 1e4, loss_item, eta / 3600.
                        )
                    )
                print(msg)
                log_fp.write(msg + '\n')

                writer.add_scalar('loss vs. iter', loss_item, num_iter_accum)

                random_index = np.random.choice(range(b))
                ori_img = gt_data[random_index].cpu().numpy()  # ([RGB] H W) float32
                enh_img = enhanced_data.detach()
                enh_img = enh_img[random_index].cpu().numpy()
                writer.add_image('raw', ori_img, num_iter_accum)
                writer.add_image('enhanced', enh_img, num_iter_accum)

            # fetch next batch
            train_data = tra_prefetcher.next()

        # end of this epoch (training dataloader exhausted)
    # end of all epochs

    # final log & close logger
    if rank == 0:
        total_time = total_timer.get_interval() / 3600
        msg = "TOTAL TIME: [{:.1f}] h".format(total_time)
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

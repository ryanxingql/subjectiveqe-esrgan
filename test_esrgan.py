import time
import yaml
import argparse
import torch
import numpy as np
from tqdm import tqdm
import utils  # my tool box
import dataset
from collections import OrderedDict
from network import RRDBNet
from pathlib import PurePath, Path
from cv2 import cv2


def receive_arg():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--opt_path', type=str, default='option.yml', 
        help='Path to option YAML file.'
        )
    args = parser.parse_args()
    
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path

    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['val']['log_path'] = PurePath('exp') / opts_dict['train']['exp_name'] / "log_val.log"
    opts_dict['val']['img_save_folder'] = Path('exp') / opts_dict['train']['exp_name'] / 'img_val'
    opts_dict['val']['ckp_load_path'] = PurePath('exp') / opts_dict['train']['exp_name'] / f"ckp_{opts_dict['val']['load_iter']}.pt"
    
    return opts_dict


def ImgTransfer(img, opt_trans=True, opt_uint8=True):
    if opt_trans:
        img = np.transpose(img, (1, 2, 0))  # (H W [RGB])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # (H W [BGR])
    if opt_uint8:
        img = np.clip((img * 255.), 0, 255).astype(np.uint8)
    return img


def main():
    opts_dict = receive_arg()

    # create logger
    if not opts_dict['val']['img_save_folder'].exists():
        opts_dict['val']['img_save_folder'].mkdir(parents=True)
    
    msg = (
        f"{'<' * 10} Hello {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]\n"
        f"\n{'<' * 10} Options {'>' * 10}\n"
        f"{utils.dict2str(opts_dict)}"
        )
    print(msg)
    log_fp = open(opts_dict['val']['log_path'], 'w')
    log_fp.write(msg + '\n')  # log all parameters
    log_fp.flush()

    # speed up
    #torch.backends.cudnn.benchmark = False  # if reproduce
    #torch.backends.cudnn.deterministic = True  # if reproduce
    torch.backends.cudnn.benchmark = True  # if speed up

    # create dataset
    val_ds_type = opts_dict['dataset']['val']['type']
    assert val_ds_type in dataset.__all__, "Not implemented!"
    val_ds_cls = getattr(dataset, val_ds_type)
    val_ds = val_ds_cls(opts_dict=opts_dict['dataset']['val'])

    # create datasampler
    val_sampler = None  # no need to sample val data

    # create dataloader
    val_loader = utils.create_dataloader(
        dataset=val_ds, 
        opts_dict=opts_dict, 
        sampler=val_sampler, 
        phase='val'
        )
    assert val_loader is not None
   
    # create dataloader prefetcher
    val_prefetcher = utils.CPUPrefetcher(val_loader)

    # create model
    model_g = RRDBNet(opts_dict=opts_dict['network']['network_g'])  # generator
    model_g = model_g.cuda()
    model_g.eval()
        
    # load pre-trained generator
    ckp_path = str(opts_dict['val']['ckp_load_path'])
    checkpoint = torch.load(ckp_path)
    state_dict = checkpoint['state_dict_g']
    if 'module.' in list(state_dict.keys())[0]:  # multi-gpu pre-trained
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model_g.load_state_dict(new_state_dict)
    else:  # single-gpu pre-trained
        model_g.load_state_dict(state_dict)

    # define criterion
    criterion_type = opts_dict['val']['criterion']['type']
    assert criterion_type in ['PSNR', 'LPIPS'], "Not implemented."
    if criterion_type == 'PSNR':
        criterion = utils.PSNR()
    if criterion_type == 'LPIPS':
        criterion = utils.LPIPS(**opts_dict['val']['criterion']['setting'])
    unit = opts_dict['val']['criterion']['unit']

    #> start evaluation

    msg = f"\n{'<' * 10} Evaluation {'>' * 10}"
    print(msg)
    log_fp.write(msg + '\n')

    total_timer = utils.Timer()
    per_aver = utils.Counter()
    val_num = len(val_ds)
    pbar = tqdm(total=val_num, ncols=opts_dict['val']['pbar_len'])
    unit = opts_dict['train']['criterion']['unit']

    with torch.no_grad():
        # fetch the first batch
        val_prefetcher.reset()
        val_data = val_prefetcher.next()
        
        while val_data is not None:
            # get data
            gt_data = val_data['gt'].cuda()  # (B [RGB] H W)
            lq_data = val_data['lq'].cuda()
            b, _, _, _  = lq_data.shape
            assert b == 1, 'Not supported!'
            name_img = val_data['name_vid'][0]  # here, bs must be 1!
            enhanced_data = model_g(lq_data)  # (B [RGB] H W)
            batch_perf = np.mean([criterion(enhanced_data[i], gt_data[i]) for i in range(b)])  # bs must be 1!

            enh_img = enhanced_data[0].cpu().numpy()
            enh_img = ImgTransfer(enh_img, opt_trans=True, opt_uint8=True)  # (H W [BGR]) uint8
            cv2.imwrite(str(opts_dict['val']['img_save_folder'] / f'{name_img}.png'), enh_img)

            # display
            pbar.set_description("{:s}: [{:.3f}] {:s}".format(name_img, batch_perf, unit))
            pbar.update()

            # log
            per_aver.accum(volume=batch_perf)

            # fetch next batch
            val_data = val_prefetcher.next()

    pbar.close()

    # log
    ave_per = per_aver.get_ave()
    msg = "> ave per: [{:.3f}] {:s}".format(ave_per, unit)
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()

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


if __name__ == '__main__':
    main()

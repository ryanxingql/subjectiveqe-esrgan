"""Create LMDB for DIV2K training set."""
import os
import argparse
import yaml
from pathlib import Path
from utils import make_lmdb_from_imgs


parser = argparse.ArgumentParser()
parser.add_argument(
    '--opt_path', type=str, default='option.yml', 
    help='Path to option YAML file.'
    )
args = parser.parse_args()


def create_lmdb():
    yml_path = args.opt_path
    with open(yml_path, 'r') as fp:
        fp = yaml.load(fp, Loader=yaml.FullLoader)
        root_dir = Path(fp['dataset']['root'])
        gt_folder = Path(fp['dataset']['train']['gt_folder'])
        lq_folder = Path(fp['dataset']['train']['lq_folder'])
        gt_path = Path(fp['dataset']['train']['gt_path'])
        lq_path = Path(fp['dataset']['train']['lq_path'])
    gt_dir = root_dir / gt_folder
    lq_dir = root_dir / lq_folder
    lmdb_gt_path = root_dir / gt_path
    lmdb_lq_path = root_dir / lq_path

    # scan all images
    gt_img_list = list(gt_dir.glob('*.png'))
    lq_img_list = []
    for vid in gt_img_list:
        lq_img_list.append(lq_dir / (vid.stem+'.png'))

    # generate LMDB for GT and LQ
    for stem, img_list, lmdb_path in zip(['GT', 'LQ'], [gt_img_list, lq_img_list], [lmdb_gt_path, lmdb_lq_path]):
        
        print(f'Scaning {stem} images...')
        msg = f'> {len(img_list)} images found.'
        print(msg)
        key_list = []
        img_path_list = []
        for img_path in img_list:
            key_list.append(f'{img_path.stem}')
            img_path_list.append(img_path)
        
        print("Reading & writing LMDB...")
        make_lmdb_from_imgs(
            img_path_list=img_path_list,
            lmdb_path=lmdb_path,
            keys=key_list,
            batch=1000,
            compress_level=1,
            multiprocessing_read=True,
            map_size=None,
            )
        print(f'> Finish. Stored at {lmdb_path}')

    # sym-link
    data_path = Path('data')
    new_path = data_path / root_dir.name
    if not new_path.exists():
        if not data_path.exists():
            Path.mkdir(data_path)
        os.system(f'ln -s {root_dir} {new_path}')
        print("Sym-linking done.")
    else:
        print(f'{new_path} already exists.')
    

if __name__ == '__main__':
    create_lmdb()

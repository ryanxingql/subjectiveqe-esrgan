import lmdb
from pathlib import Path
from cv2 import cv2
from tqdm import tqdm
from multiprocessing import Pool
from utils.file_io import import_yuv
import numpy as np


def _read_img_worker(path, key, compress_level):
    """Read image worker.

    Args:
        path (str): Image path.
        key (str): Image key.
        compress_level (int): Compress level when encoding images.

    Returns:
        str: Image key.
        byte: Image byte.
        tuple[int]: Image shape.

    不要把该函数放到主函数里，否则无法并行。
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape
    _, img_byte = cv2.imencode(
        '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level]
        )
    return (key, img_byte, (h, w, c))


def make_lmdb_from_imgs(
        lmdb_path,
        img_path_list,
        keys,
        batch=5000,
        compress_level=1,
        multiprocessing_read=False,
        map_size=None
        ):
    """Make lmdb from images.

    Args:
        img_dir (str): Image root dir.
        lmdb_path (str): LMDB save path.
        img_path_list (str): Image subpath under the image_dir.
        keys (str): LMDB keys.
        batch (int): After processing batch images, lmdb commits.
        compress_level (int): Compress level when encoding images. ranges from 
            0 to 9, where 0 means no compression.
        multiprocessing_read (bool): Whether use multiprocessing to read all
            the images to memory. If True, it will read all the images to 
            memory using multiprocessing. Thus, your server needs to have 
            enough memory.
        map_size (int | None): Map size for lmdb env. If None, use the
            estimated size from images. Default: None

    Usage instance: see STDF-PyTorch.

    Contents of lmdb. The file structure is:
    example.lmdb
    ├── data.mdb
    ├── lock.mdb
    └── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files. Refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records:
        1)image name (with extension), 
        2)image shape, 
        3)compression level, 
    separated by a white space.

    E.g., 00001/0001/im1.png (256,448,3) 1
        Image path: 00001/0001/im1.png
        (HWC): (256,448,3)
        Compression level: 1
        Key: 00001/0001/im1
    """
    # check
    assert len(img_path_list) == len(keys), (
        'img_path_list and keys should have the same length, '
        f'but got {len(img_path_list)} and {len(keys)}')
    assert lmdb_path.suffix == '.lmdb', "lmdb_path must end with '.lmdb'."
    assert not lmdb_path.exists(), f'Folder {lmdb_path} already exists. Exit.'

    # display info
    num_img = len(img_path_list)

    # read all the images to memory by multiprocessing
    if multiprocessing_read:
        def _callback(arg):
            """Register imgs and shapes into the dict & update pbar."""
            key, img_byte, img_shape = arg
            dataset[key], shapes[key] = img_byte, img_shape
            pbar.set_description(f'Read {key}')
            pbar.update(1)
        
        dataset = {}  # use dict to keep the order for multiprocessing
        shapes = {}
        pbar = tqdm(total=num_img, ncols=80)
        pool = Pool()  # default: cpu core num
        # read an image, and record its byte and shape into the dict 
        for path, key in zip(img_path_list, keys):
            pool.apply_async(
                _read_img_worker,
                args=(path, key, compress_level),
                callback=_callback
                )
        pool.close()
        pool.join()
        pbar.close()
        
    # estimate map size if map_size is None
    if map_size is None:
        # obtain data size for one image
        img = cv2.imread(str(img_path_list[0]), cv2.IMREAD_UNCHANGED)
        _, img_byte = cv2.imencode(
            '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level]
            )
        data_size_per_img = img_byte.nbytes
        data_size = data_size_per_img * len(img_path_list)
        map_size = data_size * 10  # enlarge the estimation

    # create lmdb environment & write data to lmdb
    env = lmdb.open(str(lmdb_path), map_size=map_size)
    txn = env.begin(write=True)
    txt_file = open(str(lmdb_path / 'meta_info.txt'), 'w')
    pbar = tqdm(total=num_img, ncols=80)
    for idx, (path, key) in enumerate(zip(img_path_list, keys)):
        pbar.set_description(f'Write {key}')
        pbar.update(1)

        # load image bytes
        if multiprocessing_read:
            img_byte = dataset[key]  # read from prepared dict
            h, w, c = shapes[key]
        else:
            _, img_byte, img_shape = _read_img_worker(
                path, key, compress_level
                )  # use _read function
            h, w, c = img_shape

        # write lmdb
        key_byte = key.encode('ascii')
        txn.put(key_byte, img_byte)
        
        # write meta
        txt_file.write(f'{key} ({h},{w},{c}) {compress_level}\n')

        # commit per batch
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
    pbar.close()
    txn.commit()
    env.close()
    txt_file.close()


def _read_y_from_yuv_worker(video_path, yuv_type, h, w, index_frame, key, compress_level):
    """不要把该函数放到主函数里，否则无法并行。"""
    if h == None:
        w, h = [int(k) for k in video_path.stem.split('_')[1].split('x')]  # for VidName_wxh_nfs.yuv videos
    img = import_yuv(
        seq_path=video_path, 
        yuv_type=yuv_type, 
        h=h, 
        w=w, 
        tot_frm=1, 
        start_frm=index_frame, 
        only_y=True
        )
    img = np.squeeze(img)
    c = 1
    _, img_byte = cv2.imencode(
        '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level]
        )
    return (key, img_byte, (h, w, c))


def make_y_lmdb_from_yuv(
        video_path_list, index_frame_list, key_list, lmdb_path, 
        yuv_type='420p', h=None, w=None, 
        batch=7000, compress_level=1, multiprocessing_read=False, map_size=None
        ):

    assert lmdb_path.suffix == '.lmdb', "lmdb_path must end with '.lmdb'."
    assert not lmdb_path.exists(), f'Folder {lmdb_path} already exists.'

    num_img = len(key_list)

    # read all the images to memory by multiprocessing
    assert multiprocessing_read, "Not implemented."

    def _callback(arg):
        """Register imgs and shapes into the dict & update pbar."""
        key, img_byte, img_shape = arg
        dataset[key], shapes[key] = img_byte, img_shape
        pbar.set_description(f'Reading {key}')
        pbar.update(1)
    
    dataset = {}  # use dict to keep the order for multiprocessing
    shapes = {}
    pbar = tqdm(total=num_img, ncols=80)
    pool = Pool()  # default: cpu core num
    
    # read an image, and record its byte and shape into the dict
    for iter_frm in range(num_img):
        pool.apply_async(
            _read_y_from_yuv_worker,
            args=(
                video_path_list[iter_frm], 
                yuv_type, 
                h, 
                w, 
                index_frame_list[iter_frm], 
                key_list[iter_frm], 
                compress_level
                ),
            callback=_callback
            )
    
    pool.close()
    pool.join()
    pbar.close()
    
    # estimate map size if map_size is None
    if map_size is None:
        # find the first biggest frame
        biggest_index = 0
        biggest_size = 0
        for iter_img in range(num_img):
            vid_path = video_path_list[iter_img]
            if w == None:
                w, h = map(int, str(vid_path.stem).split('_')[1].split('x'))  # for videos following VidName_wxh_nfs
            img_size = w * h
            if img_size > biggest_size:
                biggest_size = img_size
                biggest_index = iter_img
        # obtain data size of one image
        _, img_byte, _ = _read_y_from_yuv_worker(
            video_path_list[biggest_index], 
            yuv_type, 
            h, 
            w, 
            index_frame_list[biggest_index], 
            key_list[biggest_index], 
            compress_level
            )
        data_size_per_img = img_byte.nbytes
        data_size = data_size_per_img * num_img
        map_size = data_size * 10  # enlarge the estimation

    # create lmdb environment & write data to lmdb
    env = lmdb.open(str(lmdb_path), map_size=map_size)
    txn = env.begin(write=True)
    txt_file = open(str(lmdb_path / 'meta_info.txt'), 'w')
    pbar = tqdm(total=num_img, ncols=80)
    for idx, key in enumerate(key_list):
        pbar.set_description(f'Writing {key}')
        pbar.update(1)

        # load image bytes
        img_byte = dataset[key]  # read from prepared dict
        h, w, c = shapes[key]

        # write lmdb
        key_byte = key.encode('ascii')
        txn.put(key_byte, img_byte)
        
        # write meta
        txt_file.write(f'{key} ({h},{w},{c}) {compress_level}\n')

        # commit per batch
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
        
    pbar.close()
    txn.commit()
    env.close()
    txt_file.close()


def _read_yuv444p_worker(
        video_path,
        h,
        w,
        index_frame,
        key,
        compress_level
        ):
    """
    读一帧YUV444P，不转换为RGB，直接按PNG格式无损压缩。
    不支持YUV420P，因为其3通道尺寸不同！
    
    不要把该函数放到主函数里，否则无法并行。
    
    update: 20-10-24
    """
    if h == None:
        w, h = [int(k) for k in video_path.stem.split('_')[1].split('x')]  # for VidName_wxh_nfs.yuv videos
    img_y, img_u, img_v = import_yuv(
        seq_path=video_path,
        h=h,
        w=w,
        tot_frm=1,
        yuv_type='444p',
        start_frm=index_frame,
        only_y=False,
        )
    img = np.concatenate((img_y, img_u, img_v), axis=0)  # C H W
    img = np.transpose(img, (1, 2, 0))  # H W C

    _, img_byte = cv2.imencode(
        '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level]
        )
    c = 3
    return (key, img_byte, (h, w, c))


def make_yuv444p_lmdb(
        video_path_list=['demo0.yuv','demo1.yuv',],
        index_frame_list=[3,3,],
        key_list=['vid0_im3','vid1_im3',],
        lmdb_path='demo.lmdb',
        h=None,
        w=None,
        batch=7000,
        compress_level=1, 
        multiprocessing_read=True,
        map_size=None
        ):
    """
    对一系列YUV444P视频，制作一个LMDB。每个视频只记录一帧。

    YUV444P帧不转换为RGB，直接按PNG格式无损压缩并存储。
    LMDB中还包括一个meta_info.txt，记录图像key、尺寸和压缩强度。

    compress_level：从0到9可选整数，最低0表示无压缩。
    map_size：LMDB env的设定大小。若None，会自动根据单幅图像大小乘以图像数量以评估。
    
    update: 20-10-24
    """
    assert lmdb_path.suffix == '.lmdb', "lmdb_path must end with '.lmdb'."
    assert not lmdb_path.exists(), f'Folder {lmdb_path} already exists.'
    assert multiprocessing_read, "Not implemented."

    # read all the images to memory by multiprocessing
    def _callback(arg):
        """Register imgs and shapes into the dict & update pbar."""
        key, img_byte, img_shape = arg
        dataset[key], shapes[key] = img_byte, img_shape
        pbar.set_description(f'Reading {key}')
        pbar.update(1)
    
    dataset = {}  # use dict to keep the order for multiprocessing
    shapes = {}
    pool = Pool()  # default: cpu core num
    
    num_img = len(key_list)
    pbar = tqdm(total=num_img, ncols=80)
    for iter_frm in range(num_img):
        pool.apply_async(
            _read_yuv444p_worker,  # read img
            args=(
                video_path_list[iter_frm], 
                h, 
                w, 
                index_frame_list[iter_frm], 
                key_list[iter_frm], 
                compress_level
                ),
            callback=_callback  # record into a dict
            )
    pool.close()
    pool.join()
    pbar.close()
    
    # estimate map size if map_size is None
    if map_size is None:
        # find the first biggest frame
        biggest_index = 0
        biggest_size = 0
        for iter_img in range(num_img):
            vid_path = video_path_list[iter_img]
            if h == None:
                w, h = [int(k) for k in video_path.stem.split('_')[1].split('x')]  # for VidName_wxh_nfs.yuv videos
            img_size = w * h
            if img_size > biggest_size:
                biggest_size = img_size
                biggest_index = iter_img
        # obtain data size of one image
        _, img_byte, _ = _read_yuv444p_worker(
            video_path_list[biggest_index],
            h, 
            w, 
            index_frame_list[biggest_index], 
            key_list[biggest_index], 
            compress_level
            )
        data_size_per_img = img_byte.nbytes
        data_size = data_size_per_img * num_img
        map_size = data_size * 10  # enlarge the estimation

    # create lmdb environment & write data to lmdb
    env = lmdb.open(str(lmdb_path), map_size=map_size)
    txn = env.begin(write=True)
    txt_file = open(str(lmdb_path / 'meta_info.txt'), 'w')
    pbar = tqdm(total=num_img, ncols=80)
    for idx, key in enumerate(key_list):
        pbar.set_description(f'Writing {key}')
        pbar.update(1)

        # load image bytes
        img_byte = dataset[key]  # read from prepared dict
        h, w, c = shapes[key]

        # write lmdb
        key_byte = key.encode('ascii')
        txn.put(key_byte, img_byte)
        
        # write meta
        txt_file.write(f'{key} ({h},{w},{c}) {compress_level}\n')

        # commit per batch
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
        
    pbar.close()
    txn.commit()
    env.close()
    txt_file.close()
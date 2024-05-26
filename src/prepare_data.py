import os 
import io
import sys
import urllib
import zipfile
import asyncio
from copy import deepcopy
from typing import Sequence
from collections import defaultdict

import httpx
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants as const
from utils import open_json
from utils import save_json


async def download_async(client: httpx.AsyncClient, url: str, output_filepath: str, position: int | None = None) -> None:
    with open(output_filepath, 'wb') as output_file:
        async with client.stream('GET', url) as response:
            total = int(response.headers['Content-Length'])

            with atqdm(total=total, unit_scale=True, unit_divisor=1024, unit='B', desc=output_filepath, position=position) as progress:
                num_bytes_downloaded = response.num_bytes_downloaded
                async for chunk in response.aiter_bytes():
                    output_file.write(chunk)
                    progress.update(response.num_bytes_downloaded - num_bytes_downloaded)
                    num_bytes_downloaded = response.num_bytes_downloaded


# CPPE-5


async def download_cppe5() -> None:
    urls = [
        'https://huggingface.co/datasets/cppe-5/resolve/main/data/test-00000-of-00001.parquet?download=true',
        'https://huggingface.co/datasets/cppe-5/resolve/main/data/train-00000-of-00001.parquet?download=true',
    ]
    fps = [
        const.CPPE_DATA_DIR + os.path.basename(urllib.parse.urlparse(url).path) for url in urls
    ]
    fps = [fp for fp in fps if not os.path.exists(fp)]
    if not fps:
        return
    print('Downloading CPPE-5...')
    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = [asyncio.create_task(download_async(client, url, fp)) for url, fp in zip(urls, fps)]
        await asyncio.gather(*tasks)


def preprocess_cppe_data(df: pd.DataFrame, root: str) -> tuple[list[dict], list[dict]]:
    annotations = []
    images = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['image_id']
        if img_id in (616, 648, 674, 699, 702, 762, 897):
            continue 
        ann = row['objects']
        # gather images
        img = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
        true_width, true_height = img.width, img.height
        images.append(
            {
                'file_name': row['image']['path'],
                'height': true_height,
                'width': true_width,
                'id': row['image_id'],
            }
        )
        # gather annotations
        maybe_width, maybe_height = row['width'], row['height']
        for id_, bbox, category in zip(ann['id'], ann['bbox'], ann['category']):
            if img_id in (83, 97):
                rescale = np.array([true_width/maybe_width, true_height/maybe_height]*2)
                bbox = (bbox*rescale).astype('int')
            area = bbox[2]*bbox[3]
            annotations.append({
                'area': area,
                'iscrowd': 0,
                'image_id': img_id,
                'bbox': bbox,
                'category_id': category+1,
                'id': id_,
            })
        # save image
        fp = root + row['image']['path'] 
        img.save(fp)

    return annotations, images


def prepare_cppe5() -> None:
    print('Preparing CPPE-5.')
    os.makedirs(const.CPPE_DATA_DIR, exist_ok=True)
    os.makedirs(const.CPPE_ANNOTATIONS_DIR, exist_ok=True)
    asyncio.run(download_cppe5())

    # train
    if not os.path.isdir(const.CPPE_TRAIN_IMAGES):
        print('Preparing train annotations...')
        os.makedirs(const.CPPE_TRAIN_IMAGES, exist_ok=True)
        df = pd.read_parquet(const.CPPE_RAW_TRAIN)
        annotations, images = preprocess_cppe_data(df, const.CPPE_TRAIN_IMAGES)
        annotations = {
            'info': '',
            'licenses': '',
            'images': images,
            'annotations': annotations,
            'categories': const.CPPE_CATEGORIES,
        }
        save_json(const.CPPE_TRAIN_ANNOTATIONS, annotations)
    else:
        print('Found CPPE-5 training image directory. Skipping trianing set.')

    # test
    if not os.path.isdir(const.CPPE_VAL_IMAGES):
        print('Preparing validation annotations...')
        os.makedirs(const.CPPE_VAL_IMAGES, exist_ok=True)
        df = pd.read_parquet(const.CPPE_RAW_VAL)
        annotations, images = preprocess_cppe_data(df, const.CPPE_VAL_IMAGES)
        annotations = {
            'info': '',
            'licenses': '',
            'images': images,
            'annotations': annotations,
            'categories': const.CPPE_CATEGORIES,
        }
        save_json(const.CPPE_VAL_ANNOTATIONS, annotations)
    else:
        print('Found CPPE-5 validation image directory. Skipping validation set.')

    print('Done.')


def generate_seeds_cppe(seed_start: int, seed_stop: int, shots: Sequence[int]) -> None:
    print('Generating seeds for CPPE-5.')
    data = open_json(const.CPPE_TRAIN_ANNOTATIONS)

    # list of IDs of all categories 
    category_ids = [category['id'] for category in const.CPPE_CATEGORIES]
    # mapping of all image ID to image info dict
    img_id_to_img = {img['id']: img for img in data['images']}
    # mapping of category ID to category annotations 
    anns_per_category = defaultdict(list)
    for ann in data['annotations']:
        if ann.get('iscrowd') == 1:
            continue
        anns_per_category[ann['category_id']].append(ann)

    for seed in tqdm(range(seed_start, seed_stop)):
        for shot in shots:
            np.random.seed(seed)
            sample_anns = []
            sample_imgs = []
            used_imgs = set()
            for category_id in category_ids:
                category_sample_anns = []
                # mapping of image ID to annotations for a single category
                img_id_to_cat_anns = defaultdict(list)
                for ann in anns_per_category[category_id]:
                    img_id_to_cat_anns[ann['image_id']].append(ann)

                img_ids = list(img_id_to_cat_anns.keys())
                np.random.shuffle(img_ids)
                for img_id in img_ids:
                    anns = img_id_to_cat_anns[img_id]
                    img = img_id_to_img[img_id]

                    if len(category_sample_anns) + len(anns) > shot:
                        continue

                    if img_id not in used_imgs:
                        sample_imgs.append(img)
                        used_imgs |= {img_id}
                    category_sample_anns += anns

                    if len(category_sample_anns) == shot:
                        sample_anns += category_sample_anns
                        break

            assert len(sample_anns) == (len(category_ids) * shot), 'Bad seed.'

            new_data = {
                'info': data['info'],
                'licenses': data['licenses'],
                'categories': data['categories'],
                'images': sample_imgs,
                'annotations': sample_anns,
            }
            fp = const.CPPE_ANNOTATIONS_DIR + f'cppe_shots_{shot}_seed_{seed}.json'
            save_json(fp, new_data)
    print('Done.')


# COCO


async def download_coco() -> None:
    urls = [
        'http://images.cocodataset.org/zips/train2017.zip',
        'http://images.cocodataset.org/zips/val2017.zip',
        'http://images.cocodataset.org/zips/test2017.zip',
        'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    ]
    fps = [
        const.COCO_DATA_DIR + os.path.basename(urllib.parse.urlparse(url).path) for url in urls
    ]
    dirs = [os.path.splitext(fp)[0] for fp in fps]
    to_download = [
        (fp, url) for dir_, fp, url in zip(dirs, fps, urls) 
        if not (os.path.isdir(dir_) or os.path.exists(fp))
        and not ('annotations' in dir_ and os.path.isdir(dir_.rsplit('_', 1)[0])) 
    ]
    if to_download:
        print('Downloading COCO...')
        async with httpx.AsyncClient(follow_redirects=True) as client:
            tasks = [asyncio.create_task(download_async(client, url, fp)) for fp, url in to_download]
            await asyncio.gather(*tasks)
    for fp, dir_ in zip(fps, dirs):
        if os.path.isdir(dir_) or ('annotations' in dir_ and os.path.isdir(dir_.rsplit('_', 1)[0])):
            continue
        print(f'Unzipping {os.path.basename(fp)} to {const.COCO_DATA_DIR}...')
        with zipfile.ZipFile(fp, 'r') as zip_file:
            zip_file.extractall(const.COCO_DATA_DIR)


def prepare_coco() -> None:
    print('Preparing COCO.')
    os.makedirs(const.COCO_DATA_DIR, exist_ok=True)
    asyncio.run(download_coco())
    
    for ds in ('training', 'validation'):
        print(f'Preparing {ds} annotations...')
        data = open_json(
            const.COCO_RAW_TRAIN_ANNOTATIONS if ds == 'training' else const.COCO_RAW_VAL_ANNOTATIONS
        )
        all_images = {img['id']: img for img in data['images']}
        annotations, images = [], []
        for ann in tqdm(data['annotations']):
            # check for zero height/width bboxes
            bbox = ann['bbox']
            if bbox[2] < 1e-3 or bbox[3] < 1e-3:
                continue
            annotations.append(ann)
            images.append(all_images[ann['image_id']])

        print(f'Saving {ds} annotations...')
        data['annotations'] = annotations
        data['images'] = images
        fp = const.COCO_TRAIN_ANNOTATIONS if ds == 'training' else const.COCO_VAL_ANNOTATIONS
        save_json(fp, data)

    print('Done.')


def generate_seeds_coco(seed_start: int, seed_stop: int, shots: Sequence[int]) -> None:
    print('Generating seeds for COCO.')
    data = open_json(const.COCO_TRAIN_ANNOTATIONS)

    coco_categories = data['categories']
    # list of IDs of all categories 
    category_ids = [category['id'] for category in coco_categories]
    # mapping of all image ID to image info dict
    img_id_to_img = {img['id']: img for img in data['images']}
    # mapping of category ID to category annotations 
    anns_per_category = defaultdict(list)
    for ann in data['annotations']:
        if ann.get('iscrowd') == 1:
            continue
        anns_per_category[ann['category_id']].append(ann)

    for seed in tqdm(range(seed_start, seed_stop)):
        for shot in shots:
            np.random.seed(seed)
            sample_anns = []
            sample_imgs = []
            used_imgs = set()
            for category_id in category_ids:
                category_sample_anns = []
                # mapping of image ID to annotations for a single category
                img_id_to_cat_anns = defaultdict(list)
                for ann in anns_per_category[category_id]:
                    img_id_to_cat_anns[ann['image_id']].append(ann)

                img_ids = list(img_id_to_cat_anns.keys())
                np.random.shuffle(img_ids)
                for img_id in img_ids:
                    anns = img_id_to_cat_anns[img_id]
                    img = img_id_to_img[img_id]

                    if len(category_sample_anns ) + len(anns) > shot:
                        continue

                    if img_id not in used_imgs:
                        sample_imgs.append(img)
                        used_imgs |= {img_id}
                    category_sample_anns += anns

                    if len(category_sample_anns) == shot:
                        sample_anns += category_sample_anns
                        break

            assert len(sample_anns) == len(category_ids) * shot, 'Bad seed.'

            new_data = {
                'info': data['info'],
                'licenses': data['licenses'],
                'categories': data['categories'],
                'images': sample_imgs,
                'annotations': sample_anns,
            }
            fp = const.COCO_ANNOTATIONS_DIR + f'coco_shots_{shot}_seed_{seed}.json'
            save_json(fp, new_data)
    print('Done.')


# FSL


def prepare_train_fsl(seed_start: int, seed_stop: int, shots: Sequence[int]) -> None:
    print('Preparing training FSL.')
    os.makedirs(const.FSL_DATA_DIR, exist_ok=True)
    os.makedirs(const.FSL_ANNOTATIONS_DIR, exist_ok=True)

    cppe_img_dir = os.path.relpath(const.CPPE_TRAIN_IMAGES, const.DATA_DIR).rstrip('/') + '/'
    coco_img_dir = os.path.relpath(const.COCO_TRAIN_IMAGES, const.DATA_DIR).rstrip('/') + '/'

    # increase CPPE-5 categories IDs to follow after COCOs
    cppe_categories = deepcopy(const.CPPE_CATEGORIES)
    for category in cppe_categories:
        category['id'] += const.COCO_NUM_CATEGORIES 

    for seed in tqdm(range(seed_start, seed_stop)):
        for shot in shots:
            # load CPPE-5 samples
            fp_cppe = const.CPPE_ANNOTATIONS_DIR + f'cppe_shots_{shot}_seed_{seed}.json'
            data_cppe = open_json(fp_cppe)
            # increase CPPE-5 image IDs to follow after COCOs and update filepaths
            for img in data_cppe['images']:
                ext = os.path.splitext(img['file_name'])[1]
                img['file_name'] = cppe_img_dir + str(img['id']) + ext
                img['id'] += const.CPPE_IMG_ID_ADDEND
            # update annotations id, category_id and image_id
            for ann in data_cppe['annotations']:
                ann['id'] += const.CPPE_ANN_ID_ADDEND
                ann['image_id'] += const.CPPE_IMG_ID_ADDEND
                ann['category_id'] += const.COCO_NUM_CATEGORIES

            # load COCO samples
            fp_coco = const.COCO_ANNOTATIONS_DIR + f'coco_shots_{shot}_seed_{seed}.json'
            data_coco = open_json(fp_coco)
            # update image filepaths
            for img in data_coco['images']:
                img['file_name'] = coco_img_dir + img['file_name']

            new_data = {
                'info': '',
                'licenses': '',
                'categories': cppe_categories + data_coco['categories'],
                'images': data_cppe['images'] + data_coco['images'],
                'annotations': data_cppe['annotations'] + data_coco['annotations'],
            }
            fp = const.FSL_ANNOTATIONS_DIR + f'fsl_shots_{shot}_seed_{seed}.json'
            save_json(fp, new_data)
    print('Done.')


def prepare_val_fsl() -> None:
    print('Preparing validation FSL.')
    cppe_img_dir = os.path.relpath(const.CPPE_VAL_IMAGES, const.DATA_DIR).rstrip('/') + '/'
    coco_img_dir = os.path.relpath(const.COCO_VAL_IMAGES, const.DATA_DIR).rstrip('/') + '/'

    # increase CPPE-5 categories IDs to follow after COCOs
    cppe_categories = deepcopy(const.CPPE_CATEGORIES)
    for category in cppe_categories:
        category['id'] += const.COCO_NUM_CATEGORIES 

    # load CPPE-5 samples
    data_cppe = open_json(const.CPPE_VAL_ANNOTATIONS)
    # increase CPPE-5 image IDs to follow after COCOs and update filepaths
    for img in data_cppe['images']:
        ext = os.path.splitext(img['file_name'])[1]
        img['file_name'] = cppe_img_dir + str(img['id']) + ext
        img['id'] += const.CPPE_IMG_ID_ADDEND
    # update annotations id, category_id and image_id
    for ann in data_cppe['annotations']:
        ann['id'] += const.CPPE_ANN_ID_ADDEND
        ann['image_id'] += const.CPPE_IMG_ID_ADDEND
        ann['category_id'] += const.COCO_NUM_CATEGORIES
    # save CPPE-5 only annotations
    new_data_cppe = {
        'info': '',
        'licenses': '',
        'categories': cppe_categories,
        'images': data_cppe['images'],
        'annotations': data_cppe['annotations'],
    }
    save_json(const.FSL_VAL_ANNOTATIONS_CPPE, new_data_cppe)

    # load COCO samples
    data_coco = open_json(const.COCO_VAL_ANNOTATIONS)
    # update image filepaths
    for img in data_coco['images']:
        img['file_name'] = coco_img_dir + img['file_name']
    # save COCO only annotations
    new_data_coco = {
        'info': '',
        'licenses': '',
        'categories': data_coco['categories'],
        'images': data_coco['images'],
        'annotations': data_coco['annotations'],
    }
    save_json(const.FSL_VAL_ANNOTATIONS_COCO, new_data_coco)

    new_data = {
        'info': '',
        'licenses': '',
        'categories': cppe_categories + data_coco['categories'],
        'images': data_cppe['images'] + data_coco['images'],
        'annotations': data_cppe['annotations'] + data_coco['annotations'],
    }
    save_json(const.FSL_VAL_ANNOTATIONS, new_data)
    print('Done.')


if __name__ == '__main__':
    seed_start = const.SEED_START
    seed_stop = const.SEED_STOP
    shots = const.SHOTS

    prepare_cppe5()
    generate_seeds_cppe(seed_start, seed_stop, shots)
    prepare_coco()
    generate_seeds_coco(seed_start, seed_stop, shots)
    prepare_train_fsl(seed_start, seed_stop, shots)
    prepare_val_fsl()


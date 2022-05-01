import os
import h5py
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wget
from tqdm import tqdm
import params 
import subprocess
from zipfile import ZipFile
from PIL import Image
from argparse import ArgumentParser
import fiftyone.zoo as foz
from utils.io import (list_dir, mkdir_if_not_exists, extract_zip, extract_targz,
    download_from_url, load_json_object)

from grit_paths import GritPaths

log = logging.getLogger('__main__')

def download_coco(args):
    base = params.GRIT['images']
    img_dir = f'{args.data_dir}/{base}/coco'
    mkdir_if_not_exists(img_dir,recursive=True)
    download_from_url(params.COCO_TRAIN_2014,img_dir)
    extract_zip(f'{img_dir}/test2015.zip',img_dir)
    download_from_url(params.COCO_TEST_2015,img_dir)
    extract_zip(f'{img_dir}/train2014.zip',img_dir)
    log.info('COCO downloaded')
# def download_coco(cfg):
#     img_dir = f'{cfg.grit.images}/coco'
#     mkdir_if_not_exists(img_dir,recursive=True)

#     # download test2015
#     download_from_url(cfg.urls.images.coco.test2015,img_dir)
#     extract_zip(f'{img_dir}/test2015.zip', img_dir)

#     # download train2014 for refcoco
#     download_from_url(cfg.urls.images.coco.train2014,img_dir)
#     extract_zip(f'{img_dir}/train2014.zip', img_dir)

def download_refclef(args):
    base = params.GRIT['images']
    img_dir = f'{args.data_dir}/{base}'
    mkdir_if_not_exists(img_dir,recursive=True)
    download_from_url(params.REFCLEF,img_dir)
    extract_zip(f'{img_dir}/saiapr_tc-12.zip',img_dir)
    log.info('Refclef downloaded')
# def download_refclef(cfg):
#     img_dir = f'{cfg.grit.images}/refclef'
#     mkdir_if_not_exists(img_dir,recursive=True)

#     download_from_url(cfg.urls.images.refclef, img_dir)
#     extract_zip(f'{img_dir}/saiapr_tc-12.zip', img_dir)

def download_construction(args):
    base = params.GRIT['images']
    img_dir = f'{args.data_dir}/{base}'
    mkdir_if_not_exists(img_dir,recursive=True)
    download_from_url(params.CONSTRUCTION,img_dir)
    extract_targz(
        f'{img_dir}/construction_images.tar.gz',img_dir)
    log.info('Construction downloaded')
    
# def download_construction(cfg):
#     mkdir_if_not_exists(cfg.grit.images,recursive=True)
#     download_from_url(cfg.urls.images.construction, cfg.grit.images)
#     extract_targz(
#         f'{cfg.grit.images}/construction_images.tar.gz',
#         cfg.grit.images)
def download_open_images(args):
    base = params.GRIT['images']
    img_dir = f'{args.data_dir}/{base}/open_images'
    mkdir_if_not_exists(img_dir,recursive=True)
    image_ids = set()
    for task in ['localization','categorization','segmentation']:
        grit_paths = GritPaths(params.GRIT['base'])
        for subset in ['ablation','test']:
            samples = load_json_object(grit_paths.samples(task,subset))
            image_ids.update(
                [os.path.splitext(os.path.basename(s['image_id']))[0] \
                    for s in samples if 'open_images' in s['image_id']]
            )
    foz.download_zoo_dataset(
        "open-images-v6",
        dataset_dir=img_dir,
        split='test',
        image_ids = image_ids 
    )
    src_dir = f'{img_dir}/test/data'
    tgt_dir = f'{img_dir}/test'
    subprocess.call(
        f'mv {src_dir}/* {tgt_dir}',
        shell=True)
    
    subprocess.call(
        f'rm -rf {img_dir}/test/data & rm -rf {img_dir}/test/metadata & rm -rf {img_dir}/test/labels',
        shell=True)
    log.info('Downloaded open images')

# def download_open_images(cfg):
#     img_dir = f'{cfg.grit.images}/open_images'
#     mkdir_if_not_exists(img_dir,recursive=True)
#     image_ids = set()
#     for task in ['localization','categorization','segmentation']:
#         grit_paths = GritPaths(cfg.grit.base)
#         for subset in ['ablation','test']:
#             samples = load_json_object(grit_paths.samples(task, subset))
#             image_ids.update(
#                 [os.path.splitext(os.path.basename(s['image_id']))[0] \
#                     for s in samples if 'open_images' in s['image_id']])
    
#     foz.download_zoo_dataset(
#         "open-images-v6",
#         dataset_dir=img_dir,
#         split='test',
#         image_ids=image_ids)

#     src_dir = f'{img_dir}/test/data'
#     tgt_dir = f'{img_dir}/test'
#     subprocess.call(
#         f'mv {src_dir}/* {tgt_dir}',
#         shell=True)
    
#     subprocess.call(
#         f'rm -rf {img_dir}/test/data & rm -rf {img_dir}/test/metadata & rm -rf {img_dir}/test/labels',
#         shell=True)

def download_visual_genome(args):
    base = params.GRIT['images']
    img_dir = f'{args.data_dir}/{base}/visual_genome'
    mkdir_if_not_exists(img_dir,recusrive=True)
    image_ids = set()
    grit_base = params.GRIT['base']
    grit_paths = GritPaths(f'{args.data_dir}/{grit_base}')
    for subset in ['ablation','test']:
        samples = load_json_object(grit_paths.samples('vqa',subset))
        image_ids.update([
            '/'.join(s['image_id'].split('/')[1:]) for s in samples if 'visual_genome' in s['image_id']])
    for image_id in image_ids:
        subdir = os.path.join(img_dir,image_id.split('/')[0])
        mkdir_if_not_exists(subdir,recursive=True)
        download_from_url(f'https://cs.stanford.edu/people/rak248/{image_id}',
            subdir)
    log.info('Visual genome downloaded')
      
# def download_visual_genome(cfg):
#     img_dir = f'{cfg.grit.images}/visual_genome'
#     mkdir_if_not_exists(img_dir,recursive=True)
#     image_ids = set()
#     grit_paths = GritPaths(cfg.grit.base)
#     for subset in ['ablation','test']:
#         samples = load_json_object(grit_paths.samples('vqa', subset))
#         image_ids.update([
#             '/'.join(s['image_id'].split('/')[1:]) for s in samples if 'visual_genome' in s['image_id']])
    
#     for image_id in image_ids:
#         subdir = os.path.join(img_dir,image_id.split('/')[0])
#         mkdir_if_not_exists(subdir,recursive=True)
#         download_from_url(
#             f'https://cs.stanford.edu/people/rak248/{image_id}',
#             subdir)

def save_nyuv2(args):
    base = params.GRIT['images']
    img_dir = f'{args.data_dir}/{base}/nyuv2'
    mkdir_if_not_exists(img_dir,recursive=True)
    download_from_url(params.NYUV2,img_dir)
    f = h5py.File(os.path.join(img_dir,'nyu_depth_v2_labeled.mat'),'r')
    images = f['images'][()].transpose(0,3,2,1)
    num_images = images.shape[0]
    for i in tqdm(range(num_images)):
        im = Image.fromarray(images[i])
        im.save(f'{img_dir}/{i+1}.jpg')
    log.info('NYU downloaded')
# def save_nyuv2(cfg):
#     img_dir = f'{cfg.grit.images}/nyuv2'
#     mkdir_if_not_exists(img_dir,recursive=True)

#     download_from_url(cfg.urls.images.nyuv2, img_dir)
#     f = h5py.File(os.path.join(img_dir,'nyu_depth_v2_labeled.mat'),'r')
#     images = f['images'][()].transpose(0,3,2,1)
#     num_images = images.shape[0]
#     for i in tqdm(range(num_images)):
#         im = Image.fromarray(images[i])
#         im.save(f'{img_dir}/{i+1}.jpg')

def download_blended_mvs(args):
    base = params.GRIT['images']
    img_dir = f'{args.data_dir}/{base}'
    mkdir_if_not_exists(img_dir,recursive=True)
    download_from_url(params.BLENDED_MVS,img_dir)
    extract_targz(
        os.path.join(img_dir,'blended_mvs_images.tar.gz'),
        img_dir)
    log.info('MVS downloaded')
# def download_blended_mvs(cfg):
#     mkdir_if_not_exists(cfg.grit.images,recursive=True)

#     download_from_url(cfg.urls.images.blended_mvs,cfg.grit.images)
#     extract_targz(
#         os.path.join(cfg.grit.images,'blended_mvs_images.tar.gz'),
#         cfg.grit.images)

def download_dtu(args):
    base = params.GRIT['images']
    img_dir = f'{args.data_dir}/{base}'
    mkdir_if_not_exists(img_dir,recursive=True)
    download_from_url(params.DTU,img_dir)
    extract_targz(
        os.path.join(img_dir,'dtu_images.tar.gz'),
        img_dir)
    log.info('DTU downloaded')
# def download_dtu(cfg):
#     mkdir_if_not_exists(cfg.grit.images,recursive=True)

#     download_from_url(cfg.urls.images.dtu,cfg.grit.images)
#     extract_targz(
#         os.path.join(cfg.grit.images,'dtu_images.tar.gz'),
#         cfg.grit.images)
    

def download_scannet(cfg):
    print("ATTENTION - scannet has not been downloaded")
    print("You must sign a Terms of Service agreement before downloading scannet manually")
    print("Instructions can be found at https://github.com/allenai/grit_official/blob/main/download/scannet_download_instructions.md")



def main():
    parser = ArgumentParser()
    parser.add_argument("--datasets",nargs='+',default=None)
    parser.add_argument("--subsets_to_distort",'+',default=None)
    parser.add_argument("--tasks_to_distort",nargs='+',default=None)
    parser.add_argument("--output_dir",default=f'{params.PREFIX}/grit_official/outputs/')
    parser.add_argument("--data_dir",default='/data/michal5')
    args = parser.parse_args()
    base = params.GRIT['images']
    img_dir = f'{args.data_dir}/{base}/'
    for dataset in args.datasets:
        if dataset == 'coco':
            log.info('Downloading coco')
            if len(os.listdir(f'{img_dir}/coco')) == 0:
                download_coco(args)
            else:
                log.info(f'COCO is already downloaded at {img_dir}/coco ')
        elif dataset == 'construction':
            log.info('Downloading construction')
            if len(os.listdir(f'{img_dir}/construction_images')) == 0:
                download_construction(args)
            else:
                log.info(f'Construction is already downloaded at {img_dir}/construction_images ')
            
        elif dataset == 'refclef':
            log.info('Downloading refclef')
            if len(os.listdir(f'{img_dir}/saiapr_tc-12')) == 0:
                download_refclef(args)
            else:
                log.info(f'Refclef is already downloaded at {img_dir}/saiapr_tc-12')
            
        elif dataset == 'open_images':
            log.info('Downloading open images')
            if len(os.listdir(f'{img_dir}/open_images')) == 0:
                download_open_images(args)
            else:
                log.info(f'Open images is already downloaded at {img_dir}/open_images')
        elif dataset == 'nyuv2':
            log.info('Downloading NYUv2')
            if len(os.listdir(f'{img_dir}/nyuv2')) == 0:
                save_nyuv2(args)
            else:
                log.info(f'NYUv2 is already downloaded at {img_dir}/nyuv2')
        elif dataset == 'visual_genome':
            log.info('Downloading visual genome')
            if len(os.listdir(f'{img_dir}/visual_genome')) == 0:
                download_visual_genome(args)
            else:
                log.info(f'Visual genome is already downloaded at {img_dir}/visual_genome')
        elif dataset == 'blended_mvs':
            log.info('Downloading Blended MVS')
            if len(os.listdir(f'{img_dir}/blended_mvs')) == 0:
                download_blended_mvs(args)
            else:
                log.info(f'Blended MVS is already downloaded at {img_dir}/blended_mvs')
        elif dataset == 'dtu':
            log.info('Downloading DTU')
            if len(os.listdir(f'{img_dir}/dtu_images')) == 0:
                download_dtu(args)
            else:
                log.info(f'DTU is already downloaded at {img_dir}/dtu_images')
        
        

        


    


# @hydra.main(config_path='../configs',config_name='default')
# def main(cfg: DictConfig):
#     if cfg.prjpaths.data_dir is None or cfg.prjpaths.output_dir is None:
#         print("Please provide data_dir and output_dir paths in `configs/prjpaths/default.yaml`")
#         return
#     log.debug('\n' + OmegaConf.to_yaml(cfg))
    
#     for dataset in cfg.datasets_to_download:  
#         print(f"\n\nDownloading {dataset}...")  
#         if dataset=='coco':
            
#             if 
#             download_coco(cfg)
#         elif dataset=='construction':
#             download_construction(cfg)
#         elif dataset=='refclef':
#             download_refclef(cfg)
#         elif dataset=='open_images':
#             download_open_images(cfg)
#         elif dataset=='nyuv2':
#             save_nyuv2(cfg)
#         elif dataset=='visual_genome':
#             download_visual_genome(cfg)
#         elif dataset=='blended_mvs':
#             download_blended_mvs(cfg)
#         elif dataset=='scannet':
#             download_scannet(cfg)
#         elif dataset=='dtu':
#             download_dtu(cfg)
#         else:
#             raise NotImplementedError
    

if __name__=='__main__':
    main()

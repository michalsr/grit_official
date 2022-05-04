import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from utils.io import (mkdir_if_not_exists, extract_targz, download_from_url)
import params 
from grit_paths import GritPaths
import logging 
from argparse import ArgumentParser
log = logging.getLogger('__main__')
logging.basicConfig(filename='example.log', encoding='utf-8')

def download_distortions(args):
    base = params.GRIT_BASE
    img_dir = f'{base}'
    mkdir_if_not_exists(img_dir,recursive=True)
    download_from_url(params.DISTORTIONS, img_dir)
    extract_targz(
        os.path.join(img_dir,'distortions.tar.gz'),
       img_dir)
def main():
    logging.info('Downloading distortions')
    parser = ArgumentParser()
    parser.add_argument('--data_dir',default='/data/michal5')
    args = parser.parse_args()
    if not os.path.exists(f'{args.data_dir}/distortions'):
        download_distortions(args)
    elif len(os.listdir(f'{args.data_dir}/distortions')) == 0:
        download_distortions(args)
    else:
        logging.info(f'Distortions already downloaded at {args.data_dir} ')


# @hydra.main(config_path='../configs',config_name='default')
# def main(cfg: DictConfig):
#     if cfg.prjpaths.data_dir is None or cfg.prjpaths.output_dir is None:
#         print("Please provide data_dir and output_dir paths in `configs/prjpaths/default.yaml`")
#         return
#     log.debug('\n' + OmegaConf.to_yaml(cfg))
#     download_distortions(cfg)
    

if __name__=='__main__':
    main()

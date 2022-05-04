import os 
from data_loader import EvalDataset
import torch
import clip
import sys 
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import io
from models import clip
import logging
import json 
from tqdm import tqdm 
from torchvision.datasets import CIFAR100
from argparse import ArgumentParser
import params

def get_model(args,device):

    model, preprocess = None, None
 
    if 'clip' in args.model:
        if 'ViT' in args.model:
            model, preprocess = clip.load("ViT-B/32", device=device)
    return model,preprocess
def get_image(args,img_loc,bbox):
    im = Image.open(f'{args.data_dir}/GRIT/images/{img_loc}')
    new_im = im.crop(bbox)
    return new_im 
def get_top_pred(args,model,crop_img,text,preprocess,device):
    if 'clip' in args.model:
        image_input = crop_img.to('cuda')
        text_input = text.to('cuda')
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
     
        similarity = (100.0 * image_features @ text).softmax(dim=-1)

        values, indices = torch.topk(similarity,1)
       
        return values, indices
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')     
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    handler_2 = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler_2)


    return logger
def sort_by_dataset(logger):
    data_dict = {'nyuv2':[],'open_images':[],'coco':[]}
    objects = io.load_json_object(f'/data/michal5/GRIT/samples/ablation/categorization.json')
    for entry in objects:
        if entry['output_options'] == 'coco_categories': 
            data_dict['coco'].append(entry)
        elif entry['output_options'] == 'open_images_categories':
            data_dict['open_images'].append(entry)
        else:
            assert entry['output_options']  == 'nyuv2_categories'
            data_dict['nyuv2'].append(entry)
    total_dataset = len(data_dict['nyuv2'])+len(data_dict['open_images']) + len(data_dict['coco'])
    logger.info(f'Total dataset length {total_dataset}')
    return data_dict 
def convert_submission(args,cat_preds,logger,task,model):
    if not os.path.exists(f'{args.output_dir}/ablation'):
        io.mkdir_if_not_exists(f'{args.output_dir}/ablation',recursive=True)
        
    logger.info(f'Saved results to {args.output_dir}/ablation/{task}.json')
    io.dump_json_object(cat_preds,f'{args.output_dir}/ablation/{task}.json')
    if not os.path.exists(f'{args.output_dir}/ablation/params.json'):
        param_dict = {"params_in_millions":len(list(model.model.parameters()))}
        io.dump_json_object(cat_preds,f'{args.output_dir}/ablation/params.json')

def evaluate(args):
    logger = setup_logger('logger',f'{args.output_dir}/log_file.log')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_examples = 0 
    final_predictions = {}
    logger.info('Getting model')
    if 'clip' in args.model:
        model = clip.CLIP(args,logger)

   
    for task in args.tasks:
        if task == 'categorization':
            logger.info('Predicting for categorization')
            cat_preds = []
            dataset_dict = sort_by_dataset(logger)
   
            for data in dataset_dict:
                cats = io.load_json_object(f'{args.data_dir}/GRIT/output_options/{data}_categories.json')
                dataset = model.get_data_set(data=dataset_dict[data],task='categorization')
                data_loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False)
                total_examples += len(list(dataset.idx_to_id.keys()))
                logger.info('Start eval')
                id_list = []
                img_features = []
                txt_features = []
                for entry,indicies in tqdm(data_loader):
                    # output from model 
                    confidence, class_pred= model.get_top_pred(entry,data)
                    for i,conf in enumerate(confidence):
                        final_answer = {}
                        final_answer['example_id'] = dataset.idx_to_id[indicies[i].item()]
                        
                        final_answer['confidence'] = conf.item()
                        final_answer['words'] = cats[class_pred[i].item()]
                        cat_preds.append(final_answer)
        convert_submission(args,cat_preds,logger,task,model)        
          
               



def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir',default='/data/michal5')
    parser.add_argument("--tasks",nargs='+',default=None)
    parser.add_argument("--model",default=None)
    parser.add_argument("--output_dir",default=None)
    parser.add_argument("--batch_size",default=32)
    args = parser.parse_args()
    output_dir = f'{args.output_dir}'
    if not os.path.exists(output_dir):
        io.mkdir_if_not_exists(output_dir,recursive=True)
    with open(output_dir+'/command_line.txt','w+') as f:
      json.dump(args.__dict__,f,indent=2)
    evaluate(args)
if __name__=='__main__':
    main()

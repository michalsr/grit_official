import torch 
from utils import io
from PIL import Image 
class EvalDataset(torch.utils.data.Dataset):

    def __init__(self,task,args,data,transform):
        self.task = task 
        self.args = args
        self.data = data
        self.transform = transform
        self.idx_to_id = self.get_id_tensor()
     
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        entry = self.data[idx]
        im = self.get_image(entry['image_id'],entry['task_bbox'])
        im = self.transform(im)
     
        return im,idx
    def get_image(self,img_loc,bbox):
        im = Image.open(f'{self.args.data_dir}/GRIT/images/{img_loc}')
        new_im = im.crop(bbox)
        return new_im 
    def get_id_tensor(self):
        idx_to_id = {}

        for i,entry in enumerate(self.data):
            idx_to_id[i] = entry['example_id']
        return idx_to_id


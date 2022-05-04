from models import model,templates
import torch
from data_loader import EvalDataset
import clip
from utils import io 
from tqdm import tqdm 
class CLIP(model.Model):
    def __init__(self,args,logger,backbone='vit'):
        super().__init__(args)
        self.args = args
        self.device = 'cuda'
        self.backbone = backbone
        self.model,self.preprocess = self.get_model()
        self.templates = templates.imagenet_templates
        self.text_embeddings = {}
        self.logger = logger
    
    
    def get_model(self):
        if self.backbone != 'vit':
            model, preprocess = clip.load(self.backbone, device=self.device)
        else:
            model, preprocess = clip.load("ViT-B/32", device=self.device)
        return model, preprocess
    
    def get_text_embeddings(self,classes,dataset_name):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classes):
                texts = [template.format(classname) for template in self.templates] #format with class
                texts = clip.tokenize(texts).cuda() #tokenize
                class_embeddings = self.model.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            self.text_embeddings[dataset_name] = zeroshot_weights
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()

        return zeroshot_weights
    def get_data_set(self,data,task):
        dataset = EvalDataset(task=task,args=self.args,data=data,transform=self.preprocess)
        return dataset
    
    def get_top_pred(self,img_batch,dataset_name):
        if dataset_name not in self.text_embeddings:
            cats = io.load_json_object(f'{self.args.data_dir}/GRIT/output_options/{dataset_name}_categories.json')
            self.get_text_embeddings(cats,dataset_name)
        text_embeddings = torch.stack(self.text_embeddings[dataset_name], dim=1).cuda()

        image_input = img_batch.to('cuda')
     
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            #text_features = model.encode_text(text_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        #text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_embeddings).softmax(dim=-1)

        values, indices = torch.topk(similarity,1)
        return values, indices





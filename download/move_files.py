import os 
import shutil 

for f in os.listdir('/data/michal5/GRIT/images/open_images/test/data/'):
    shutil.move('/data/michal5/GRIT/images/open_images/test/data/'+f,'/data/michal5/GRIT/images/open_images/test/')
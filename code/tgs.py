
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from random import randint

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from sklearn.model_selection import train_test_split

from skimage.transform import resize

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout

from tqdm import tqdm_notebook

from fastai.conv_learner import *
from fastai.dataset import *
#from fastai.models.resnet import vgg_resnet50

import json


# In[2]:


def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax


# In[3]:


PATH = Path("../input")
MASKS_FN = "train.csv"
META_FN = "depths.csv"
masks_csv = pd.read_csv(PATH/MASKS_FN)
meta_csv = pd.read_csv(PATH/META_FN)


# In[4]:


def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax


# In[5]:


TRAIN_DN = '../data/tgs/train/images'
MASKS_DN = '../data/tgs/train/masks'
sz = 128
bs = 64
nw = 16


# In[6]:


class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
    def get_y(self, i): return open_image(os.path.join(self.path, self.y[i]))
    def get_c(self): return 0

    


# In[7]:


x_names = np.array([Path(TRAIN_DN)/f'{o}.png' for o in masks_csv['id']])
y_names = np.array([Path(MASKS_DN)/f'{o}.png' for o in masks_csv['id']])

#x_names = [np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=True)) for idx in tqdm_notebook(train_df.index)]
#y_names = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) for idx in tqdm_notebook(train_df.index)]


# In[8]:


val_idxs = list(range(1008))
((val_x,trn_x),(val_y,trn_y)) = split_by_idx(val_idxs, x_names, y_names)



# In[9]:


aug_tfms = [RandomRotate(4, tfm_y=TfmType.CLASS),
            RandomFlip(tfm_y=TfmType.CLASS),
            RandomLighting(0.05, 0.05, tfm_y=TfmType.CLASS)]


# In[79]:


import pandas as pd
testfile_name = pd.read_csv("testfile.csv") #This testfile.csv file is in code folder. 

TEST_DN = '../data/tgs/test'
t_names = np.array([Path(TEST_DN)/f'{o}' for o in testfile_name['img']])
test_name = (t_names ,t_names)


# In[80]:


tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x,trn_y), (val_x,val_y), tfms, test = test_name, path= PATH)
md = ImageData(PATH, datasets, bs, num_workers=16, classes=None)
denorm = md.trn_ds.denorm


# In[81]:


x,y = next(iter(md.trn_dl))


# In[82]:


print(len(y))


# In[83]:


x.shape,y.shape


# In[84]:


f = resnet34
cut,lr_cut = model_meta[f]

def get_base():
    layers = cut_model(f(True), cut)
    return nn.Sequential(*layers)

def dice(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


class StdUpsample (nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.conv = nn.ConvTranspose2d(nin, nout, 2, stride=2)
        self.bn = nn.BatchNorm2d(nout)
        
    def forward(self, x): return self.bn(F.relu(self.conv(x)))

class Upsample34(nn.Module):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn
        self.features = nn.Sequential(
            rn, nn.ReLU(),
            StdUpsample(512,256),
            StdUpsample(256,256),
            StdUpsample(256,256),
            StdUpsample(256,256),
            nn.ConvTranspose2d(256, 1, 2, stride=2))
        
    def forward(self,x): return self.features(x)[:,0]


class UpsampleModel ():
    def __init__(self,model,name='upsample'):
        self.model,self.name = model,name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [lr_cut]))
        return lgs + [children(self.model.features)[1:]]


# In[85]:


m_base = get_base()


# In[86]:


m = to_gpu(Upsample34(m_base))
models = UpsampleModel(m)


# In[87]:


learn = ConvLearner(md, models) # built my model
learn.opt_fn=optim.Adam
learn.crit=nn.BCEWithLogitsLoss()
learn.metrics=[accuracy_thresh(0.5), dice]


# In[88]:


learn.summary()


# In[89]:


learn.freeze_to(1)


# In[31]:


learn.lr_find()
learn.sched.plot()


# In[73]:


lr=1
wd=1
lrs = np.array([lr,lr,lr])/2


# In[74]:


learn.fit(lr,1, wds=wd, cycle_len=1,use_clr=(20,8))


# In[ ]:


#import pdb; 
#pdb.set_trace()
learn.predict(is_test=True) 


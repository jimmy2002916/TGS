
# coding: utf-8

# In[117]:


import numpy as np
import pandas as pd

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout

from random import randint

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from sklearn.model_selection import train_test_split
from skimage.transform import resize

from tqdm import tqdm_notebook

from fastai.conv_learner import *
from fastai.dataset import *
#from fastai.models.resnet import vgg_resnet50

import json


# In[118]:


def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax


# In[119]:


PATH = Path("..")
MASKS_FN = "train.csv"
META_FN = "depths.csv"
masks_csv = pd.read_csv("../input/train.csv")
meta_csv = pd.read_csv("../input/train.csv")


# In[120]:


def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax


# In[121]:


TRAIN_DN = 'data/tgs/train/images'
MASKS_DN = 'data/tgs/train/masks'
sz = 128
bs = 64
nw = 16


# In[122]:


class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
    def get_y(self, i): return open_image(os.path.join(self.path, self.y[i]))
    def get_c(self): return 0

    


# In[123]:


x_names = np.array([Path(TRAIN_DN)/f'{o}.png' for o in masks_csv['id']])
y_names = np.array([Path(MASKS_DN)/f'{o}.png' for o in masks_csv['id']])


# In[124]:


val_idxs = list(range(1)) # 1008
((val_x,trn_x),(val_y,trn_y)) = split_by_idx(val_idxs, x_names, y_names)



# In[125]:


aug_tfms = [RandomRotate(4, tfm_y=TfmType.CLASS),
            RandomFlip(tfm_y=TfmType.CLASS),
            RandomLighting(0.05, 0.05, tfm_y=TfmType.CLASS)]


# In[126]:


#!rm ../data/tgs/test/.DS_Store


# In[127]:


file_list = os.listdir("../data/tgs/test")
if '.DS_Store' in file_list:
    print("T")
else:
    print("F")


# In[130]:


#file_list.remove('.DS_Store')
#file_list.remove('README.md')
#file_list.remove('.git')

if '.DS_Store' in file_list:
    print("T")
else:
    print("F")
    
print(len(file_list))


# In[131]:


testfile_name = pd.DataFrame({'img':file_list})

TEST_DN = 'data/tgs/test'
t_names = np.array([Path(TEST_DN)/f'{o}' for o in testfile_name['img']])
test_name = (t_names ,t_names)


# In[132]:


tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
#datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x,trn_y), (val_x,val_y), tfms, test = test_name, path= PATH)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x,trn_y), test_name, tfms, test = None, path= PATH)
#md = ImageData(PATH, datasets, bs, num_workers=16, classes=None)
md = ImageData(PATH, datasets, bs, num_workers=16, classes=None)
denorm = md.trn_ds.denorm


# In[133]:


x,y = next(iter(md.trn_dl))


# In[134]:


x.shape,y.shape


# In[135]:


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


# In[136]:


m_base = get_base()


# In[137]:


m = to_gpu(Upsample34(m_base))
models = UpsampleModel(m)


# In[138]:


#learn = ConvLearner(md, models) # built my model
learn = ConvLearner(md, models) # change the valid set
learn.opt_fn=optim.Adam
learn.crit=nn.BCEWithLogitsLoss()
learn.metrics=[accuracy_thresh(0.5), dice]


# In[27]:


learn.summary()


# In[155]:


learn.freeze_to(1)


# In[156]:


learn.lr_find()
learn.sched.plot()


# In[153]:


lr=4e-2
wd=1e-7

lrs = np.array([lr/100,lr/10,lr])


# In[154]:


learn.fit(lr,1, wds=wd, cycle_len=1,use_clr=(20,8))


# In[150]:


learn.save('tmp')
learn.load('tmp')


# In[151]:


learn.unfreeze()
learn.bn_freeze(True)


# In[152]:


learn.fit(lrs,1,cycle_len=1,use_clr=(20,8))


# In[155]:


preds_test = learn.predict() 


# In[ ]:


def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


# In[ ]:


testfile_name = pd.DataFrame({'img':file_list})
testfile_name = pd.DataFrame(testfile_name.img.str.split('.',1).tolist(), columns = ['img','png'])


# In[ ]:


img_size_ori = 101
img_size_target = 128

def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)


# In[ ]:


pred_dict = {idx: RLenc(np.round(downsample(preds_test[i]) > 0.5)) for i, idx in enumerate(tqdm_notebook(testfile_name["img"]))}


# In[ ]:


sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')


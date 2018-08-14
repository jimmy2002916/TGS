# TGS Salt Identification Challenge

The idea is to use Unet with Fastai. As Unet is currently the most efficient model for image recognization. Also, instead of using keras, the reason I go for fastai is it is really easy to understand, and quite efficient source with pytorch running as a backend. 

Currently I am ranked at 248 / 1149, please use "Jimmy H" as a keyword to search on https://www.kaggle.com/c/tgs-salt-identification-challenge/leaderboard . As for the strategies I use, I will write down my thoughts about the competition after it finish. 

There's a big opportunity, which is to create a Unet downpath, and add a classifier in the end, and train that on Imagenet, and now since you got a Imagenet-trained classifier, which is specifically designed to be a good backboe of Unet, and you should be able to be pretty close to win this competition.

You can use https://nbviewer.jupyter.org/ to open jupyter notebook if you failed to view the source code.

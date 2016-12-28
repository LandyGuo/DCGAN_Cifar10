#coding=utf-8
import glob
import numpy as np
from matplotlib import pylab as plt
import os
import cPickle



def unpickle(file):
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict

def get_images(file):
	data = unpickle(file)['data']
	channels = np.split(data,3,1)
	reshape_channels = [c.reshape([10000,32,32,1]) for c in channels]
	image_arr = np.concatenate(reshape_channels, axis=3)
	return image_arr


def load_cifar10(cifar_dir):
	images = [get_images(f) for f in glob.glob(
		os.path.join(cifar_dir,"data_batch_[0-9]"))]
	arr =  np.concatenate(images,axis=0)/255.0
	return arr


# r = load_cifar10('./cifar-10-batches-py')
# plt.imshow(r[2])
# plt.show()
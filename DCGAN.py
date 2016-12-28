#coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import os
import codecs
import logging
from PIL import Image

from cifar10 import load_cifar10

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='Train.log',
                filemode='w+')

if not os.path.exists('./Model'):
	os.makedirs("./Model")
##############################################################################
#finetune params
"""
if you want to tune this DCGAN model into your own needs,you need to:
1. finetune the params below
2. write your own generator follow the "deconv priciple"
3. prepare loading your dataset methods

"""

Use_Real_Data = True
Image_Size = 32
Image_Channel = 3
BN_DECAY = 0.9
Batch_size = 64
Learning_rate = 0.0002
Train_Epochs = 20
Discriminator_Start_Filters = 64
Generator_Start_Filters = 64
Save_Path = "./"
Learning_rate_Decay = 0.999


IMAGES_DIR = "./bu4d64_n"
#######################################################################

#datasets manipulations

#function to load mnist data
def load_mnist_dataset(one_hot=True):
	"""Load the MNIST handwritten digits dataset.
	:param mode: 'supervised' or 'unsupervised' mode
	:param one_hot: whether to get one hot encoded labels
	:return: train, validation, test data:
	        for (X, y) if 'supervised',
	        for (X) if 'unsupervised'
	"""
	mnist = input_data.read_data_sets("./MNIST_data", one_hot=one_hot)

	# Training set
	trX = mnist.train.images
	trY = mnist.train.labels

	# Validation set
	vlX = mnist.validation.images
	vlY = mnist.validation.labels

	# Test set
	teX = mnist.test.images
	teY = mnist.test.labels

	return trX, trY, vlX, vlY, teX, teY


class Labels(object):
	'''
	definition for facial emotion labels
	include: copnversion from labels' num to labels' name and vice versa
	'''
	NE = 0
	AN = 1
	DI = 2
	HA = 3
	SA = 4
	SU = 5
	FE = 6

	@classmethod
	def getNumFromLabel(cls,label_name):
		#convert emotion label_name into number
		try:
			return cls.__dict__[label_name]
		except KeyError:
			return None

	@classmethod
	def getLabelFromNum(cls,num):
		#convert num into emotion label
		for k,v in cls.__dict__.items():
			if v==num:
				return k
		return None

	@classmethod
	def to_one_hot(cls,data):
		rows, cols = len(data),7
		one_hot_data = np.zeros([rows,cols])
		for row in range(rows):
			index = data[row]
			one_hot_data[row,index] = 1
		return one_hot_data




class DataSet(object):

	def __init__(self,real_data=True):
		self._images, self._labels = self.load_image(fake=not real_data)
		self._index_in_epoch = 0
		self._epoch_finished = 0
		self.samples = self._images.shape[0]

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def epoch_finished(self):
		return self._epoch_finished

	@property
	def index_in_epoch(self):
		return self._index_in_epoch

	def load_image(self, grayScaleNormalize=True,fake=True):
		if fake:
			print "loading fake data"
			images = np.random.random_sample([1000,64,64,1])
			labels = np.random.random_integers(0,6,size=(1000,))
		else:
			print "loading real data"
			filelists = codecs.open('img_idx','r','utf-8').readlines()
			filenames = [os.path.join(IMAGES_DIR,f.strip()) for f in filelists]
			labels = [Labels.getNumFromLabel(filename.split('_')[1]) for filename in filelists]
			images = []
			for imgfile in filenames:
				image = Image.open(imgfile, 'r').convert("L")
				image = np.asarray(image)
				if grayScaleNormalize:
					image = (image-np.mean(image))/np.std(image)
				image = image.reshape((64,64,1))
				images.append(image)
		assert len(images)==len(labels)
		return np.asarray(images),np.asarray(labels)

	def next_batch(self, batch_size=50,one_hot_label=False):
		start_index = self._index_in_epoch
		end_index = start_index + batch_size
		if end_index > self.samples:
			#end one epoch ,try to shuffle the images and labels
			reorder = np.arange(self.samples)
			np.random.shuffle(reorder)
			self._images = self._images[reorder]
			self._labels = self._labels[reorder]
			#reset start_index
			start_index = 0
			end_index = start_index + batch_size
			self._epoch_finished+=1
		self._index_in_epoch = end_index
		sl = slice(start_index, start_index+batch_size)
		return self._images[sl], (Labels.to_one_hot(self._labels[sl]) if one_hot_label else self._labels[sl])


#################################################################################


#placeholders
def create_placeholders():
	z = tf.placeholder(dtype=tf.float32,shape=[Batch_size,100])
	image = tf.placeholder(dtype=tf.float32,shape=[Batch_size,Image_Size,Image_Size,Image_Channel])
	return z, image

#######################################################################
class batch_norm(object):
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		with tf.variable_scope(name):
		    self.epsilon  = epsilon
		    self.momentum = momentum
		    self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x,
											decay=self.momentum, 
											updates_collections=None,
											epsilon=self.epsilon,
											scale=True,
											is_training=train,
											scope=self.name)
#untils function
# batch normalization : deals with poor initialization helps gradient flow
d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')

g_bn0 = batch_norm(name='g_bn0')
g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')
g_bn3 = batch_norm(name='g_bn3')


def conv2d(x,filter_shape,stddev=0.02):
	w = tf.get_variable("w",shape=filter_shape,initializer=tf.random_normal_initializer(stddev=stddev))
	b = tf.get_variable("b",shape=[filter_shape[-1]],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
	x = tf.nn.conv2d(x,w,[1,2,2,1],padding='SAME')
	x = tf.nn.bias_add(x,b)
	return x


def deconv2d(x,out_shape,stddev=0.02):
	w = tf.get_variable("w",shape=[5,5,out_shape[-1],x.get_shape()[-1]],initializer=tf.random_normal_initializer(stddev=stddev))
	b = tf.get_variable("b",shape=[out_shape[-1]],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
	x = tf.nn.conv2d_transpose(x,w,out_shape,[1,2,2,1])
	x = tf.nn.bias_add(x,b)
	return x

def linear(x,affine_Matrix_shape,stddev=0.02):
	w = tf.get_variable("w",shape=[x.get_shape()[-1],affine_Matrix_shape[-1]],initializer=tf.random_normal_initializer(stddev=stddev))
	b = tf.get_variable("b",shape=[affine_Matrix_shape[-1]],initializer=tf.constant_initializer(0.0))
	x = tf.add(tf.matmul(x,w),b)
	return x


def leakyRelu(x,scope=0.2):
	return tf.maximum(x,scope*x)


def gen_batches(data,batch_size=50,shuffle=True):
	data = np.array(data)
	data_size = data.shape[0]
	if shuffle:
		idx = np.arange(data_size)
		np.random.shuffle(idx) 
		data = data[idx]
	batch_length = (data_size-1)//batch_size
	for i in range(batch_length):
		yield data[i*batch_size:i*batch_size+batch_size]

###################################################################################

def generator(z,reuse=False,isTrain=True):
	with tf.variable_scope("generator") as scope:
		if reuse:
			scope.reuse_variables()

		s = Image_Size

		"""
		#for mnist
		s8,s4,s2 = 4,7,14
		with tf.variable_scope("linear1"):
			x = linear(z,[100,64*4*s8*s8])
			x = tf.reshape(x,[Batch_size,s8,s8,64*4])
			x = g_bn0(x,isTrain)
			project_out = tf.nn.relu(x)
			# print x.get_shape()


		with tf.variable_scope("deconv1"):
			x = deconv2d(project_out,[Batch_size,s4,s4,64*2])
			x = g_bn1(x,isTrain)
			deconv1_out = tf.nn.relu(x)
			# print deconv1_out.get_shape()

		with tf.variable_scope("deconv2"):
			x = deconv2d(deconv1_out,[Batch_size,s2,s2,64])
			x = g_bn2(x,isTrain)
			deconv2_out = tf.nn.relu(x)
			# print deconv2_out.get_shape()

		with tf.variable_scope("deconv3"):
			x = deconv2d(deconv2_out,[Batch_size,s,s,Image_Channel])
			# x = g_bn3(x,isTrain)
			deconv3_out = tf.nn.tanh(x)

		return deconv3_out


		"""
		"""For cifar10"""
		s8,s4,s2,s = 4,8,16,32
		#project and reshape
		with tf.variable_scope("linear1"):
			x = linear(z,[100,64*8*s8*s8])
			x = tf.reshape(x,[Batch_size,s8,s8,64*8])
			x = g_bn0(x,isTrain)
			project_out = tf.nn.relu(x)
			# print x.get_shape()

		with tf.variable_scope("deconv1"):
			x = deconv2d(project_out,[Batch_size,s4,s4,64*4])
			x = g_bn1(x,isTrain)
			deconv1_out = tf.nn.relu(x)
			# print deconv1_out.get_shape()

		with tf.variable_scope("deconv2"):
			x = deconv2d(deconv1_out,[Batch_size,s2,s2,64*2])
			x = g_bn2(x,isTrain)
			deconv2_out = tf.nn.relu(x)
			# print deconv2_out.get_shape()

		with tf.variable_scope("deconv3"):
			x = deconv2d(deconv2_out,[Batch_size,s,s,Image_Channel])
			# x = g_bn3(x,isTrain)
			deconv3_out = tf.nn.tanh(x)
			# print deconv2_out.get_shape()

	return deconv3_out

	"""For bu4d
		s16,s8,s4,s2 = 4,8,16,32
		#project and reshape
		with tf.variable_scope("linear1"):
			x = linear(z,[100,64*8*s16*s16])
			x = tf.reshape(x,[Batch_size,s16,s16,64*8])
			x = g_bn0(x,isTrain)
			project_out = tf.nn.relu(x)
			# print x.get_shape()


		with tf.variable_scope("deconv1"):
			x = deconv2d(project_out,[Batch_size,s8,s8,64*4])
			x = g_bn1(x,isTrain)
			deconv1_out = tf.nn.relu(x)
			# print deconv1_out.get_shape()

		with tf.variable_scope("deconv2"):
			x = deconv2d(deconv1_out,[Batch_size,s4,s4,64*2])
			x = g_bn2(x,isTrain)
			deconv2_out = tf.nn.relu(x)
			# print deconv2_out.get_shape()

		with tf.variable_scope("deconv3"):
			x = deconv2d(deconv2_out,[Batch_size,s2,s2,64])
			x = g_bn3(x,isTrain)
			deconv3_out = tf.nn.relu(x)
			# print deconv2_out.get_shape()


		with tf.variable_scope("deconv4"):
			x = deconv2d(deconv3_out,[Batch_size,s,s,Image_Channel])
			# x = bn(x)
			deconv4_out = tf.nn.tanh(x)
			# print deconv3_out.get_shape()

	return deconv4_out
	"""

##################################################################
def discrimator(image,reuse=False):

	with tf.variable_scope("discrimator") as scope:
		if reuse:
			scope.reuse_variables()

		with tf.variable_scope("conv1"):
			x = conv2d(image,[5,5,Image_Channel,64])
			# x = d_bn1(x)
			conv1_out = leakyRelu(x)
			# print conv1_out.get_shape()

		with tf.variable_scope("conv2"):
			x = conv2d(conv1_out,[5,5,64,128])
			x = d_bn1(x)
			conv2_out = leakyRelu(x)
			# print conv2_out.get_shape()

		with tf.variable_scope("conv3"):
			x = conv2d(conv2_out,[5,5,128,256])
			x = d_bn2(x)
			conv3_out = leakyRelu(x)
			# print conv3_out.get_shape()

		with tf.variable_scope("conv4"):
			x = conv2d(conv3_out,[5,5,256,512])
			x = d_bn3(x)
			conv4_out = leakyRelu(x)
			print "conv4_out:",conv4_out.get_shape()


		with tf.variable_scope("linear1"):
			x = tf.reshape(conv4_out,[Batch_size,-1])
			d_out = linear(x,[x.get_shape()[-1],1])


	return d_out


##########################################################
def loss_function(z,image):
	"""
	update discriminator first ,then generator"""
	#generate image from z
	fake_image = generator(z)
	real_image = image
	logging.debug("fake_image size:%s"%str(fake_image.get_shape()))
	logging.debug("real_image size:%s"%str(real_image.get_shape()))
	#discriminate fake image and real image
	d_fake_pro = discrimator(fake_image)
	d_real_pro = discrimator(real_image,True)
	#d_fake_pro->0 d_real_pro->1   1:real 0:fake
	d4fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_fake_pro,tf.zeros_like(d_fake_pro,dtype=tf.float32)))
	d4real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_real_pro,tf.ones_like(d_real_pro,dtype=tf.float32)))
	gfake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_fake_pro,tf.ones_like(d_fake_pro,dtype=tf.float32)))
	d_total_loss = d4fake_loss + d4real_loss
	g_total_loss = gfake_loss
	#dicriminator precision
	print "d_fake_pro_shape",d_fake_pro.get_shape()
	fake_right = tf.cast(tf.nn.sigmoid(d_fake_pro)<0.5,tf.float32)
	real_right = tf.cast(tf.nn.sigmoid(d_real_pro)>=0.5,tf.float32)
	print "fake_right shape:",fake_right.get_shape()
	precision = tf.reduce_mean(tf.concat(0,[fake_right,real_right]))

	"""optimizer"""
	g_variables = []
	d_variables = []
	for x in tf.trainable_variables():
		if x.op.name.startswith("discrimator"):
			d_variables.append(x)
		elif x.op.name.startswith("generator"):
			g_variables.append(x)

	with tf.variable_scope("learn_param"):
		lr = tf.get_variable("learning_rate",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(Learning_rate),trainable=False)
	d_grads, _ = tf.clip_by_global_norm(tf.gradients(d_total_loss, d_variables), 5)
	g_grads, _ = tf.clip_by_global_norm(tf.gradients(g_total_loss, g_variables), 5)
	optimizer = tf.train.AdamOptimizer(lr,beta1=0.5)
	
	
	
	with tf.device("/gpu:0"):
		d_optim = optimizer.apply_gradients(zip(d_grads, d_variables))
		# d_optim = tf.train.AdamOptimizer(lr).minimize(d_total_loss, var_list=d_variables)
	with tf.device("/gpu:1"):
		g_optim = optimizer.apply_gradients(zip(g_grads, g_variables))
		# g_optim = tf.train.AdamOptimizer(lr).minimize(g_total_loss, var_list=g_variables)

	return d_total_loss,g_total_loss,d_optim, g_optim,precision

#########################################################################

def create_feed_dict(placeholder_z,placeholder_image,z,image):
	return {placeholder_z:z,placeholder_image:image}



def train():
	#for minist
	# trX,_,vlX,_,teX,_ = load_mnist_dataset()
	# testdata = trX
	#for cifar10
	testdata = load_cifar10('./cifar-10-batches-py')
	
	#For 64*64 dataset	
	# ds = DataSet(Use_Real_Data)
	# testdata = ds.images
	
	g = tf.Graph()

	with g.as_default():
		sess = tf.Session()

		z, image = create_placeholders()
		d_total_loss,g_total_loss,d_optim, g_optim,precision = loss_function(z,image)

		saver = tf.train.Saver()
		sess.run(tf.initialize_all_variables())


		module_file = tf.train.latest_checkpoint('./Model')

		if module_file:
			saver.restore(sess,module_file)
			logging.debug("load module file:%s, model restored" % module_file)


		for epoch in range(Train_Epochs):
			for step,real_images in enumerate(gen_batches(testdata,batch_size=Batch_size)):
				batch_images = real_images.reshape([Batch_size,Image_Size,Image_Size,Image_Channel])
				batch_z = np.random.uniform(-1, 1, [Batch_size, 100]).astype(np.float32)
				#optimize d once, optimize g twice
				_ ,dloss,prec = sess.run([d_optim,d_total_loss,precision],feed_dict=create_feed_dict(z,image,batch_z,batch_images))
				for _ in range(3):
					_ ,gloss = sess.run([g_optim,g_total_loss],feed_dict={z:batch_z})


				logging.debug("Epoch:%s step:%s dloss:%s gloss:%s d_precision:%s"%(epoch,step,dloss,gloss,prec))
				if (step)%1000==0:
					with tf.variable_scope("learn_param",reuse=True):
						lr = tf.get_variable("learning_rate",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(Learning_rate),trainable=False)
						lr = tf.assign(lr,lr*Learning_rate_Decay)
						logging.debug("learning_rate decay:%6f" % sess.run(lr))
					batch_z = np.random.uniform(-1, 1, [Batch_size, 100]).astype(np.float32)
					# samples = sess.run([sample_image],feed_dict={z:batch_z})
					# plot_sample(samples, str(epoch)+"_"+str(step))
					logging.debug("Sample saved")
					saver.save(sess,"./Model/GANModel",global_step=epoch)


def plot_sample(batch_images,num_str,show = False,save=True):
		batch_images = inverse_transform(np.array(batch_images))
		# batch_images = np.array(batch_images>0.5,dtype=np.float32)

		ori = batch_images.reshape([-1,Image_Size,Image_Size,Image_Channel])
		figs, axes = plt.subplots(8,8, figsize=(7,7))
		for ax in axes.flatten():
			ax.set_xticks([])
			ax.set_yticks([])
			ax.axis('off')
		images_num = batch_images.shape[0]
		for i in range(8):
			for j in range(8):
				axes[i,j].imshow(ori[i*8+j])
		if show:
			plt.show()
		if save:
			plt.savefig('test_%s.png' % num_str)

def inverse_transform(images):
    return (images+1.)/2.

def gen_samples():
	g = tf.Graph()
	with g.as_default():
		sess = tf.Session()
		z, _ = create_placeholders()
		
		fake_image = generator(z)
		sample_image = generator(z,reuse=True,isTrain=False)
		module_file = tf.train.latest_checkpoint('./Model')
		saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
		sess.run(tf.initialize_all_variables())
		if module_file:
			saver.restore(sess,module_file)
			print "load module file:%s, model restored" % module_file

		batch_z = np.random.uniform(-1.0, 1.0, [Batch_size, 100]).astype(np.float32)
		samples = sess.run([sample_image],feed_dict={z:batch_z})
		plot_sample(samples,None,show = True,save=False)



############################################################################
#use train to train DCGAN model, use gen_samples to create learned samples
# train()
gen_samples() 

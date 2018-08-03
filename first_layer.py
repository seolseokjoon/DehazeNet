import sys
import caffe
import numpy as np
import cv2
import math
import os



if __name__ == '__main__':
	if not len(sys.argv) == 2:
		print( 'Usage: python DeHazeNet.py haze_img_path')
		exit()
	else:
		im_path = sys.argv[1]
	#read weights from caffemodel and write into kernel
	caffe_model='./DehazeNet.prototxt'
	caffe_weight='./DehazeNet.caffemodel'
	weight_path='./img'
	caffe_net=caffe.Net(caffe_model,caffe_weight,caffe.TEST)
	weight=caffe_net.params['conv1'][0].data[...]
	print(weight.shape)
	kernel = np.zeros((20,3,5,5))
	kernel = weight;

	#read image and padding
	src = cv2.imread(im_path)
	npad = ((7,8), (7,8), (0,0))
	src = np.pad(src, npad, 'symmetric')
	#print(src.shape)
	
	#write size of image
	height = src.shape[0]
	width = src.shape[1]
	channel = src.shape[2]



	#write RGB into input.txt
	d=open('input.txt','w')
	for y in np.arange(0,height):
		for x in np.arange(0,width):
			d.write(hex(src[y][x][2])+'\n'+hex(src[y][x][1])+'\n'+hex(src[y][x][0])+'\n')
	d.close()

	#set stride, padding, 
	stride=1
	padding=0

	#write size of kernel
	kh=kernel.shape[2]
	kw=kernel.shape[3]
	num_k=kernel.shape[0]
	result = np.int32

	#write weight
	f=open('weights.txt','w')
	for n in np.arange(0,num_k):
		for h in np.arange(0,kh):
			for w in np.arange(0,kw):
				f.write(str(kernel[n][2][h][w])+'\n'+str(kernel[n][1][h][w])+'\n'+str(kernel[n][0][h][w])+'\n')
	f.close()



	#write size of output
	oh=int((height+2*padding-kh)/stride)+1
	ow=int((width+2*padding-kw)/stride)+1

	#write result of CNN into output.txt
	f=open('output.txt','w')
	for out_n in np.arange(0,num_k):
		for h in np.arange(0,oh):
			for w in np.arange(0,ow):
				result=0
				for ch in np.arange(0,channel):
					for k_h in np.arange(0,kh):
						for k_w in np.arange(0,kw):
							W=w*stride+k_w
							H=h*stride+k_h

							result = result + int(src[H][W][ch]*kernel[out_n][ch][k_h][k_w])

				f.write(hex(result)+'\n')
	f.close()



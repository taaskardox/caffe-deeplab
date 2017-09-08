import argparse
import sys
_CAFFE_ROOT = "../"
sys.path.insert(0, _CAFFE_ROOT + "python")
import caffe
import cv2
import numpy as np
import os
import json
from os.path import exists, join, split, splitext
import shutil

#image_path =''
def predict_image(gpu, image_path):
  deploy_net = '../food17/config/food17/test.prototxt'
  weights = '../food17/model/food17/train_iter_3000.caffemodel'
  mean_pixel=[97.382,97.709,98.044]
  net = caffe.Net(deploy_net, weights)
  net.set_phase_test() 
  if gpu >= 0:
    net.set_mode_gpu()
    net.set_device(gpu)
    print('Using GPU ', gpu)
  else:
    net.set_mode_cpu()
    print('Using CPU')
  input_dims = list(net.blobs['data'].data.shape)
  assert input_dims[0] == 1
  batch_size, num_channels, input_height, input_width = input_dims
  print('Input size:', input_dims)

  caffe_in = np.zeros(input_dims, dtype=np.float32)
  output_height = input_height
  output_width = input_width
  with open('test_list.txt', 'r') as tl:
    testlist= tl.read()
  name_list = testlist.split('\n')[0:-1]
  
  #print(name_list)

  with open('pascal_voc.json', 'r') as fp:
    info = json.load(fp)
  palette = np.array(info['palette'], dtype=np.uint8)

  print('Predicting...')
  for file_name in name_list:
    image_path = 'test_input/' + file_name
    image_ori = cv2.imread(image_path).astype(np.float32) - mean_pixel        
    image_size = image_ori.shape
    image = cv2.resize(image_ori, (input_dims[2],input_dims[3]), interpolation = cv2.INTER_CUBIC)
  
    caffe_in[0] = image.transpose([2, 0, 1])
    out = net.forward_all(blobs=[], **{net.inputs[0]: caffe_in})
  #print(out)
    prob = out['pred'][0]

    prediction = np.argmax(prob.transpose([1, 2, 0]), axis=2)
    prediction = cv2.resize(prediction, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
    from PIL import PngImagePlugin, Image
    out_path = 'test_output/'+ file_name[0:-4] + '.png'

    im = Image.fromarray(prediction.astype(np.uint8), mode='P')
    im.putpalette(palette.flatten())
    im.save(out_path)
    return out_path

#if __name__ == "__main__":
#  test_image(-1, image_path)


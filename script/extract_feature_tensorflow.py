import sys
sys.path.insert(0, ".")
import os, pathlib, glob, argparse
import tensorflow as tf
import numpy as np
import torch.nn.functional as F
import lib
from lib.fid.inception_modified import inception_v3
from lib.fid import tf_fid

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="", help="The path to pytorch inceptionv3 weight. You can obtain this by torchvision incetion_v3 function.")
args = parser.parse_args()

inception_path = tf_fid.check_or_download_inception("pretrained")
with tf.gfile.FastGFile("pretrained/classify_image_graph_def.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString( f.read())
    _ = tf.import_graph_def( graph_def, name='FID_Inception_Net')
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

layername = "FID_Inception_Net/pool_3:0"
layer = sess.graph.get_tensor_by_name(layername)
ops = layer.graph.get_operations()
for op_idx, op in enumerate(ops):
    for o in op.outputs:
        shape = o.get_shape()
        if shape._dims != []:
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            # print(o.name, shape, new_shape)
            o.__dict__['_shape_val'] = tf.TensorShape(new_shape)

if len(args.path) < 1:
    dir_format = "/home/xujianjing/LBSGAN/logs/cifar_bs128/eval_epoch_%d/"
    for i in range(1, 200, 5):
        d = dir_format % i
        print("=> tf %s" % d)
        tf_fid.extract_feature_given_path(d, sess)
        os.system("mv %s/mu_sigma.npy %s/../tf/%d_mu_sigma.npy" % (d, d, i))
else:
    print("=> tf %s" % args.path)
    tf_fid.extract_feature_given_path(args.path, sess)
    os.system("mv %s/mu_sigma.npy %s/../tf_mu_sigma.npy" % (args.path, args.path))
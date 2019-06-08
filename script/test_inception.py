"""
A script to test Pytorch and Tensorflow InceptionV3 have consistent behavior.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import tensorflow as tf
import torch
from lib.fid.inception_modified import inception_v3
from PIL import Image
from tf_fid import *

def torch2numpy(x):
    return x.detach().cpu().numpy().transpose(0, 2, 3, 1)

torch.backends.cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)

data_dir = "data/cifar10_test/"
imgs_pil = [Image.open(open(data_dir + s, "rb")).resize((299,299)) for s in os.listdir(data_dir)]
imgs = [np.asarray(img).astype("float32") for img in imgs_pil]
x_arr = np.array(imgs)
# TF InceptionV3 graph receive [0, 255] scale image
feed = {'FID_Inception_Net/ExpandDims:0': x_arr}
# This is identical to TF image transformation
x_arr = (x_arr - 128) * 0.0078125
x_torch = torch.from_numpy(x_arr.transpose(0, 3, 1, 2)).float().cuda()

model = inception_v3(pretrained=True, aux_logits=False, transform_input=True)
#model.load_state_dict(torch.load("pretrained/inception_v3_google.pth"))
model.cuda()
model.eval()

if x_torch.size(2) != 299:
    import torch.nn.functional as F
    x_torch = F.interpolate(x_torch,
            size=(299, 299),
            mode='bilinear',
            align_corners=False)
features = model.get_feature(x_torch)
feature_pytorch = features[-1].detach().cpu().numpy()
print(feature_pytorch.shape)
print(feature_pytorch[0])
np.save("pytorch_feature.npy", feature_pytorch)

inception_path = check_or_download_inception("pretrained")
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

tensor_list = [n.name for n in tf.get_default_graph().as_graph_def().node]

target_layer_names = ["FID_Inception_Net/Mul:0", "FID_Inception_Net/conv:0", "FID_Inception_Net/pool_3:0"]
target_layers = [sess.graph.get_tensor_by_name(l) for l in target_layer_names]

sess.run(tf.global_variables_initializer())
res = sess.run(target_layers, feed)
x_tf = res[0]
feature_tensorflow = res[-1][:, 0, 0, :]
np.save("tensorflow_feature.npy", feature_tensorflow)

print(feature_pytorch[0][:6])
print(feature_tensorflow[0][:6])

"""
x_torch = (torch2numpy(x_torch) + 1) * 127.5
x_tf = (x_tf + 1) * 127.5
x_torch = x_torch.astype("uint8")
x_tf = x_tf.astype("uint8")
for i in range(8):
    Image.fromarray(x_tf[i]).save(open("temp/tf_%d.jpg" % i, "wb"), format="JPEG")
    Image.fromarray(x_torch[i]).save(open("temp/torch_%d.jpg" % i, "wb"), format="JPEG")
    imgs_pil[i].resize((299, 299), Image.BILINEAR).save(open("temp/pil_%d.jpg" % i, "wb"), format="JPEG")
"""

def get_tf_layer(name):
    return sess.run(sess.graph.get_tensor_by_name(name + ':0'), feed)




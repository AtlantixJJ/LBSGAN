import sys
sys.path.insert(0, ".")
import os, pathlib, glob
import torch, torchvision
import numpy as np
import torch.nn.functional as F
import lib, argparse
from lib.fid.inception_origin import inception_v3
from lib.fid import tf_fid, fid_score

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="", help="The path to pytorch inceptionv3 weight. You can obtain this by torchvision incetion_v3 function.")
parser.add_argument("--gpu", default=-1, help="Use cuda, -1 for CPU")
args = parser.parse_args()

cuda = (args.gpu != -1)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

model = inception_v3(pretrained=True, aux_logits=False, transform_input=False)
evaluator = lib.evaluator.FIDEvaluator(model, "/home/xujianjing/data/cifar10_image/cifar10_test/", cuda=cuda)

if len(args.path) < 1:
    dir_format = "/home/xujianjing/LBSGAN/logs/cifar_bs128/eval_epoch_%d/"
    for i in range(1, 200, 5):
        d = dir_format % i
        print("=> original %s" % d)
        evaluator.calculate_statistics_given_path(d)
        os.system("mv %s/mu_sigma.npy %s/../original/%d_mu_sigma.npy" % (d, d, i))
    model.load_state_dict(torch.load("pretrained/inception_v3_google.pth"))
    evaluator.model = model
    for i in range(1, 200, 5):
        d = dir_format % i
        print("=> changed %s" % d)
        evaluator.calculate_statistics_given_path(d)
        os.system("mv %s/mu_sigma.npy %s/../changed/%d_mu_sigma.npy" % (d, d, i))
else:
    print("=> original %s" % args.path)
    evaluator.calculate_statistics_given_path(args.path)
    os.system("mv %s/mu_sigma.npy %s/../original_mu_sigma.npy" % (args.path, args.path))
    print("=> changed %s" % args.path)
    model.load_state_dict(torch.load("pretrained/inception_v3_google.pth"))
    evaluator.model = model
    evaluator.calculate_statistics_given_path(args.path)
    os.system("mv %s/mu_sigma.npy %s/../changed_mu_sigma.npy" % (args.path, args.path))
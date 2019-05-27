import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, ".")
import os, argparse
import numpy as np
from lib.fid import fid_score

"""
parser = argparse.ArgumentParser()
parser.add_argument("--type", default=0, type=int, help="The path to pytorch inceptionv3 weight. You can obtain this by torchvision incetion_v3 function.")
args = parser.parse_args()
"""

types = ["original", "changed", "tf"]

def calc_fid_given_type(t):
    fids = []
    ref_path = "/home/xujianjing/data/cifar10_image/%s_mu_sigma.npy" % t
    tar_path = "/home/xujianjing/LBSGAN/logs/cifar_bs128/%s/%d_mu_sigma.npy"

    ref = np.load(ref_path).tolist()
    ref_mu, ref_sigma = ref['mu'], ref['sigma']
    for i in range(1, 200, 5):
        d = tar_path % (t, i)
        tar = np.load(d).tolist()
        tar_mu, tar_sigma = tar['mu'], tar['sigma']
        print("=> calc fid %s" % d)
        fid = fid_score.calculate_frechet_distance(ref_mu, ref_sigma, tar_mu, tar_sigma)
        print("Epoch %03d\t\t%.3f" % (i, fid))
        fids.append(fid)
    
    return fids

for t in types:
    fids = calc_fid_given_type(t)
    plt.plot(fids)
plt.legend(types)
plt.savefig("fids.png")
plt.close()
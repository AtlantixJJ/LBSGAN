"""
Extract image from cifar10 test to images.
"""
import sys, os
sys.path.insert(0, ".")
import argparse, tqdm
from PIL import Image
import lib

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="/home/atlantix/data/celeba", type=str, help="path to celeba dataset")
parser.add_argument("--size", default=64, type=int, help="image size: 64 or 128")
args = parser.parse_args()

data_path = args.path + "/img_align_celeba.zip"
out_dir = args.path + ("/img_align_celeba%d/" % args.size)
os.system("mkdir %s" % out_dir)

test_set = lib.dataset.TFCelebADataset(data_path, args.size, train=False)
dl = lib.dataset.TFDataloader(test_set, 256)
dl.reset()
cmd = out_dir + "%06d.jpg"
prev = 0
for sample in tqdm.tqdm(dl):
    if type(sample) is tuple and sample[0] is -1: break
    sample = ((sample.transpose(0, 2, 3, 1) + 1) * 127.5).astype("uint8")
    for i in range(sample.shape[0]):
        lib.utils.save_npy_image(cmd % (prev + i), sample[i])
    prev += sample.shape[0]

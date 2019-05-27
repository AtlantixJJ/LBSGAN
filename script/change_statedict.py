"""
Load a Pytorch InceptionV3 weight file and store Tensorflow InceptionV3 weight into it.
Note that only weight before pool3 is changed.
"""
import os, h5py, argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", help="The path to pytorch inceptionv3 weight. You can obtain this by torchvision incetion_v3 function.")
parser.add_argument("--save_path", help="The path to save new weight.")
args = parser.parse_args()

# the inception weight list
f = open("script/namelist.txt").readlines()
hnames = [i.strip().replace("/","_") for i in f]
# the weight of tensorflow inception v3 should be in dump directory
hfiles = [h5py.File("dump/" + h + ".h5") for h in hnames]
tweights = []
tnames = []
keys = ["weights", "gamma", "beta", "mean", "var"]
outkeys = ["weights", "weights", "bias", "running_mean", "running_var"]
for i, h in enumerate(hfiles):
  if "pool" in h.filename:
    continue
  for v in h.values(): print(v)
  for k,ok in zip(keys, outkeys):
    tweights.append(h[k])
    tnames.append(hnames[i] + "_" + ok)

dic = torch.load(args.load_path)

# do not load AuxLogits branch
dickeys = list(dic.keys())
for k in dickeys:
  if "Aux" in k:
    print(k)
    del dic[k]
# do not include fc weight
dickeys = list(dic.keys())[:-2]

for i in range(len(dickeys)):
  pk = dickeys[i]
  tk = tnames[i]
  ps = tuple(dic[pk].shape)
  ts = tweights[i].shape
  if len(ts) == 4:
    tweights[i] = tweights[i].value.transpose(3, 2, 0, 1)
    ts = tweights[i].shape
  else:
    tweights[i] = tweights[i].value
  if ts != ps:
    print("!> Mismatch")
  dic[pk].data = torch.from_numpy(tweights[i])
  print("[%03d] PT:%s\t%s\tTF:%s\t%s" % (i, pk, tk, str(ps), str(ts)))

torch.save(dic, args.save_path)
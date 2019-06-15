import os
import argparse
from subprocess import Popen
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="celeba64", type=str, help="celeba64,celeba128,cifar10")
parser.add_argument("--bs", default="1024", type=str, help="batch size")
parser.add_argument("--dbs", default=-1, type=int, help="delayed batch size")
parser.add_argument("--gpu", default="0", type=str, help="gpu index")
parser.add_argument("--seed", default="1,5,7", type=str, help="repeat expr seed")
args = parser.parse_args()

seed = [int(s) for s in args.seed.split(",")]
os.system("which python")
bs = [int(b) for b in args.bs.split(",") if len(b) > 0]

for s in seed:
  for b in bs:
    cmd = "python gan.py --dataset %s --batch_size %d --seed %d --gpu %s" % (args.dataset, b, s, args.gpu)
    if args.dbs > 0:
      cmd += " --delayed_batch_size %d" % args.dbs
    print(cmd)
    os.system(cmd)
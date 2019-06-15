"""
Extract image from cifar10 test to images.
"""
import tqdm
import argparse, torch, torchvision, os, sys
from torchvision import transforms
sys.path.insert(0, ".")
from lib import utils

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default="CIFAR10", type=str, help="CIFAR10|CIFAR100|MNIST")
parser.add_argument("--train", default=False, type=bool, help="is train")
parser.add_argument("--path", default="", type=str, help="path to cifar10 dataset")
parser.add_argument("--output", default="", type=str, help="path to extract path")
args = parser.parse_args()

if args.dataset_name == 'MNIST':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
else:
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

ds = getattr(torchvision.datasets, args.dataset_name)
dataset = ds(args.path, train=args.train, download=True, transform=transform_test)
dl = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False)

os.system("mkdir %s" % args.output)

for i, (input, target) in enumerate(tqdm.tqdm(dl)):
    utils.save_4dtensor_image(args.output + "/%05d.jpg", i * input.size(0), (input + 1) * 127.5)
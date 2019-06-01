"""
Extract image from cifar10 test to images.
"""
import argparse, torch, torchvision, os, sys
from torchvision import transforms
sys.path.insert(0, ".")
from lib import utils

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="", type=str, help="path to cifar10 dataset")
parser.add_argument("--output", default="", type=str, help="path to extract path")
args = parser.parse_args()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

ds = torchvision.datasets.cifar.CIFAR10
test_set = ds(args.path, train=False, download=True, transform=transform_test)
dl = torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=False, pin_memory=True)

os.system("mkdir %s/cifar10_image" % args.path)

for i, (input, target) in enumerate(dl):
    utils.save_4dtensor_image(args.path + "/cifar10_image/%05d.jpg", i * input.size(0), (input + 1) * 127.5)
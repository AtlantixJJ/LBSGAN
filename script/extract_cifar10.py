"""
Extract image from cifar10 test to images.
"""
import torch, torchvision, os, sys
from torchvision import transforms
sys.path.insert(0, ".")
from lib import utils

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

ds = torchvision.datasets.cifar.CIFAR10
test_set = ds("/home/xujianjing/data/", train=False, download=True, transform=transform_test)

dl = torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=False, pin_memory=True)

os.system("mkdir cifar10_test")

for i, (input, target) in enumerate(dl):
    print(input.shape, input.max(), input.min())
    utils.save_4dtensor_image("cifar10_test/%05d.jpg", i * input.size(0), (input + 1) * 127.5)
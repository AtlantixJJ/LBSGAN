import lib
import time
import torch
import torchvision.transforms as transforms

"""
ds = lib.dataset.FileDataset("/home/atlantix/data/celeba/img_align_celeba.zip",
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    )
dl = torch.utils.data.DataLoader(
        ds,
        batch_size=128,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
"""

ds = lib.tfdl.FileDataset("/home/atlantix/data/celeba/img_align_celeba.zip")
dl = lib.tfdl.TFDataloader(ds, 512, len(ds))

start_time = time.localtime()
for i, input in enumerate(dl):
    end_time = time.localtime()
    elapsed = time.mktime(end_time) - time.mktime(start_time)
    print("=> %d\t%f" % (i+1, elapsed / (i + 1)))
    if i > 200: break
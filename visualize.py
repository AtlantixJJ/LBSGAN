import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch, argparse
import torchvision.utils as vutils
from moviepy.editor import VideoClip
import numpy as np
import lib, models
from lib.manipulate import *

parser = argparse.ArgumentParser(description='SGD/SWA training')

parser.add_argument('--ckpt', type=str, default="expr/cifar10_feat_gbn_dbn.pt", help='check point 1')
parser.add_argument('--upsample', type=int, default=3)
parser.add_argument('--seed', type=int, default=3)
args = parser.parse_args()

rng = np.random.RandomState(args.seed)
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
ckpt = torch.load(args.ckpt1)
gen_model = models.simple.ConvolutionGenerator(bn='bn', upsample=args.upsample)
disc_model = models.simple.ConvolutionDiscriminator(bn='bn', upsample=args.upsample)
interpolate_model = models.simple.ConvolutionGenerator(bn='bn', upsample=args.upsample)
interpolate_model.eval()
interpolate_model.cuda()
gen_model.load_state_dict(ckpt['gen_state_dict'])
disc_model.load_state_dict(ckpt['disc_state_dict'])
gen_model.eval(); disc_model.eval()
gen_model.cuda(); disc_model.cuda()

#print_parameters(gen_model)
#print_parameters(disc_model)

fixed_z = torch.Tensor(16, 128).normal_().cuda()
fixed_z.requires_grad = True

def generate(model, z):
    result = model(z).detach()
    grid = vutils.make_grid(result, nrow=4, padding=2, normalize=True)
    img = grid.cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).astype("uint8")
    return img, result

loss_record = {"d1":[], "d2":[]}
loss_records = [{"d1":[], "d2":[]}] * 100

Image.fromarray(generate(gen_model, fixed_z)[0]).save(open("my_1.jpg", "wb"), format="JPEG")
Image.fromarray(generate(gen_model, fixed_z)[0]).save(open("my_2.jpg", "wb"), format="JPEG")

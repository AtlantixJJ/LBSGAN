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

parser.add_argument('--ckpt1', type=str, default="expr/cifar10_feat_gbn_dbn.pt", help='check point 1')
parser.add_argument('--ckpt2', type=str, default='expr/cifar10_none_gbn_dbn.pt', help='check point 2')
parser.add_argument('--upsample', type=int, default=3)
args = parser.parse_args()

ckpts = [torch.load(args.ckpt1), torch.load(args.ckpt2)]
gen_models = [
    models.simple.ConvolutionGenerator(bn='bn', upsample=args.upsample),
    models.simple.ConvolutionGenerator(bn='bn', upsample=args.upsample)]
disc_models = [
    models.simple.ConvolutionDiscriminator(bn='bn', upsample=args.upsample),
    models.simple.ConvolutionDiscriminator(bn='bn', upsample=args.upsample)]

interpolate_model = models.simple.ConvolutionGenerator(bn='bn', upsample=args.upsample)
interpolate_model.eval()
interpolate_model.cuda()
for ckpt, g, d in zip(ckpts, gen_models, disc_models):
    g.load_state_dict(ckpt['gen_state_dict'])
    d.load_state_dict(ckpt['disc_state_dict'])
    g.eval(); d.eval()
    g.cuda(); d.cuda()

print_parameters(gen_models[0])
print_parameters(gen_models[1])

fixed_z = torch.Tensor(16, 128).normal_().cuda()

TOTAL_TIME = 5.0

def test_disc_line(alpha):
    interpolate_parameters(gen_models[0], gen_models[1], interpolate_model, alpha)
    for i in range(100):
        z = torch.Tensor(128, 128).normal_().cuda()
        img = interpolate_model(z).detach()
        d1 = disc_models[0](img).detach().mean().cpu().numpy()
        d2 = disc_models[1](img).detach().mean().cpu().numpy()
        loss_records[i]['d1'].append(d1)
        loss_records[i]['d2'].append(d2)

def generate(model, z):
    result = model(z).detach()
    grid = vutils.make_grid(result, nrow=4, padding=2, normalize=True)
    img = grid.cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).astype("uint8")
    return img, result

loss_record = {"d1":[], "d2":[]}
loss_records = [{"d1":[], "d2":[]}] * 100

def interpolate_make_frame(t):
    process = t / TOTAL_TIME
    interpolate_parameters(gen_models[0], gen_models[1], interpolate_model, process)
    img, raw_img = generate(interpolate_model, fixed_z)
    disc1, disc2 = disc_models[0](raw_img), disc_models[1](raw_img)
    loss_record["d1"].append(disc1.detach().cpu().numpy().mean())
    loss_record["d2"].append(disc2.detach().cpu().numpy().mean())
    return img

Image.fromarray(generate(gen_models[0], fixed_z)[0]).save(open("my_1.jpg", "wb"), format="JPEG")
Image.fromarray(generate(gen_models[1], fixed_z)[0]).save(open("my_2.jpg", "wb"), format="JPEG")

#interpolate_parameters(gen_models[0], gen_models[1], interpolate_model, 0.5)

animation = VideoClip(interpolate_make_frame, duration=TOTAL_TIME) # 3-second clip
animation.write_videofile("my_animation.mp4", fps=24)

for i in range(100):
    print("=> %d" % i)
    alpha = i / 100.
    test_disc_line(alpha)
for i in range(10):
    plot_dict(loss_records[i])
plt.savefig("my_fig.png")
plt.close()
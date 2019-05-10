try:
    import matplotlib.pyplot as plt
except:
    print("On philly")
    
from PIL import Image
import torch, argparse
import torchvision.utils as vutils
import numpy as np

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)    

def plot_dict(dic, **kwargs):
    for i, k in enumerate(dic.keys()):
        c_ind = (i * 2 + 7) % len(tableau20)
        plt.plot(dic[k], color=tableau20[c_ind], **kwargs)
    plt.legend(list(dic.keys()))

def interpolate_running_mean(model1, model2, model_f, alpha=0.5):
    for m1, m2, mf in zip(model1.children(), model2.children(), model_f.children()):
        if issubclass(m1.__class__, torch.nn.modules.batchnorm._BatchNorm):
            mf.running_mean = m1.running_mean * (1 - alpha) + m2.running_mean * alpha
            ## Var(aX+bY) = a^2 Var(X) + b^2 Var(Y) + 2ab Cov(X, Y)
            mf.running_var = m1.running_var * (1 - alpha) ** 2 + m2.running_var * alpha ** 2
            #mf.running_var = m1.running_var * (1 - alpha) + m2.running_var * alpha
            #mf.running_mean = m1.running_mean
            #mf.running_var = m1.running_var
        else:
            interpolate_running_mean(m1, m2, mf, alpha)

def interpolate_parameters(model1, model2, model_f, alpha=0.5):
    for p1, p2, pf in zip(model1.parameters(), model2.parameters(), model_f.parameters()):
        pf.data = p1.data * (1 - alpha) + p2.data * alpha
    
    interpolate_running_mean(model1, model2, model_f, alpha)


def print_parameters(model):
    for n, p in model.named_parameters():
        maxi = p.max().detach().cpu().numpy()
        mini = p.min().detach().cpu().numpy()
        print(n, list(p.size()), maxi, mini)
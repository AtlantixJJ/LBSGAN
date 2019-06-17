"""
Plot mnist fid plot.
"""
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

names = ['cifar', 'mnist']
data_dir = 'data/caffe_fid/'
def parse(dic, name):
    with open(name, "r") as f:
        lines = f.readlines()
    for l in lines:
        name, step, val = l.split(" ")
        name = name.replace('cifar_pool_', '').replace('mnist_pool_', '')
        val = float(val)
        try:
            dic[name].append(val)
        except:
            dic[name] = [val]
    return dic

for name in names:
    dic = {}
    parse(dic, data_dir + name + '_fid.txt')
    for k,v in dic.items():
        plt.plot([2000 * i for i in range(1, 11)], v, 'x-')
    plt.legend(dic.keys())
    plt.savefig(name + ".png")
    plt.close()
    print(dic)
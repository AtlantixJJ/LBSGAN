"""
Plot the training log.
"""
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

bss = [32, 64, 256, 512, 1024, 2048]
name = ['cifar10_bs%d' % d for d in bss]
log_dir = sys.argv[1]

def plot_repeat(ax, name):
    #name = 'cifar10_bs256'
    repeats = ['_seed1', '_seed5', '_seed7']
    st_ind = 0

    fids = []
    x = []
    for r_name in repeats:
        expr_dir = log_dir + name + r_name
        try:
            result = np.load(expr_dir + "/result.npy").tolist()
        except:
            continue
        fid = np.array(result['eval_fid'])[:, 1]
        if fid.shape[0] < 200: continue
        x = np.array(result['eval_fid'])[:, 0]
        fids.append(np.array(fid))
    fids = np.array(fids)
    print(fids.shape)
    mean_fid = fids.mean(0)
    min_fid = fids.min(0)
    max_fid = fids.max(0)
    x = x[st_ind:]
    mean_fid = mean_fid[st_ind:]
    min_fid = min_fid[st_ind:]
    max_fid = max_fid[st_ind:]
    ax.fill_between(x, min_fid, mean_fid, color='orange')
    ax.plot(x, mean_fid, color='red')
    ax.fill_between(x, mean_fid, max_fid, color='orange')

    print("Min fid: %f" % min_fid.min())

fig = plt.figure(figsize=(8, 7))
for i, n in enumerate(name):
    print("=> %s" % n)
    ax = plt.subplot(2, 3, 1 + i)
    ax.set_ylim(ymin=20, ymax=100)
    ax.set_title("%d" % bss[i])
    plot_repeat(ax, n)
plt.savefig("all.png")
plt.close()
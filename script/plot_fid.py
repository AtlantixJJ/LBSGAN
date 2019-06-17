"""
Plot the training log.
"""
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

bss = [32, 64, 256, 512, 1024, 2048, 4096, 8192]
NROW = 4; NCOL = 2
YMIN = 10; YMAX = 160
expr_name = sys.argv[2] #cifar10, celeba64
name = ['%s_bs%d' % (expr_name, d) for d in bss]
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

fig, axes = plt.subplots(nrows=NROW, ncols=NCOL, figsize=(6, 16))
fig.tight_layout()
plt.subplots_adjust(left=0.1, top=0.96, bottom=0.11, hspace=0.3)
for i, n in enumerate(name):
    print("=> %s" % n)
    ax = axes[i // NCOL, i % NCOL]
    ax.set_ylim(ymin=YMIN, ymax=YMAX)
    ax.set_title("batchsize=%d" % bss[i])
    plot_repeat(ax, n)
plt.savefig("all.png")
plt.close()
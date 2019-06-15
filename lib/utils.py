from PIL import Image
import torch

def save_npy_image(fpath, nparr):
    """
    nparr: should be 3 channel np.uint8 type
    """
    with open(fpath, "wb") as f:
        Image.fromarray(nparr).save(f, format="JPEG")

def save_3dtensor_image(fpath, t):
    """
    t: (channel, H, W)
    """
    save_npy_image(fpath, t.detach().cpu().numpy().astype("uint8").transpose(1, 2, 0))

def save_4dtensor_image(name_format, idx, t):
    """
    Args:
    t: (N, C, H, W)
    format: path/xxx_%d.jpg
    idx: start index
    """
    t_arr = t.detach().cpu().numpy().astype("uint8").transpose(0, 2, 3, 1)
    if t_arr.shape[-1] == 1:
        t_arr = t_arr[:, :, :, 0]
    for i in range(t_arr.shape[0]):
        save_npy_image(name_format % (idx + i), t_arr[i])

def is_image(name):
    exts = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    for ext in exts:
        if ext in name:
            return True
    return False
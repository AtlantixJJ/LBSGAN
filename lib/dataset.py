import tensorflow as tf
import torch
from io import open, BytesIO
import os, time, sys, zipfile
import numpy as np
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

def is_image(name):
    exts = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    for ext in exts:
        if ext in name:
            return True
    return False

class TFDataloader():
    def __init__(self, dataset, batch_size):
        """
        A workround need to specify num_iter
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iter = len(self.dataset) // self.batch_size
        self.dataset = dataset.dataset.shuffle(buffer_size=1000).batch(batch_size)
        self.iterator = self.dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()

        t_ = os.environ['CUDA_VISIBLE_DEVICES']
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
        self.sess = tf.Session()
        os.environ['CUDA_VISIBLE_DEVICES'] = t_
    
    def reset(self):
        self.sess.run(self.iterator.initializer)

    def __getitem__(self, idx):
        return [torch.Tensor(t) for t in self.sess.run(self.next_element)]
    
    def __len__(self):
        return self.num_iter

class TFFileDataset():
    def __init__(self, data_path, img_size=64, npy_dir=None):
        self.img_size = (img_size, img_size)

        if ".zip" in data_path:
            self.use_zip = True
            self.data_file = zipfile.ZipFile(data_path)
            self.files = self.data_file.namelist()
            self.files.sort()
            self.files = self.files[1:]
        else:
            self.use_zip = False
            self.files = sum([[file for file in files] for path, dirs, files in os.walk(data_path) if files], [])
            self.files.sort()

        self.idxs = np.arange(len(self.files))
        self.rng = np.random.RandomState(1)
        self.rng.shuffle

        # 图片文件的列表
        filelist_t = tf.constant(self.files)
        self.file_num = len(self.files)

        # label
        if npy_dir is not None:
            label = np.load(npy_dir)
            label_t = tf.constant(label)
            self.class_num = label.shape[-1]
            dataset = tf.data.Dataset.from_tensor_slices((filelist_t, label_t))
        else:
            self.class_num = -1
            dataset = tf.data.Dataset.from_tensor_slices((filelist_t, tf.constant(np.zeros((self.file_num,)))))
        
        self.dataset = dataset.map(self._parse_function)

    def __len__(self):
        return len(self.files)

    def read_image_from_zip(self, filename):
        """
        An eagar function for reading image from zip
        """
        f = filename.numpy().decode("utf-8")
        return np.asarray(Image.open(BytesIO(self.data_file.read(f))))

    def _parse_function(self, filename, label):
        if self.use_zip:
            x = tf.py_function(self.read_image_from_zip, [filename], tf.float32)
        else:
            x = tf.read_file(filename)
            x = tf.image.decode_image(x)
        
        x = tf.expand_dims(x, 0)
        x = tf.image.resize_bilinear(x, (self.img_size[0], self.img_size[1]))
        x = x[0]
        x = tf.cast(x, tf.float32) / 255.0
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.9, 1.1)
        x = tf.image.random_flip_left_right(x)
        x = tf.clip_by_value(x * 2 - 1, -1.0, 1.0)
        x = tf.transpose(x, (2, 0, 1)) # (H, W, C) => (C, H, W)

        if self.class_num > 0:
            return x, label
        else:
            return x

class TFCelebADataset(TFFileDataset):
    def __init__(self, data_path, img_size=64, npy_dir=None):
        super(TFCelebADataset, self).__init__(data_path, img_size, npy_dir)

    def access(self, idx):
        """
        Deprecated
        """
        if self.use_zip:
            img = np.asarray(Image.open(BytesIO(self.data_file.read(self.files[idx]))))
        else:
            img_path = os.path.join(self.data_path, self.files[idx])
            img = np.asarray(Image.open(open(img_path, "rb")))

        img = img[50:50+128, 25:25+128]

        return self.transform(img)

    # 函数将filename对应的图片文件读进来
    def _parse_function(self, filename, label):
        if self.use_zip:
            x = tf.py_function(self.read_image_from_zip, [filename], tf.float32)
        else:
            x = tf.read_file(filename)
            x = tf.image.decode_image(x)
        
        x = tf.image.crop_to_bounding_box(x, 50, 25, 128, 128)
        x = tf.expand_dims(x, 0)
        x = tf.image.resize_bilinear(x, (self.img_size[0], self.img_size[1]))
        x = x[0]
        x = tf.cast(x, tf.float32) / 255.0
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.9, 1.1)
        x = tf.image.random_flip_left_right(x)
        x = tf.clip_by_value(x * 2 - 1, -1.0, 1.0)
        x = tf.transpose(x, (2, 0, 1)) # (H, W, C) => (C, H, W)
        return x, label

    def read_label(self):
        """
        For convert txt label to npy label
        """
        self.label = []
        with open(self.attr_file) as f:
            self.label_len = int(f.readline())
            self.label_name = f.readline().strip().split(" ")
            self.class_num = len(self.label_name)
            for l in f.readlines():
                l = l.strip().replace("  ", " ").split(" ")
                l = [int(i) for i in l[1:]]
                self.label.append(np.array(l))
        self.label = np.array(self.label)
        self.label[self.label==-1] = 0
        np.save(self.attr_file.replace(".txt", ""), self.label)

class FileDataset(torch.utils.data.Dataset):
    """
    Currently label is not available
    """
    def __init__(self, data_path, transform=None, **kwargs):
        self.data_path = data_path
        self.transform = transform

        if ".zip" in data_path:
            self.use_zip = True
            self.data_file = zipfile.ZipFile(data_path)
            self.files = self.data_file.namelist()
            self.files.sort()
            self.files = self.files[1:]
        else:
            self.use_zip = False
            self.files = sum([[file for file in files] for path, dirs, files in os.walk(data_path) if files], [])
            self.files.sort()
        self.idxs = np.arange(len(self.files))
        self.rng = np.random.RandomState(1)
        self.reset()
    
    def reset(self):
        self.rng.shuffle(self.idxs)

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        fpath = self.files[idx]

        if self.use_zip:
            img = Image.open(BytesIO(self.data_file.read(fpath)))
        else:
            with open(self.data_path + fpath, "rb") as f:
                img = Image.open(f).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = 0

        return (img, label)
    
    def __len__(self):
        return len(self.files)
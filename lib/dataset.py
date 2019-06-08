import os, time, sys, zipfile
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from io import open, BytesIO
import numpy as np
from PIL import Image
import lib

def pil_bilinear_interpolation(x, size=(299, 299)):
    """
    x: [-1, 1] torch tensor
    """
    y = np.zeros((x.shape[0], size[0], size[1], x.shape[1]), dtype='uint8')
    x_arr = ((x + 1) * 127.5).detach().cpu().numpy().astype("uint8")
    x_arr = x_arr.transpose(0, 2, 3, 1)
    for i in range(x_arr.shape[0]):
        y[i] = np.asarray(Image.fromarray(x_arr[i]).resize(size, Image.BILINEAR))
    return torch.from_numpy(y.transpose(0, 3, 1, 2)).type_as(x) / 127.5 - 1

def read_image_and_resize(filename, size):
    """
    An eagar function for reading image from zip
    """
    f = filename.numpy().decode("utf-8")
    return np.asarray(Image.open(open(f, "rb")).resize(size))

class GeneratorIterator(object):
    def __init__(self, model, tot_num=50000, batch_size=64, cuda=True):
        self.model = model
        self.tot_num = tot_num
        self.cuda = cuda
        self.batch_size = batch_size
        self.num_iter = self.tot_num // self.batch_size
    
    def iterator(self, save_path=None):
        if save_path is not None:
            os.system("mkdir %s" % save_path)
        z = torch.Tensor(self.batch_size, 128)
        if self.cuda: z = z.cuda()
        if self.num_iter * self.batch_size < self.tot_num:
            self.num_iter += 1
        for i in range(self.num_iter):
            if i == self.num_iter - 1:
                bs = self.tot_num - self.batch_size * i
                if bs < self.batch_size:
                    z = torch.Tensor(bs, 128)
                    if self.cuda: z = z.cuda()
            z = z.normal_() * 2
            t = self.model(z)
            if save_path is not None:
                lib.utils.save_4dtensor_image(
                    save_path + "/%05d.jpg",
                    i * self.batch_size,
                    (t + 1) * 127.5)

            yield pil_bilinear_interpolation(t)

class PytorchDataloader(object):
    def __init__(self, train_dl, test_dl, train):
        self.train = train
        self.train_dl = train_dl
        self.test_dl = test_dl
    
    def reset(self):
        pass
    
    def __iter__(self):
        if self.train:
            return self.train_dl.__iter__()
        else:
            return self.test_dl.__iter__()

    def __len__(self):
        if self.train:
            return len(self.train_dl)
        else:
            return len(self.test_dl)

class TFDataloader():
    def __init__(self, dataset, batch_size):
        """
        A workround need to specify num_iter
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iter = len(self.dataset) // self.batch_size - 1
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
        try:
            item = self.sess.run(self.next_element)
            if type(item) is tuple:
                return [torch.Tensor(t) for t in item]
            else:
                return item
        except tf.errors.OutOfRangeError:
            print("=> TFDataloader out of range")
            return (-1, -1)
    
    def __len__(self):
        return self.num_iter

class TFFileDataset():
    def __init__(self, data_path, img_size=64, npy_dir=None, train=True, seed=1):
        self.img_size = (img_size, img_size)
        self.train = train

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
        self.rng = np.random.RandomState(seed)
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
        img = Image.open(BytesIO(self.data_file.read(f)))
        return np.asarray(img)

    def _parse_function(self, filename, label):
        if self.use_zip:
            x = tf.py_function(self.read_image_resize_from_zip, [filename, self.img_size], tf.float32)
        else:
            x = tf.py_function(read_image_resize, [filename, self.img_size], tf.float32)
        
        x = tf.expand_dims(x, 0)
        #x = tf.image.resize_bilinear(x, (self.img_size[0], self.img_size[1]))
        x = x[0]
        x = tf.cast(x, tf.float32) / 255.0
        if self.train:
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
    def __init__(self, data_path, img_size=64, npy_dir=None, train=True, seed=1):
        super(TFCelebADataset, self).__init__(data_path, img_size, npy_dir, seed)
        self.train = train

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
            #x = tf.py_function(read_image_resize, [filename, self.img_size], tf.float32)
            x = tf.read_file(filename)
            x = tf.image.decode_image(x)
        
        x = tf.image.crop_to_bounding_box(x, 50, 25, 128, 128)
        x = tf.expand_dims(x, 0)
        #TF bilinear resize is not correct
        x = tf.image.resize_bilinear(x, (self.img_size[0], self.img_size[1]))
        x = x[0]
        x = tf.cast(x, tf.float32) / 255.0
        if self.train:
            x = tf.image.random_brightness(x, 0.05)
            x = tf.image.random_contrast(x, 0.9, 1.1)
            x = tf.image.random_flip_left_right(x)
        x = tf.clip_by_value(x * 2 - 1, -1.0, 1.0)
        x = tf.transpose(x, (2, 0, 1)) # (H, W, C) => (C, H, W)
        if self.class_num > 0:
            return x, label
        else:
            return x

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

class SimpleDataset(torch.utils.data.Dataset):
    """
    Currently label is not available
    """
    def __init__(self, data_path, size, transform=None):
        self.size = size
        self.data_path = data_path
        self.transform = transform

        self.files = sum([[file for file in files if ".jpg" in file or ".png" in file] for path, dirs, files in os.walk(data_path) if files], [])
        self.files.sort()

    def __getitem__(self, idx):
        fpath = self.files[idx]
        with open(os.path.join(self.data_path, fpath), "rb") as f:
            img = Image.open(f).convert("RGB").resize(self.size, Image.BILINEAR)
        if self.transform:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.files)

class FileDataset(torch.utils.data.Dataset):
    """
    Currently label is not available
    """
    def __init__(self, data_path, transform=None):
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
        img = img.resize((299, 299))
        if self.transform:
            img = self.transform(img)

        label = 0

        return (img, label)
    
    def __len__(self):
        return len(self.files)
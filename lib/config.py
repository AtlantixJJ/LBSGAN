import argparse, os, sys
import torch, torchvision
from PIL import Image
from torchvision import transforms
import lib, models

class BaseConfig(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='large batch size GAN training')
        self.parser.add_argument('--model', type=str, default="simple", metavar='MODEL',
                            help='model name (default: None)')
        self.parser.add_argument('--log_dir', type=str, default="logs", help='training directory (default: None)')
        self.parser.add_argument('--dataset', type=str, default='cifar', help='dataset name (default: cifar)')
        self.parser.add_argument('--data_dir', type=str, default="/home/atlantix/data/", metavar='PATH',
                            help='path to datasets location (default: None)')
        self.parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
        self.parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')

        self.parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                            help='checkpoint to resume training from (default: None)')

        self.parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
        self.parser.add_argument('--summary_interval', type=int, default=1000, metavar='N', help='summary iteration (default: 1000')
        self.parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
        self.parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='eval frequency (default: 5)')
        self.parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.01)')
        self.parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
        self.parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')

        self.parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    def parse(self):
        self.args = self.parser.parse_args()

        if not os.path.isdir(self.args.data_dir):
            newdir = "/home/xujianjing/data/"
            print("=> Data dir %s not exist, switch to %s" % (self.args.data_dir, newdir))
            self.args.data_dir = newdir

        torch.backends.cudnn.benchmark = True
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)

        # preparing datasaet
        self.dataset = self.args.dataset
        self.data_dir = self.args.data_dir

        if 'cifar' in self.dataset:
            self.data_path = self.data_dir
            self.upsample = 3
            ds = torchvision.datasets.cifar.CIFAR10
            self.gen_function = models.simple.UpsampleGenerator
            self.disc_function = models.simple.DownsampleDiscriminator

            self.transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            self.train_set = ds(self.data_path, train=True, download=True, transform=self.transform_train)
            #self.test_set = ds(self.data_path, train=False, download=True, transform=self.transform_test)
            self.dl = torch.utils.data.DataLoader(
                    self.train_set,
                    batch_size=self.args.batch_size,
                    shuffle=True,
                    num_workers=self.args.num_workers,
                    pin_memory=True)

        elif 'celeba' in self.dataset:
            self.data_path = self.data_dir + "celeba/img_align_celeba.zip"

            if '64' in self.dataset:
                self.upsample = 4
                self.imgsize = 64
            elif '128' in self.dataset:
                self.upsample = 5
                self.imgsize = 128
            self.args.num_workers = 1
            ds = lib.dataset.TFCelebADataset
            self.gen_function = models.simple.UpsampleGenerator
            self.disc_function = models.simple.DownsampleDiscriminator
            #self.gen_function = models.resnet.ResNetGen
            #self.disc_function = models.resnet.ResNetDisc

            self.train_set = ds(self.data_path)
            self.dl = lib.dataset.TFDataloader(self.train_set, self.args.batch_size)

        # prepare directory
        self.name = self.dataset + "_" + str(self.args.batch_size)
        self.log_dir = self.args.log_dir + '/' + self.name
        self.model_dir = self.log_dir

        print('=> Preparing directory %s' % self.log_dir)
        print('=> Model directory %s' % self.model_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, 'command.sh'), 'w') as f:
            f.write(' '.join(sys.argv))
            f.write('\n')

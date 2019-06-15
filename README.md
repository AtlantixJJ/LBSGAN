# Large Batch Size Training of GANs

Large batch size training is an important field of study. It helps with scaling up NNs to super computer scale with simple data parallel.

But LBS training dynamics is much different than traditional medium sized batch training. Some works in training ImageNet in minutes have already proposed several modifications, such as Layerwise Adaptive Rate Scaling.

But few studies have focused on LBS training of GAN. In fact, bigGAN achieves its best performance with 4K batch size. This is not a common choice of batch size in GAN as well as other fields. It seems that LBS in GAN worth extra research.

This is what this project aims at.
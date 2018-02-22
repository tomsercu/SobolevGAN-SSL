Semi-Supervised Learning with Sobolev GAN
=========================================

Code accompanying the papers:
+ [Sobolev GAN](https://openreview.net/forum?id=SJA7xfb0b) ([arXiv](https://arxiv.org/abs/1711.04894)) - Appeared at ICLR 2018.
+ [Semi-Supervised Learning with IPM-based GANs: an Empirical Study](https://arxiv.org/abs/1712.02505) - Appeared at NIPS 2017 Workshop: Deep Learning: Bridging Theory and Practice

Tested for python 2.7, [PyTorch](http://pytorch.org) 0.3.0.

To reproduce the CIFAR-10 result for 4000 labeled samples using the `K+1` critic, imposing Fisher constraint on full critic `f` and Sobolev constraint on `f-` only:
```
python main.py --dataset cifar10 --dataroot ${DATAROOT} --cuda --outputdir ${OUTPUTDIR} \
--labeledSamples 4000 \
--SSL_critic_type Kp1 --f_component_Fisher f --f_component_Sobolev fneg
```

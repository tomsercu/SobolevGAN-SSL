import torch
import torch.nn as nn
from . import dcgan

class D(dcgan.D):
    """ Discriminator in the flavor of https://arxiv.org/abs/1606.03498 """
    def init_main(self, isize, nz, nc, ndf, n_extra_layers):
        assert isize == 32, "specific for 32x32"
        main = nn.Sequential(
            nn.Dropout(0.2),
            # stage1: 96 x 32 x 32
            nn.Conv2d(nc, 96, 3, stride=1, padding=1, bias=True), # bias cause no BN
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1, bias=False),
            self.XNorm(96, 32, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(96, 96, 3, stride=2, padding=1, bias=False),
            self.XNorm(96, 16, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # stage2: 192 x 16 x 16
            nn.Conv2d(96, 192, 3, stride=1, padding=1, bias=False),
            self.XNorm(192, 16, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(192, 192, 3, stride=1, padding=1, bias=False),
            self.XNorm(192, 16, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(192, 192, 3, stride=2, padding=1, bias=False),
            self.XNorm(192, 8, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # stage3: 192 x 8 x 8 to 6x6 to 4x4 then 1x1
            nn.Conv2d(192, 384, 3, stride=1, padding=0, bias=False),
            self.XNorm(384, 6, 6),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(384, 384, 3, stride=1, padding=0, bias=False),
            self.XNorm(384, 4, 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(384, 384, 1, stride=1, padding=0, bias=False),
            self.XNorm(384, 4, 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
        )
        self.main = main
        self.PhiD = 384*4*4

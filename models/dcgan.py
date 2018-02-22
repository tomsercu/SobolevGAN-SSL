import torch
import torch.nn as nn
import torch.nn.parallel

class Identity(nn.Module):
    def forward(self, input):
        return input
    def __repr__(self):
        return 'Identity'

class D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, n_classes=0, normalization=None):
        super(D, self).__init__()
        self.ngpu = ngpu
        self.normalization = normalization
        self.init_main(isize, nz, nc, ndf, n_extra_layers)
        self.V = nn.Linear(self.PhiD, 1, bias=False)
        if n_classes > 0:
            self.S_dot = nn.Linear(self.PhiD, n_classes, bias=True)

    def XNorm(self, numChannels=None, H=None, W=None):
        if self.normalization == 'batchnorm':
            assert numChannels, "Need to provide numChannels for BatchNorm2d"
            return nn.BatchNorm2d(numChannels)
        elif self.normalization == 'none' or self.normalization is None:
            return Identity()
        elif self.normalization == 'layernormI':
            # Standard Ishaan-Layernorm == efficient version of layernorm[1,2,3][1]
            from . import layernorm
            return layernorm.LayerNormI(numChannels)
        elif 'layernorm' in self.normalization:
            from . import layernorm
            import re
            # ex layernorm[1,2,3][1]  reduces over dims [1,2,3] and 
            # keeps (weight,bias) = (scale,bias) over first dim i.e. size 1 x numChannels x 1 x 1
            #reducedims, paramdims = re.findall('\[(.*?)\]', self.normalization)
            def parseListOfInts(str_comma_separated):
                if str_comma_separated:
                    return [int(i) for i in li.split(',')]
                else:
                    return [] # empty list
            reducedims, paramdims = [parseListOfInts(li) for li in re.findall('\[(.*?)\]', self.normalization)]
            paramSize = [v if i in paramdims else 1 for (i,v) in [(0,None), (1,numChannels), (2,H), (3,W)]]
            assert None not in paramSize, "Invalid paramSize {} from paramdims {}".format(paramSize, paramdims)
            return layernorm.LayerNorm(reducedims, paramSize)
        elif self.normalization == 'instancenorm':
            return nn.InstanceNorm2d(numChannels, affine=True)
        else:
            raise Exception('Unknown normalization: {}'.format(normalization))

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            phi_batch = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            phi_batch = self.main(input)
        phi_batch = phi_batch.view(input.size(0), self.PhiD)
        vphi = self.V(phi_batch) # bs x 1
        S_phi = self.S_dot(phi_batch) if hasattr(self, 'S_dot') else None # compute if available
        return vphi, S_phi
    
    def init_main(self, isize, nz, nc, ndf, n_extra_layers):
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        main = nn.Sequential(
            # input is nc x isize x isize
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        i, csize, cndf = 2, isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(str(i),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module(str(i+1),
                            self.XNorm(cndf, csize, csize))
            main.add_module(str(i+2),
                            nn.LeakyReLU(0.2, inplace=True))
            i += 3

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module(str(i),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module(str(i+1),
                            self.XNorm(out_feat, csize/2, csize/2))
            main.add_module(str(i+2),
                            nn.LeakyReLU(0.2, inplace=True))
            i+=3
            cndf = cndf * 2
            csize = csize / 2
        self.main = main
        # state size. cndf x 4 x 4 -> k_ncomponents
        self.PhiD = cndf * 16

class G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(G, self).__init__()
        self.ngpu = ngpu
        self.init_main(isize, nz, nc, ngf, n_extra_layers)

    def init_main(self, isize, nz, nc, ngf, n_extra_layers):
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(cngf),
            nn.ReLU(True),
        )

        i, csize, cndf = 3, 4, cngf
        while csize < isize//2:
            main.add_module(str(i),
                nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm2d(cngf//2))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(str(i),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm2d(cngf))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3

        main.add_module(str(i),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module(str(i+1), nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

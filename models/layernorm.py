import torch

# Similar to https://github.com/pytorch/pytorch/issues/1959 2nd comment.
class LayerNorm(torch.nn.Module):
    def __init__(self, reducedims, paramSize, eps=1e-5):
         #super(_InstanceNorm, self).__init__()
        super(LayerNorm, self).__init__()
        self.reducedims = reducedims
        self.paramSize  = paramSize
        self.gamma      = torch.nn.Parameter(torch.ones(paramSize))
        self.beta       = torch.nn.Parameter(torch.zeros(paramSize))
        self.eps        = eps

    def forward(self, x):
        m = x
        for d in self.reducedims[::-1]:
            m = m.mean(d, keepdim=True)
        var = (x-m)**2
        for d in self.reducedims[::-1]:
            var = var.mean(d, keepdim=True)
        return self.gamma * (x - m) / torch.sqrt(var + self.eps) + self.beta

    def __repr__(self):
        return 'LayerNorm Correct. Reduce [{}] params [{}] (eps={})'.format(
          ','.join(str(i) for i in self.reducedims), 'x'.join(str(i) for i in self.paramSize), self.eps)

class LayerNormMeanDetach(LayerNorm):
    def forward(self, x):
        m = x
        for d in self.reducedims[::-1]:
            m = m.mean(d, keepdim=True)
        var = (x-m.detach())**2 # NOTE detach mean for std calculation!
        for d in self.reducedims[::-1]:
            var = var.mean(d, keepdim=True)
        return self.gamma * (x - m) / torch.sqrt(var + self.eps) + self.beta

    def __repr__(self):
        return 'LayerNorm Mean Detach (broken). Reduce [{}] params [{}] (eps={})'.format(
          ','.join(str(i) for i in self.reducedims), 'x'.join(str(i) for i in self.paramSize), self.eps)

# Specialcased for imgs, similar to Ishaan https://github.com/igul222/improved_wgan_training/blob/4fde7f36d697576d06ee0046630064076c196ed6/tflib/ops/layernorm.py
# Meaning: reduce last 3 dims to singleton mean, var. gamma, beta over featuremaps.
class LayerNormI(torch.nn.Module):
    def __init__(self, numChannels, eps=1e-5):
        super(LayerNormI, self).__init__()
        self.numChannels = numChannels
        self.gamma       = torch.nn.Parameter(torch.ones (1, numChannels, 1, 1))
        self.beta        = torch.nn.Parameter(torch.zeros(1, numChannels, 1, 1))
        self.eps         = eps

    def forward(self, x):
        m   = x.view(x.size(0), -1).mean(1,keepdim=True)[:,:,None,None]
        var = x.view(x.size(0), -1).var (1,keepdim=True)[:,:,None,None]
        output = self.gamma * (x - m) / torch.sqrt(var + self.eps) + self.beta
        return output

    def __repr__(self):
        return 'LayerNormI reduce [1,2,3] params [1x{}x1x1] (eps={})'.format(self.numChannels, self.eps)

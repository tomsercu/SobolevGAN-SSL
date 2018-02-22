# Critic classes for 3 different flavors (K, K+1, K+1 plogp)
# Arguments (vphi, Sphi)
# Returns   (f, fpos, fneg)

import torch
import torch.nn as nn
import torch.nn.functional as F

class K(nn.Module):
    def forward(self, vphi, Sphi):
        # For K return vphi only as f, there should be no constraints specified
        # for fpos or fneg in this case (will fail).
        return vphi, None, None

class Kp1(nn.Module):
    def forward(self, vphi, S_phi):
        p_y = F.softmax(S_phi)
        fpos = (p_y * S_phi).sum(dim=1)
        fneg = vphi
        return fpos - fneg, fpos, fneg

class Kp1_plogp(nn.Module):
    def forward(self, vphi, S_phi):
        log_p_y = F.log_softmax(S_phi)
        p_y     = log_p_y.exp()
        fpos = (p_y * log_p_y).sum(dim=1)
        fneg = vphi
        return fpos - fneg, fpos, fneg

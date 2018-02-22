from __future__ import print_function
import argparse
import random, timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os, sys

import models
import critics

DEF_DATAROOT = '/parent/path/to/datasets'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | svhn ')
parser.add_argument('--dataroot', default='', help='path to dataset')
parser.add_argument('--outputdir', default='tmp', help='Where to store samples and models')
parser.add_argument('--manualSeed', type=int, default=0, help='Optionally specify to fix seed')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=350, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=2e-4, help='learning rate for Critic')
parser.add_argument('--lrDcut', type=int, default=0, help='cut lrD in half every X epochs')
parser.add_argument('--lrG', type=float, default=2e-4, help='learning rate for Generator')
parser.add_argument('--lrGcut', type=int, default=0, help='cut lrG in half every X epochs')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--wdecay', type=float, default=1e-6, help='wdecay value for Phi')
parser.add_argument('--wdecayV', type=float, default=0.001, help='wdecay value for V')
parser.add_argument('--wdecayS', type=float, default=1e-6, help='wdecay value for S')
parser.add_argument('--n_c', type=int, default=1, help='number of D iters per each G iter')
parser.add_argument('--hiStart_n_c'  , action='store_true', help='do many D iters at start')
parser.add_argument('--model_G', default='dcgan', help='models/submodule to use for G')
parser.add_argument('--model_D', default='openaistyle', help='models/submodule to use for D')
parser.add_argument('--G_extra_layers', type=int, default=2, help='Number of extra layers on gen and disc')
parser.add_argument('--D_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--normalization_D', default='none', help='none / batchnorm / layernorm / instancenorm')
parser.add_argument('--rmsprop', action='store_true', help='Whether to use rmsprop')
parser.add_argument('--sgd', action='store_true', help='Whether to use SGD')
parser.add_argument('--labeledSamples', type=int, default=4000, help='Add semi-sup learning objective, if >0')
parser.add_argument('--lambdaXE_D', type=float, default=1.5, help='Weight for XE term for D')
parser.add_argument('--conditionalG', action='store_true', help='Whether to condition G  on labels')
parser.add_argument('--lambdaXE_G', type=float, default=0.1, help='Weight for XE term for G')
parser.add_argument('--rhoFisher', type=float,  default=5e-8, help='penalty weight & lrate for fisher constr (E_mu[f^2] -1)**2')
parser.add_argument('--rhoSobolev', type=float, default=2e-8, help='penalty weight & lrate for sobolev constr')
parser.add_argument('--lambdaGP', type=float,   default=10.0,  help='WGAN-GP constraint weight factor.')
parser.add_argument('--SSL_critic_type',    default='Kp1', help='which critic for SSL: K, Kp1 or Kp1_plogp')
parser.add_argument('--f_component_Fisher',  default='', help='which part of critic to apply fisher constraint: f, fpos, fneg.')
parser.add_argument('--f_component_Sobolev', default='', help='which part of critic to apply sobolev constraint: f, fpos, fneg.')
parser.add_argument('--f_component_GP',      default='',     help='which part of critic to apply WGAN-GP constraint (on interpolated mu): f, fpos, fneg.')
opt = parser.parse_args()
print(str(opt).replace(', ', '\n'))

os.system('mkdir {0}'.format(opt.outputdir))

opt.manualSeed = random.randint(1, 10000) if not opt.manualSeed else opt.manualSeed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

critic = {'K': critics.K(), 'Kp1': critics.Kp1(), 'Kp1_plogp': critics.Kp1_plogp()}[opt.SSL_critic_type]
assert opt.f_component_Fisher or opt.f_component_Sobolev or opt.f_component_GP, "We need some constraint on the critic, dont we?"

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
if opt.cuda:
    print('CUDNN OK? %s' % (torch.backends.cudnn.is_acceptable(torch.cuda.FloatTensor(1))))
    print('CUDNN-VERSION= %d' % (torch.backends.cudnn.version()))
    torch.backends.cudnn.benchmark = True
if not opt.dataroot:
    opt.dataroot = os.path.join(DEF_DATAROOT, opt.dataset)

# DEFAULTS: dataroot, transform
transform=transforms.Compose([
    transforms.Scale(opt.imageSize),
    transforms.CenterCrop(opt.imageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
if opt.dataset == 'cifar10':
    # override transform
    transform = transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])
    dataset     = dset.CIFAR10(root=opt.dataroot, download=False, transform=transform)
    val_dataset = dset.CIFAR10(root=opt.dataroot, train=False,   transform=transform) 
    if opt.labeledSamples:
        n_classes = 10
        max_labeledSamples = 50e3
elif opt.dataset == 'svhn':
    transform = transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])
    ttransform = lambda x: x % 10 # 1-indexed, 10=0
    dataset     = dset.SVHN(root=opt.dataroot, download=False, transform=transform, target_transform = ttransform)
    val_dataset = dset.SVHN(root=opt.dataroot, split='test',   transform=transform, target_transform = ttransform)
    if opt.labeledSamples:
        n_classes = 10
        max_labeledSamples = 10 * 4659
else:
    raise Exception('Unknown dataset: ' + opt.dataset)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers), drop_last=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10*opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers), drop_last=False) # only fw pass

nc = 3
if opt.conditionalG:
    assert opt.labeledSamples, "Need labeledSamples for conditionalG setting"
    nz_tot = opt.nz + n_classes
else:
    nz_tot = opt.nz

if opt.labeledSamples:
    # assemble class-Balanced subset with labels.
    assert opt.labeledSamples % n_classes == 0 and opt.labeledSamples <= max_labeledSamples
    nSamp = [0] * n_classes
    #lab = [ [s for s in dataset if s[1]==y][:opt.labeledSamples] for y in range(10)] # too slow
    lab_x = torch.FloatTensor(opt.labeledSamples, nc, opt.imageSize, opt.imageSize)
    lab_y = torch.ByteTensor(opt.labeledSamples)
    for s in dataset:
        y = int(s[1]) # squeeze to float
        if nSamp[y] < opt.labeledSamples / n_classes:
            ix = sum(nSamp)
            lab_x[ix].copy_(s[0])
            lab_y[ix] = y
            nSamp[y]  += 1
        if sum(nSamp) == opt.labeledSamples:
            break
    assert sum(nSamp) == opt.labeledSamples # if not enough to fill lab_x, lab_y, this would be bad
    lab_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(lab_x, lab_y),
            batch_size=opt.batchSize, shuffle=True, drop_last=True)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

__import__('models.{}'.format(opt.model_G))
model_G = getattr(models, opt.model_G).G
netG = model_G(opt.imageSize, nz_tot, nc, opt.ngf, opt.ngpu, opt.G_extra_layers)
netG.apply(weights_init)
if opt.netG != '': # load checkpoint if needed
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

__import__('models.{}'.format(opt.model_D))
model_D = getattr(models, opt.model_D).D
netD = model_D(opt.imageSize, nz_tot, nc, opt.ndf, opt.ngpu, opt.D_extra_layers, n_classes, opt.normalization_D)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

x_lab = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
x_unl = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
cpulabels = torch.LongTensor(opt.batchSize)
fakelabels = torch.LongTensor(opt.batchSize)
cpunoise = torch.FloatTensor(opt.batchSize, nz_tot, 1, 1)

# DEFINE fill_noise() and fixed_noise
if opt.conditionalG:
    def code_y_onehot(z, y):
        """ z noisevector (bs x nz+n_classes x 1 x1): 
            code the y class labels [0,n_classes-1] at end of 2nd dimension"""
        assert z.size(0) == y.size(0)
        global n_classes
        z[:, :n_classes].zero_() # code one hot along FIRST n_classes feature maps.
        for ix, y in enumerate(y):
            z[ix, y] = 1

    def fill_noise(noise, lab, new_lab=False):
        global n_classes, cpulabels, cpunoise
        cpulabels.resize_(lab.data.size())
        cpunoise. resize_(noise.data.size())
        if new_lab:
            lab.data.copy_(cpulabels.random_(n_classes)) # randint
        else:
            cpulabels.copy_(lab.data)
        cpunoise.normal_(0, 1)
        code_y_onehot(cpunoise, cpulabels) # has to happen on cpu for eficiency
        noise.data.copy_(cpunoise)

    # Prepare fixed_noise
    fixed_noise = torch.FloatTensor(n_classes, nz_tot, 1, 1).normal_(0, 1)
    fixed_noise = fixed_noise.repeat(n_classes, 1,1,1)
    fixed_y = [x // n_classes for x in range(n_classes**2)] # cycle thru classes
    code_y_onehot(fixed_noise, torch.LongTensor(fixed_y))
else: # no class-conditioning
    # fill_noise same signature as above, but ignore labels
    def fill_noise(noise, lab=None, new_lab=None):
        noise.data.normal_(0,1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz_tot, 1, 1).normal_(0, 1)

one = torch.FloatTensor([1])
mone = one * -1
xe_crit = nn.CrossEntropyLoss()
lambdaF, lambdaS = torch.FloatTensor([0.0]), torch.FloatTensor([0.0]) # lagrange multipliers

if opt.cuda:
    netD.cuda()
    netG.cuda()
    x_lab, x_unl = x_lab.cuda(), x_unl.cuda()
    labels, fakelabels = cpulabels.cuda(), fakelabels.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = cpunoise.cuda(), fixed_noise.cuda()
    xe_crit = xe_crit.cuda()
    lambdaF, lambdaS = lambdaF.cuda(), lambdaS.cuda()
else:
    labels, noise = cpulabels.clone(), cpunoise.clone()

x_lab = Variable(x_lab)
x_unl = Variable(x_unl) # for logging: keep dL/dx
noise = Variable(noise)
labels = Variable(labels)
fakelabels = Variable(fakelabels)
fixed_noise = Variable(fixed_noise, volatile=True)
lambdaF = Variable(lambdaF, requires_grad=True)
lambdaS = Variable(lambdaS, requires_grad=True)

def computeParamNorm(net):
    sq_norm, sq_norm_g = 0, 0
    for p in net.parameters():
        sq_norm += p.data.norm()**2
        sq_norm_g += p.grad.data.norm()**2
    return sq_norm**0.5, sq_norm_g**0.5

# setup optimizer, weight decay to Phi but not V.
paramsD = [{'params': list(netD.main.parameters()), 'weight_decay': opt.wdecay}, 
        {'params': list(netD.V.parameters()), 'weight_decay': opt.wdecayV}]
if hasattr(netD, 'S_dot'):
    paramsD.append({'params': list(netD.S_dot.parameters()), 'weight_decay': opt.wdecayS})
assert len([p for entry in paramsD for p in entry['params']]) == len(list(netD.parameters()))

if opt.rmsprop:
    optimizerD = optim.RMSprop(paramsD,           lr=opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)
elif opt.sgd:
    optimizerD = optim.SGD(paramsD,            lr=opt.lrD)
    optimizerG = optim.SGD(netG.parameters(),  lr=opt.lrG)
else:
    optimizerD = optim.Adam(paramsD,           lr=opt.lrD, betas=(opt.beta1, opt.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
def set_lr(optimizer, lrval):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lrval

gen_iterations = 0
if opt.labeledSamples:
    def inf_iter(dl):
        while True:
            newiter = iter(dl)
            for batch in newiter:
                yield batch
    inf_lab_iter = inf_iter(lab_dataloader)
xe_p, xe_q, xe_gen = None, None, None
for epoch in range(opt.niter):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
        tic = timeit.default_timer()
        is_log_iter = gen_iterations < 25 or gen_iterations % 10 == 0
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        # train the discriminator n_c times
        if opt.hiStart_n_c and (gen_iterations < 25 or gen_iterations % 500 == 0):
            n_c = 100
        else:
            n_c = opt.n_c
        j = 0
        while j < n_c and i < len(dataloader):
            netD.zero_grad()
            if opt.labeledSamples:
                # netD: XE with x_lab
                ###########################
                x_lab_cpu, y_lab_cpu = inf_lab_iter.next()
                x_lab.data.resize_(x_lab_cpu.size()).copy_(x_lab_cpu)
                labels.data.resize_(y_lab_cpu.size(0)).copy_(y_lab_cpu)
                _, S_phi_lab = netD(x_lab)
                xe_p = xe_crit(S_phi_lab, labels)
                # do bw pass, weighted with factor lambdaXE. retain state for GAN loss
                xe_p.backward(one * opt.lambdaXE_D)
            # netD: IPM with x_unl and x_gen
            ###########################
            x_unl_cpu, _ = data_iter.next()
            x_unl.data.resize_(x_unl_cpu.size()).copy_(x_unl_cpu)
            x_unl.requires_grad = True # for grads in objective.
            vphi_p, S_phi_p = netD(x_unl)
            fill_noise(noise, fakelabels, new_lab=True)
            x_gen = netG(noise).detach()
            x_gen.requires_grad = True # for grads in objective.
            vphi_q, S_phi_q = netD(x_gen)
            Ep_vphi, Eq_vphi     = vphi_p.mean(), vphi_q.mean()
            f_p, fpos_p, fneg_p  = critic(vphi_p, S_phi_p)
            f_q, fpos_q, fneg_q  = critic(vphi_q, S_phi_q)
            Ep_f, Eq_f           = f_p.mean(), f_q.mean()
            obj_D = Ep_f - Eq_f
            if opt.f_component_Fisher:
                # Select the right critic component to operate on.
                Fisher_f_p = ({'f': f_p, 'fpos': fpos_p, 'fneg': fneg_p})[opt.f_component_Fisher]
                Fisher_f_q = ({'f': f_q, 'fpos': fpos_q, 'fneg': fneg_q})[opt.f_component_Fisher]
                Ep_f_2, Eq_f_2   = (Fisher_f_p**2).mean(), (Fisher_f_q**2).mean()
                constraintF = (0.5*Ep_f_2 + 0.5*Eq_f_2 - 1)
                obj_D = obj_D - lambdaF * constraintF - opt.rhoFisher/2  * constraintF**2
            if opt.f_component_Sobolev:
                # Select the right critic component to operate on.
                Sobolev_f_p = ({'f': f_p, 'fpos': fpos_p, 'fneg': fneg_p})[opt.f_component_Sobolev]
                Sobolev_f_q = ({'f': f_q, 'fpos': fpos_q, 'fneg': fneg_q})[opt.f_component_Sobolev]
                grad_f_p = torch.autograd.grad(Sobolev_f_p.sum(), x_unl, create_graph=True)[0]
                grad_f_q = torch.autograd.grad(Sobolev_f_q.sum(), x_gen, create_graph=True)[0]
                normgrad_f2_p = grad_f_p.view(grad_f_p.size(0), -1).pow(2).sum(dim=1, keepdim=False)
                normgrad_f2_q = grad_f_q.view(grad_f_q.size(0), -1).pow(2).sum(dim=1, keepdim=False)
                Ep_normgrad_f2 = normgrad_f2_p.mean()
                Eq_normgrad_f2 = normgrad_f2_q.mean()
                constraintS = (0.5*Ep_normgrad_f2 + 0.5*Eq_normgrad_f2 - 1)
                obj_D = obj_D - lambdaS * constraintS - opt.rhoSobolev/2 * constraintS**2
            if opt.f_component_GP:
                # WGAN-GP with interpolates
                interpol_alpha = torch.rand(1)[0]
                x_mu = (x_unl + interpol_alpha * (x_gen - x_unl)).detach()
                x_mu.requires_grad=True
                vphi_mu, S_phi_mu = netD(x_mu)
                f_mu, fpos_mu, fneg_mu = critic(vphi_mu, S_phi_mu)
                GP_f_mu = ({'f': f_mu, 'fpos': fpos_mu, 'fneg': fneg_mu})[opt.f_component_GP]
                grad_f_mu = torch.autograd.grad(GP_f_mu.sum(), x_mu, create_graph=True)[0]
                normgrad_f_mu = grad_f_mu.view(grad_f_mu.size(0), -1).pow(2).sum(dim=1).sqrt()
                constraintGP = ((normgrad_f_mu - 1) ** 2).mean() # NOTE  -1 inside, per-sample
                obj_D = obj_D - opt.lambdaGP * constraintGP
            obj_D.backward(mone) # max_w,v min_alpha
            optimizerD.step()
            # artisanal sgd. Note we minimze lambdaF, lambdaS so a <- a + lr * grad
            if opt.f_component_Fisher:
                lambdaF.data += opt.rhoFisher * lambdaF.grad.data
                lambdaF.grad.data.zero_()
            if opt.f_component_Sobolev:
                lambdaS.data += opt.rhoSobolev * lambdaS.grad.data
                lambdaS.grad.data.zero_()

            i, j = i+1, j+1

        ############################
        # (2) Update G network
        ###########################
        # Here use suffix _gen instead of _p and _q
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        netG.zero_grad()
        fill_noise(noise, fakelabels, new_lab=True) # fakelabels never change size
        x_gen = netG(noise)
        if is_log_iter:
            def hook(g):
                global grad_x_gen_norm
                grad_x_gen_norm = g.data.norm() / (g.size(0) ** 0.5) # per sample: mean per minibatch
            hookhandle = x_gen.register_hook(hook)
        vphi_gen, S_phi_gen = netD(x_gen)
        if opt.conditionalG:
            xe_gen    = xe_crit(S_phi_gen, fakelabels)
            xe_gen.backward(one * opt.lambdaXE_G, retain_variables=True)
        f_gen, fpos_gen, fneg_gen  = critic(vphi_gen, S_phi_gen)
        Egen_vphi = vphi_gen.mean()
        Egen_f    = f_gen.mean()
        obj_G = - Egen_f
        obj_G.backward() # G: min_theta max_beta
        optimizerG.step()

        gen_iterations += 1

        if is_log_iter:
            # For logging: take vals at last D update iteration
            IPM_enum  = Ep_f.data[0] - Eq_f.data[0]
            constr_fisher = (0.5*Ep_f_2.data[0] + 0.5*Eq_f_2.data[0])**0.5 if opt.f_component_Fisher else 0.0
            constr_sobolev = (0.5*Ep_normgrad_f2.data[0] + 0.5*Eq_normgrad_f2.data[0])**0.5 if opt.f_component_Sobolev else 0.0
            constr_gp      = normgrad_f_mu.data[0] if opt.f_component_GP else 0.0
            w_numel, w_norm, grad_w_norm = 0, 0, 0
            w_norm, grad_w_norm         = computeParamNorm(netD.main)
            theta_norm, grad_theta_norm = computeParamNorm(netG)
            v_norm    = netD.V.weight.data.norm()
            s_norm    = 's_norm: %.4f ' % netD.S_dot.weight.data.norm() if hasattr(netD, 'S_dot') else ''
            grad_x_unl_norm = x_unl.grad.data.norm() / (x_unl.size(0) ** 0.5) # per sample: mean over minibatch
            x_unl.grad.data.zero_() # computed on demand; clear it afterwards
            toc = timeit.default_timer()
            #import ipdb; ipdb.set_trace() # try locals() globals()
            logString = ('[%d/%d][%d/%d] IPM_enum: %.4f constr_fisher: %.4f constr_sobolev: %.4f constr_gp: %.4f '
                         'Ep_vphi: %.4f Ep_f: %.4f ' #Ep_f^2: %.4f Ep_normgradf^2: %.4f '
                         'Eq_vphi: %.4f Eq_f: %.4f ' #Eq_f^2: %.4f Eq_normgradf^2: %.4f '
                         'Egen_vphi: %.4f Egen_f: %.4f v_norm: %.4f %s w_norm: %.4f '
                         'grad_w_norm %.4f theta_norm: %.4f grad_theta_norm %.4f grad_x_norm %.4f '
                         'grad_x_gen_norm %.4f lambdaF: %.4f lambdaS: %.4f iter_time %.4f')  % (
                    epoch, opt.niter, gen_iterations, len(dataloader), IPM_enum, constr_fisher, constr_sobolev, constr_gp,
                    Ep_vphi.data[0], Ep_f.data[0], # Ep_f_2.data[0], Ep_normgrad_f2.data[0], 
                    Eq_vphi.data[0], Eq_f.data[0], # Eq_f_2.data[0], Eq_normgrad_f2.data[0],
                    Egen_vphi.data[0], Egen_f.data[0],
                    v_norm, s_norm, w_norm, grad_w_norm, theta_norm, grad_theta_norm,
                    grad_x_unl_norm, grad_x_gen_norm, lambdaF.data[0], lambdaS.data[0], toc-tic)
            if xe_p is not None:
                logString += ' xe_p: %.4f' % xe_p.data[0]
            if xe_q is not None:
                logString += ' xe_q: %.4f' % xe_q.data[0]
            if xe_gen is not None:
                logString += ' xe_gen: %.4f' % xe_gen.data[0]
            print(logString)
        if gen_iterations % 500 == 0:
            vutils.save_image(x_unl_cpu, '{0}/real_samples.png'.format(opt.outputdir), normalize=True)
            netG.eval()
            fake = netG(fixed_noise)
            ncols = n_classes if opt.conditionalG else int(opt.batchSize ** 0.5)
            vutils.save_image(fake.data, '{0}/fake_samples_eval_{1}.png'.format(opt.outputdir, gen_iterations), ncols, normalize=True)
            netG.train()
            fake = netG(fixed_noise)
            vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.outputdir, gen_iterations), ncols, normalize=True)
    
    # End of epoch: validation data
    if val_dataset:
        x_lab.volatile=True
        def eval_except_BN(mod):
            mod.training = bool('BatchNorm' in type(mod).__name__)
        netD.apply(eval_except_BN)
        E_val_f, E_val_fpos, E_val_fneg, E_val_f_2, xe_val, acc = 0, 0, 0, 0, 0, 0
        for val_cpu, lab_cpu in val_dataloader:
            x_lab.data.resize_(val_cpu.size()).copy_(val_cpu)
            labels.data.resize_(lab_cpu.squeeze().size()).copy_(lab_cpu.squeeze())
            vphi_val, S_phi_val = netD(x_lab)
            f_val, fpos_val, fneg_val = critic(vphi_val, S_phi_val)
            if S_phi_val is not None:
                xe_val_ = xe_crit(S_phi_val, labels)
                xe_val      += xe_val_.data[0]
                _, labels_pred = torch.max(S_phi_val, 1, keepdim=False)
                acc         += labels.data.eq(labels_pred.data).sum()
            E_val_f    += f_val.mean().data[0]
            E_val_f_2 += (f_val**2).mean().data[0]
            E_val_fpos += fpos_val.mean().data[0]
            if fneg_val is not None:
                E_val_fneg += fneg_val.mean().data[0]
        netD.train()
        E_val_f    /= len(val_dataloader)
        E_val_f_2  /= len(val_dataloader)
        E_val_fpos /= len(val_dataloader)
        E_val_fneg /= len(val_dataloader)
        xe_val     /= len(val_dataloader)
        acc       = float(acc) / len(val_dataset)
        x_lab.volatile=False
        print('VAL[%d][%d] E_f: %.4f E_f^2: %.4f E_fpos: %.4f E_fneg: %.4f xe: %.4f acc: %.4f' %  
                (epoch, gen_iterations, E_val_f, E_val_f_2, E_val_fpos, E_val_fneg, xe_val, acc))
    if epoch > 0 and opt.lrDcut > 0 and epoch % opt.lrDcut == 0:
        lrFac = 2**(epoch // opt.lrDcut)
        print('End of epoch {}, use lrD = opt.lrD / {} = {}'.format(epoch, lrFac, opt.lrD/lrFac))
        set_lr(optimizerD, opt.lrD/lrFac)
    if epoch > 0 and opt.lrGcut > 0 and epoch % opt.lrGcut == 0:
        lrFac = 2**(epoch // opt.lrGcut)
        print('End of epoch {}, use lrG = opt.lrG / {} = {}'.format(epoch, lrFac, opt.lrG/lrFac))
        set_lr(optimizerG, opt.lrG/lrFac)
    # checkpointing
    torch.save(netG.state_dict(), '{0}/netG_last.pth'.format(opt.outputdir))
    torch.save(netD.state_dict(), '{0}/netD_last.pth'.format(opt.outputdir))
    sys.stdout.flush()

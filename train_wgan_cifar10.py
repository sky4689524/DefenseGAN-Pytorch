import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import copy
import time
import numpy as np
import pickle
from optparse import OptionParser


from gan_model import Generator, Discriminator

INPUT_LATENT = 64 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
ITERS = 100000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)

batch_size = 64 
inital_epoch = 0
learning_rate = 1e-4

display_steps = 10
in_channel = 3
height = 32
width = 32

def get_args():
    
    parser = OptionParser()
    parser.add_option('--data_path', dest='data_path',type='string',
                      default='data/', help='data path')
    parser.add_option('--iterations', dest='ITERS', default=100000, type='int',
                      help='number of critic iterations per generator iteration')
    parser.add_option('--channels', dest='channels', default=3, type='int',
                      help='number of channels')
    parser.add_option('--width', dest='width', default=32, type='int',
                      help='image width')
    parser.add_option('--height', dest='height', default=32, type='int',
                      help='image height')
    parser.add_option('--deviceD', dest='deviceD', default=0, type='int',
                      help='discriminator device number')
    parser.add_option('--deviceG', dest='deviceG', default=0, type='int',
                      help='generator device number')

    (options, args) = parser.parse_args()
    return options

def adjust_lr(optimizer, iteration, init_lr = 1e-4, total_iteration = 200000):
    
    gradient = (float(-init_lr) / total_iteration)
    
    lr = gradient * iteration + init_lr 

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
"""

To Calculate the gradient penalty loss for WGAN GP

"""

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    alpha = alpha.expand(real_samples.size(0), real_samples.size(1), real_samples.size(2), real_samples.size(3))
    
    alpha = alpha.to(device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = D(interpolates)
    
    fake = autograd.Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    
    fake = fake.to(device)
    
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_GAN(netD, netG, inital_epoch, args):
    
    # set parameters
    
    ITERS = args.ITERS
    in_channel = args.channels
    height = args.height
    width = args.width
    
    # set optimizer for generator and discriminator

    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.9))

    # load dataset

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=transform_train)

    train_indices, val_indices = train_test_split(np.arange(len(trainset)), test_size=0.2)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        num_workers=4,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        num_workers=4,
        sampler=valid_sampler
    )

    print('train_loader : {}, val_loader : {}'.format(len(train_loader), len(val_loader)))

    save_losses = []
    dev_disc_costs = []

    if os.path.exists('./defensive_models/cifar10_losses_gp.pickle'):

        with open ('./defensive_models/cifar10_losses_gp.pickle', 'rb') as fp:
            save_losses = pickle.load(fp)

    if os.path.exists('./defensive_models/dev_disc_costs.pickle'):

        with open ('./defensive_models/dev_disc_costs.pickle', 'rb') as fp:
            dev_disc_costs = pickle.load(fp)

    one = torch.FloatTensor([1])
    mone = one * -1

    one = one.to(device_D)
    mone = mone.to(device_D)       


    # Training

    print('training start')

    for iteration in range(inital_epoch, ITERS, 1):

        start_time = time.time()

        adjust_lr(optimizerD, iteration, init_lr = learning_rate, total_iteration = ITERS)
        adjust_lr(optimizerG, iteration, init_lr = learning_rate, total_iteration = ITERS)

        d_loss_real = 0
        d_loss_fake = 0

        #for iter_d in range(CRITIC_ITERS):
        for i, (imgs, _) in enumerate(train_loader):

            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            real_imgs = autograd.Variable(imgs.to(device_D))

            optimizerD.zero_grad()

            # Sample noise as generator input
            z = autograd.Variable(torch.randn(imgs.size(0), INPUT_LATENT))
            z = z.to(device_G)

            # Generate a batch of images
            fake_imgs = netG(z).cpu()
            fake_imgs = fake_imgs.to(device_D)

            # Real images
            real_validity = netD(real_imgs)
            d_loss_real = real_validity.mean()
            d_loss_real.backward(mone)

            # Fake images
            fake_validity = netD(fake_imgs)
            d_loss_fake = fake_validity.mean()
            d_loss_fake.backward(one)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(netD, real_imgs.data, fake_imgs.data, device_D)
            gradient_penalty.backward()

            # Adversarial loss
            loss_D = d_loss_fake - d_loss_real + LAMBDA * gradient_penalty

            #loss_D.backward()
            optimizerD.step()

            optimizerG.zero_grad()

            del real_validity, fake_validity, fake_imgs, gradient_penalty, real_imgs

            # Train the generator every n_critic iterations

            if (i + 1)% CRITIC_ITERS == 0 or (i + 1) == len(train_loader):

                ############################
                # (2) Update G network
                ###########################
                for p in netD.parameters():
                    p.requires_grad = False  # to avoid computation

                # Generate a batch of images    
                fake_imgs = netG(z).cpu()
                fake_imgs = fake_imgs.to(device_D)

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = netD(fake_imgs)
                g_loss = fake_validity.mean()
                g_loss.backward(mone)
                loss_G = -g_loss

                #loss_G.backward()
                optimizerG.step()

                del fake_validity


        save_losses.append([loss_D.item(), loss_G.item()])

        if (iteration + 1) % display_steps == 0 or (iteration + 1) == ITERS:

            print('batch {:>3}/{:>3}, D_cost {:.4f}, G_cost {:.4f}\r'\
                          .format(iteration + 1, ITERS,loss_D.item(), loss_G.item()))

            with open('./defensive_models/cifar10_losses_gp.pickle', 'wb') as fp:
                pickle.dump(save_losses, fp)


            # snapshots for model

            modelG_copy = copy.deepcopy(netG)
            modelG_copy = modelG_copy.cpu()

            modelG_state_dict = modelG_copy.state_dict() 

            modelD_copy = copy.deepcopy(netD)
            modelD_copy = modelD_copy.cpu()

            modelD_state_dict = modelD_copy.state_dict() 

            torch.save({
                'netG_state_dict': modelG_state_dict,
                'netD_state_dict': modelD_state_dict,
                'epoch': iteration
                }, check_point_path)

            del modelG_copy, modelG_state_dict, modelD_copy, modelD_state_dict

        # save generator model after certain iteration
        if (iteration + 1) % 100 == 0 :

            g_path = './defensive_models/gen_cifar10_gp_' + str(iteration) + '.pth' 

            model_copy = copy.deepcopy(netG)
            model_copy = model_copy.cpu()
            model_state_dict = model_copy.state_dict()
            torch.save(model_state_dict, g_path)

            del model_copy

        # save CIFAR 10 generated images by generator model every 1000 time

        if (iteration + 1) % 1000 == 0 :

            save_image(fake_imgs.data, './CIFAR10/samples_{}.png'.format(iteration), nrow=5,normalize=True)

            costs_avg = 0.0
            disc_count = 0

            # validate GAN model
            with torch.no_grad():
                for images,_ in val_loader:

                    imgs = images.to(device_D)

                    D = netD(imgs)

                    costs_avg += -D.mean().cpu().data.numpy()
                    disc_count += 1

                    del images, imgs

            costs_avg = costs_avg / disc_count

            dev_disc_costs.append(costs_avg)

            with open('./defensive_models/dev_disc_costs.pickle', 'wb') as fp:
                pickle.dump(dev_disc_costs, fp)

            print('batch {:>3}/{:>3}, validation disc cost : {:.4f}'.format(iteration, ITERS, costs_avg))
        
        
if __name__ == "__main__":
    
    args = get_args()
    
    device_D = torch.device(args.deviceD)
    device_G = torch.device(args.deviceG)

    # load generator and discriminator model
    netG = Generator()
    summary(netG, input_size = (INPUT_LATENT,), device = 'cpu')

    netD = Discriminator()
    summary(netD, input_size = (3, 32, 32), device = 'cpu')
    
    # set folder to save model checkpoints 
    model_folder = os.path.abspath('./defensive_models')
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    check_point_path =  './defensive_models/snapshots.pth' 

    if os.path.exists(check_point_path):
        checkpoint = torch.load(check_point_path)

        inital_epoch = checkpoint['epoch']

        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])

    netG = netG.to(device_G)
    netD = netD.to(device_D)
    
    train_GAN(netD, netG, inital_epoch, args)

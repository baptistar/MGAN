import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for testing purposes, please do not change!

import copy
import numpy as np
import argparse
import scipy.io

import sys
sys.path.append('../')
from utilities import MatReader, UnitGaussianNormalizer
from DeterministicLotkaVolterra import DeterministicLotkaVolterra

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        output_dim: the dimension of the output vector, a scalar
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, input_dim=10, output_dim=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim),
            self.make_gen_block(hidden_dim, hidden_dim * 2),
            self.make_gen_block(hidden_dim * 2, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, output_dim, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
            )

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, input_dim)
        '''
        x = noise.view(len(noise), self.input_dim)
        return self.gen(x)


def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
      n_samples: the number of samples to generate, a scalar
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)


class Critic(nn.Module):
    '''
    Critic Class
    Values:
        input_dim: the number of channels of the output image, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, input_dim=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(input_dim, hidden_dim * 4),
            self.make_crit_block(hidden_dim * 4, hidden_dim * 2),
            self.make_crit_block(hidden_dim * 2, hidden_dim),
            self.make_crit_block(hidden_dim, 1, final_layer=True),
        )

    def make_crit_block(self, input_channels, output_channels, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a critic block of DCGAN;
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the critic: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)


def get_gradient(crit, real, fake, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)
    
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        inputs=mixed_images,
        outputs=mixed_scores,
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    
    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean((gradient_norm - 1)**2)
    return penalty


def monotonicity_penalty(fake, fake_prime, z, z_prime):
    '''
    Return the average monotonicty penalty, given generator outputs:
        monp = mean(<gen(y,z) - gen(y',z'), z - z'>)
    '''    
    # compute penalty for each sample
    bsize = len(fake)
    mon_penalty = torch.sum(((fake - fake_prime).view(bsize, -1))*((z.detach() - z_prime.detach()).view(bsize,-1)), 1)

    # penalize the average monotonicity across batch of samples
    penalty = torch.mean(mon_penalty)
    return penalty


def get_gen_loss(crit_fake_pred, monp, m_lambda):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    gen_loss = -1. * torch.mean(crit_fake_pred) - m_lambda * monp
    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, gp_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        gp_lambda: the current weight of the gradient penalty 
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + gp_lambda * gp
    return crit_loss

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=100000, help="number of training samples")
    parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs")
    parser.add_argument("--normalize", type=int, default=1, help="normalize or not inputs")
    parser.add_argument("--crit_repeats", type=int, default=1, help="number of F map updates")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size (Should divide Ntest)")
    parser.add_argument("--m_lambda", type=float, default=0.01, help="monotone penalty constant")
    parser.add_argument("--gp_lambda", type=float, default=1.0, help="penalty for gradient in WGAN loss")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    args = parser.parse_args()
    
    # set other parameters
    beta_1 = 0.5
    beta_2 = 0.999
    device = 'cpu' #'cpu'
    display_step = 100
    hidden_dim = 128
    
    # define model
    T = 20;
    LV = DeterministicLotkaVolterra(T)
    dataset = 'DeterministicLotkaVolterra'

    # load training and test data
    Ntrain = args.n_train
    x_train = LV.sample_prior(Ntrain)
    y_train,_ = LV.sample_data(x_train)
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()
    
    # save data
    scipy.io.savemat('training_data.mat',mdict={'x_train':x_train.detach().numpy(), 'y_train':y_train.detach().numpy()})

    #normalize inputs
    if args.normalize == 1:
        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)
    
    # define data
    xy_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    y_loader = DataLoader(TensorDataset(y_train, ), batch_size=args.batch_size, shuffle=True)
    
    # determine reference dimension
    z_dim = x_train.shape[1]
    y_dim = y_train.shape[1]
    
    # define generator and discrminator/critic
    gen = Generator(input_dim=z_dim+y_dim, output_dim=z_dim, hidden_dim=hidden_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(beta_1, beta_2))
    crit = Critic(input_dim=z_dim+y_dim, hidden_dim=hidden_dim).to(device) 
    crit_opt = torch.optim.Adam(crit.parameters(), lr=args.lr, betas=(beta_1, beta_2))
    
    def weights_init(m):
        #if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
    gen = gen.apply(weights_init)
    crit = crit.apply(weights_init)
    
    # define arrays to store results
    cur_step = 0
    generator_losses = []
    critic_losses = []
    gp_penalty = []
    monotonicity = []
    
    for epoch in range(args.n_epochs):
        # Dataloader returns the batches
        for (x,y) in tqdm(xy_loader):
    
            cur_batch_size = len(x)
            #real = real.to(device)
            x, y = x.to(device), y.to(device)
    
            mean_iteration_critic_loss = 0
            mean_iteration_gp_penalty = 0
    
            ### Update critic ###
            for _ in range(args.crit_repeats):
                
                crit_opt.zero_grad()
    
                # evaluate critic at fake inputs
                fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                fake_z = torch.cat((y, fake_noise),1)
                fake = gen(fake_z)
                joint_fake = torch.cat((y, fake), 1)
                crit_fake_pred = crit(joint_fake.detach())
    
                # evaluate critic at real inputs
                joint_real = torch.cat((y, x), 1)
                crit_real_pred = crit(joint_real)
    
                # compute gradient penalty
                epsilon = torch.rand(len(x), 1, device=device, requires_grad=True)
                gradient = get_gradient(crit, joint_real, joint_fake.detach(), epsilon)
                gp = gradient_penalty(gradient)
    
                # compute critic loss
                crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, args.gp_lambda)
    
                # Keep track of the average critic loss in this batch
                mean_iteration_critic_loss += crit_loss.item() / args.crit_repeats
                # Keep track of average gp penalty
                mean_iteration_gp_penalty += gp.item() / args.crit_repeats
    
                # Update gradients
                crit_loss.backward(retain_graph=True)
                # Update optimizer
                crit_opt.step()
    
            critic_losses += [mean_iteration_critic_loss]
            gp_penalty += [mean_iteration_gp_penalty]
    
            ### Update generator ###
    
            gen_opt.zero_grad()
    
            # evaluate critic at fake inputs
            fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
            fake_z_2 = torch.cat((y, fake_noise_2),1)
            fake_2 = gen(fake_z_2)
            joint_fake_2 = torch.cat((y, fake_2), 1)
            crit_fake_pred_2 = crit(joint_fake_2)
            
            # compute monotonicity penalty
            fake_noise_prime = get_noise(cur_batch_size, z_dim, device=device)
            y_prime = next(iter(y_loader))[0].to(device)
            if y_prime.size()[0] > cur_batch_size:
                y_prime = y_prime[0:cur_batch_size,:]
            fake_z_prime = torch.cat((y_prime, fake_noise_prime),1).detach()
            fake_prime = gen(fake_z_prime)
            monp = monotonicity_penalty(fake_2, fake_prime, fake_noise_2, fake_noise_prime) #fake_z_2, fake_z_prime)
    
            # compute generator map loss
            gen_loss = get_gen_loss(crit_fake_pred_2, monp, args.m_lambda)
    
            # Update the weights
            gen_loss.backward()
            gen_opt.step()
    
            # Keep track of the average generator loss
            generator_losses += [gen_loss.item()]
            # Keep track of monotonicity penalty
            monotonicity += [monp.item()]
    
    # plot losses
    step_bins = 20
    num_examples = (len(generator_losses) // step_bins) * step_bins
    plt.figure()
    plt.plot(
        range(num_examples // step_bins), 
        torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
        label="Generator Loss"
    )
    plt.plot(
        range(num_examples // step_bins), 
        torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
        label="Critic Loss"
    )
    plt.legend()
    plt.savefig('losses_'+dataset+'.pdf')
    plt.close()
    
    # plot GP penalty
    plt.figure()
    plt.plot(
        range(num_examples // step_bins), 
        torch.Tensor(gp_penalty[:num_examples]).view(-1, step_bins).mean(1)
    )
    plt.ylabel('GP penalty')
    plt.ylim(-1.5,1.5)
    plt.savefig('gradient_penalty_'+dataset+'.pdf')
    plt.close()
    
    # plot monotonicity
    plt.figure()
    plt.plot(
        range(num_examples // step_bins), 
        torch.Tensor(monotonicity[:num_examples]).view(-1, step_bins).mean(1)
    )
    plt.ylabel('Monotonicity')
    plt.savefig('monotonicity_penalty_'+dataset+'.pdf')
    plt.close()
    
    # save map
    name = 'mgan_' + dataset + '_mon' + str(args.m_lambda)[2:] + '_gp' + str(args.gp_lambda)
    torch.save(gen.state_dict(), name + '_gen.pt')



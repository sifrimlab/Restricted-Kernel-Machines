import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from utils import *
import numpy as np
import torch
import urllib.request
import os.path

""" Instruction for Pre-trained model: Load 'MNIST_trained_RKM_f_mse.tar' when trained with Mean-Squared error for 
    reconstruction. Load (default) 'MNIST_trained_RKM_f_bce.tar' when trained with Binary-Cross entropy error
    for reconstruction """

# Load a Pre-trained model or saved model ====================
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--filename', type=str, default='pre_trained_models/MNIST_trained_RKM_f_bce', help='Enter Filename')
opt_gen = parser.parse_args()

sd_mdl = torch.load('{}.tar'.format(opt_gen.filename, map_location=lambda storage, loc: storage))

net1 = sd_mdl['net1'].cpu()
net3 = sd_mdl['net3'].cpu()
net2 = sd_mdl['net2'].cpu()
net4 = sd_mdl['net4'].cpu()
net1.load_state_dict(sd_mdl['net1_state_dict'])
net3.load_state_dict(sd_mdl['net3_state_dict'])
net2.load_state_dict(sd_mdl['net2_state_dict'])
net4.load_state_dict(sd_mdl['net4_state_dict'])
h = sd_mdl['h'].detach().cpu()
s = sd_mdl['s'].detach().cpu()
V = sd_mdl['V'].detach().cpu()
U = sd_mdl['U'].detach().cpu()
if 'opt' in sd_mdl:
    opt = sd_mdl['opt']
    opt_gen = argparse.Namespace(**vars(opt), **vars(opt_gen))
else:
    opt_gen.mb_size = 200

with torch.no_grad():
    # Generate reconstructed samples ================================================
    opt_gen.shuffle = False
    xt, _, _ = get_mnist_dataloader(args=opt_gen)  # loading data without shuffle
    xtrain = xt.dataset.train_data[:h.shape[0], :, :, :]
    ytrain = xt.dataset.targets[:h.shape[0], :]

    perm1 = torch.randperm(xtrain.size()[0])
    fig2, ax = plt.subplots(4, 4)
    it = 0
    for i in range(4):
        for j in range(4):
            it += 1
            ax[i, j].imshow(xtrain[perm1[it], 0, :, :], cmap='Greys_r')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_title('$' + str(np.argmax(ytrain[perm1[it], :].numpy())) + '$', fontsize=10)
    plt.suptitle('Ground Truth')
    plt.show()

    fig1, ax = plt.subplots(4, 4)
    it = 0
    for i in range(4):
        for j in range(4):
            it += 1
            x_gen = net3(torch.mv(U, h[perm1[it], :])).numpy()
            x_gen = x_gen.reshape(1, 28, 28)
            y_gen = net4(torch.mv(V, h[perm1[it], :])).numpy()

            ax[i, j].imshow(x_gen[0, :], cmap='Greys_r')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_title('$' + str(np.argmax(y_gen)) + '$', fontsize=10)
    plt.suptitle('Reconstructed samples')
    plt.show()

    # # Random samples from fitted distribution over H ============================================================
    gmm = GMM(n_components=1, covariance_type='full', random_state=0).fit(h.numpy())
    z = gmm.sample(400)
    z = torch.FloatTensor(z[0])

    perm2 = torch.randperm(z.shape[0])
    m = 5
    fig3, ax = plt.subplots(m, m)
    it = 0
    for i in range(m):
        for j in range(m):
            it += 1
            x_gen = net3(torch.mv(U, z[perm2[it], :])).numpy()
            x_gen = x_gen.reshape(1, 28, 28)
            y_gen = net4(torch.mv(V, z[perm2[it], :])).numpy()

            ax[i, j].imshow(x_gen[0, :], cmap='Greys_r')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_title('$' + str(np.argmax(y_gen)) + '$', fontsize=10)
    plt.suptitle('Randomly sampled from dist. over $\mathcal{H}$')
    plt.show()

    # Interpolations ================================================
    indx1 = 0
    indx2 = 1
    indx3 = 2
    indx4 = 3

    y1 = h[indx1, :]
    y2 = h[indx2, :]
    y3 = h[indx3, :]
    y4 = h[indx4, :]

    # 2-D Interpolation %%%%%%%%%%%%
    m = 12
    T, S = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))
    T = np.ravel(T, order="F")
    S = np.ravel(S, order="F")

    fig5, ax = plt.subplots(m, m)
    it = 0
    for i in range(m):
        for j in range(m):
            # weights
            lambd = np.flip(
                np.hstack((S[it] * T[it], (1 - S[it]) * T[it], S[it] * (1 - T[it]), (1 - S[it]) * (1 - T[it]))), 0)

            # computation
            yop = lambd[0] * y1 + lambd[1] * y2 + lambd[2] * y3 + lambd[3] * y4

            x_gen = net3(torch.mv(U, yop)).detach().numpy()
            x_gen = x_gen.reshape(1, 28, 28)
            y_gen = net4(torch.mv(V, yop)).detach().numpy()

            y_gen[y_gen < 0] = 0
            ps = np.exp(y_gen)
            ps /= np.sum(ps)
            ind = np.argpartition(ps, -2)[-2:]

            ax[i, j].imshow(x_gen[0, :], cmap='Greys_r')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            it += 1
    plt.suptitle('2D Interpolation')
    plt.show()

    # 1-D Interpolation %%%%%%%%%%
    m = 30
    lambd = torch.tensor(np.linspace(0, 1, m))
    fig, (ax1, ax2) = plt.subplots(1, 2)


    def animate(j):
        # computation
        yop = (1 - lambd[j]) * y1 + lambd[j] * y2

        x_gen = net3(torch.mv(U, yop)).detach().numpy()
        x_gen = x_gen.reshape(1, 28, 28)
        y_gen = net4(torch.mv(V, yop)).detach().numpy()

        ind = np.argmax(y_gen)

        xtr = (1 - lambd[j]) * xtrain[indx1, :] + lambd[j] * xtrain[indx2, :]
        xtr = xtr.numpy()

        # display
        ax1.imshow(xtr[0, :], cmap='Greys_r')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title('Ground Truth')

        ax2.imshow(x_gen[0, :], cmap='Greys_r')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title('Generated Label & Image:\n {:.0f}'.format(ind), fontsize=10)
        plt.pause(0.05)


    plt.suptitle('Interpolations')
    ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, m))
    plt.show()

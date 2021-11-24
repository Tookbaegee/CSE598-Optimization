import torch
import copy
import numpy as np
import torchvision
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


import matplotlib.pyplot as plt


from fashionmnist import FashionCNN, FashionDataset

def init_directions(model):
    noises = []

    n_params = 0
    for name, param in model.named_parameters():
        delta = torch.normal(.0, 1, size=param.size()).to('cuda:0')
        nu = torch.normal(.0, 1, size=param.size()).to('cuda:0')

        param_norm = torch.norm(param).to('cuda:0')
        delta_norm = torch.norm(delta).to('cuda:0')
        nu_norm = torch.norm(nu).to('cuda:0')

        delta /= delta_norm
        delta *= param_norm

        nu /= nu_norm
        nu *= param_norm
        noises.append((delta, nu))

        n_params += np.prod(param.size())

    print(f'A total of {n_params:,} parameters.')

    return noises


def init_network(model, all_noises, alpha, beta):
    with torch.no_grad():
        for param, noises in zip(model.parameters(), all_noises):
            delta, nu = noises
            new_value = param + (alpha * delta) + (beta * nu)
            param.copy_(new_value)
    return model

def run_landscape_gen(name, model, batch_size):
    BATCH_SIZE = batch_size
    RESOLUTION = 50

    test_csv = pd.read_csv("fashion-mnist_test.csv")
    dataset = FashionDataset(test_csv, transform=transforms.Compose([transforms.ToTensor()]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    error = torch.nn.CrossEntropyLoss()
    print('Testing model')
      
    noises = init_directions(model)

    A, B = np.meshgrid(np.linspace(-1, 1, RESOLUTION),
                    np.linspace(-1, 1, RESOLUTION), indexing='ij')

    loss_surface = np.empty_like(A)

    for i in range(RESOLUTION):
        for j in range(RESOLUTION):
            total_loss = 0.
            n_batch = 0
            alpha = A[i, j]
            beta = B[i, j]
            modelcp = copy.deepcopy(model)
            net = init_network(modelcp, noises, alpha, beta).to('cuda:0')

            for batch, labels in dataloader:
                batch = batch.to('cuda:0')
                labels = labels.to('cuda:0')
                with torch.no_grad():
                    preds = net(batch)
                    loss = error(preds, labels)
                    total_loss += loss.item()
                    n_batch += 1
            loss_surface[i, j] = total_loss / (n_batch * BATCH_SIZE)
            del net
            torch.cuda.empty_cache()
            # print(f'alpha : {alpha}, beta : {beta}, loss : {loss_surface[i, j]}')
            

    plt.figure(figsize=(10, 10))
    plt.contour(A, B, loss_surface)                       
    plt.savefig(f'{name}_results/contour_bs_.png', dpi=300)
    plt.close()

    np.save(f'{name}_results/xx_model.npy', A)
    np.save(f'{name}_results/yy_model.npy', B)
    np.save(f'{name}_results/zz_model.npy', loss_surface)

def generate_plots(name):
    xx = np.load(f'{name}_results/xx_model.npy')
    yy = np.load(f'{name}_results/yy_model.npy')
    zz = np.load(f'{name}_results/zz_model.npy')

    zz = np.log(zz)

    plt.figure(figsize=(10, 10))
    plt.contour(xx, yy, zz)
    plt.savefig(f'{name}_results/log_contour1.png', dpi=300)
    plt.close()

    ## 3D plot
    fig, ax = plt.subplots(subplot_kw={'projection' : '3d'})
    ax.set_axis_off()
    surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    plt.savefig(f'{name}_results/log_surface1.png', dpi=300,
                format='png', bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    ax.set_axis_off()

    def init():
        ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        return fig,

    def animate(i):
        ax.view_init(elev=(15 * (i // 15) + i % 15) + 0., azim=i)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        return fig,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=100, interval=20, blit=True)

    anim.save(f'{name}_results/log_surface1.gif',
              fps=15,  writer='imagemagick')


if __name__ == '__main__':
    model = FashionCNN()
    model.load_state_dict(torch.load('fashionCNN-adam.pt'))
    model.to('cuda:0')
    run_landscape_gen('adam', model, 100)
    

    model = FashionCNN()
    model.load_state_dict(torch.load('fashionCNN-rms.pt'))
    model.to('cuda:0')
    run_landscape_gen('rms', model, 100)
    

    model = FashionCNN()
    model.load_state_dict(torch.load('fashionCNN-adagrad.pt'))
    model.to('cuda:0')
    run_landscape_gen('adagrad', model, 100)
    

    model = FashionCNN()
    model.load_state_dict(torch.load('fashionCNN-sgd.pt'))
    model.to('cuda:0')
    run_landscape_gen('sgd', model, 100)
    
    
    generate_plots('adam')
    generate_plots('rms')
    generate_plots('adagrad')
    generate_plots('sgd')

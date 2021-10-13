import numpy as np
import argparse
from utils import concave_fun_eval, create_pf, circle_points
from hco_search import hco_search

import matplotlib.pyplot as plt
from latex_utils import latexify


parser = argparse.ArgumentParser()
parser.add_argument('--n', type = int, default=20, help="the batch size for the unbiased model")
parser.add_argument('--m', type = int, default=20, help="the batch size for the predicted model")
parser.add_argument('--eps0', type = float, default=0.6, help="max_epoch for unbiased_moe")
parser.add_argument('--eps1', type = float, default=1e-4, help="max_epoch for predictor")
parser.add_argument('--eps2', type = float, default=1e-4, help="iteras for printing the loss info")
parser.add_argument('--para_delta', type = float, default=0.1, help="max_epoch for unbiased_moe")
parser.add_argument('--lr_delta', type = float, default=0.01, help="max_epoch for predictor")
parser.add_argument('--step_size', type = float, default=0.005, help="iteras for printing the loss info")
parser.add_argument('--max_iters', type = int, default=700, help="iteras for printing the loss info")
parser.add_argument('--grad_tol', type = float, default=1e-4, help="iteras for printing the loss info")
parser.add_argument('--store_xs', type = bool, default=False, help="iteras for printing the loss info")

args = parser.parse_args() 



def case1_satisfyingMCF():
    n = args.n     # dim of solution space
    m = args.m       # dim of objective space
    ##construct x0
    x0 = np.zeros(n)
    x0[range(0, n, 2)] = -0.2
    x0[range(1, n, 2)] = -0.2
    eps_set = [0.8, 0.6, 0.4, 0.2]
    color_set = ["c", "g", "orange", "b"]
    latexify(fig_width=2.2, fig_height=1.8)
    l0, _ = concave_fun_eval(x0)
    max_iters = args.max_iters
    relax = True
    pf = create_pf()
    fig = plt.figure()
    fig.subplots_adjust(left=.12, bottom=.12, right=.9, top=.9)
    label = 'Pareto\nFront' if relax else ''
    plt.plot(pf[:, 0], pf[:, 1], lw=2.0, c='k', label=label)
    label = r'$l(\theta^0)$' 
    plt.scatter([l0[0]], [l0[1]], c='r', s=40)
    plt.annotate(label, xy = (l0[0]+0.03, l0[1]), xytext = (l0[0]+0.03, l0[1]))
    for idx, eps0 in  enumerate(eps_set):
        if eps0 == 0.2:
            eps_plot = np.array([[ i*0.1 * 0.903, 0.2] for i in range(11)])
            plt.plot(eps_plot[:,0], eps_plot[:,1], color = "gray", label = r'$\epsilon$', lw=1, ls='--')
        elif eps0 == 0.4:
            eps_plot = np.array([[ i*0.1 * 0.807, 0.4] for i in range(11)])
            plt.plot(eps_plot[:,0], eps_plot[:,1], color = "gray",  lw=1, ls='--')
        elif eps0 == 0.6:
            eps_plot = np.array([[ i*0.1* 0.652, 0.6] for i in range(11)])
            plt.plot(eps_plot[:,0], eps_plot[:,1], color = "gray",  lw=1, ls='--')
        elif eps0 == 0.8:
            eps_plot = np.array([[ i*0.1 * 0.412, 0.8] for i in range(11)])
            plt.plot(eps_plot[:,0], eps_plot[:,1], color = "gray",  lw=1, ls='--')
        else:
            print("error eps0")
            exit()
        c = color_set[idx]
        _, res = hco_search(concave_fun_eval, x=x0, deltas = [0.5, 0.5], para_delta = 0.5, lr_delta = args.lr_delta, relax=False, eps=[eps0, args.eps1, args.eps2], max_iters=max_iters,
                n_dim=args.n, step_size=args.step_size, grad_tol=args.grad_tol, store_xs=args.store_xs)
        ls = res['ls']
        alpha = 1.0
        zorder = 1 
        # plt.plot(ls[:, 0], ls[:, 1], c=c, lw=2.0, alpha=alpha, zorder=zorder)
        plt.plot(ls[:, 0], ls[:, 1], c=c, lw=2.0)
        print(ls[-1])
        plt.scatter(ls[[-1], 0], ls[[-1], 1], c=c, s=40)
    plt.xlabel(r'$l_1$')
    plt.ylabel(r'$l_2$', rotation = "horizontal")
    plt.legend(loc='lower left', handletextpad=0.3, framealpha=0.9)
    ax = plt.gca()
    ax.xaxis.set_label_coords(1.05, -0.02)
    ax.yaxis.set_label_coords(-0.02, 1.02)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.savefig('figures/satifying'  + '.pdf')
    plt.close()



def case2_violatingMCF():
    n = args.n     # dim of solution space
    m = args.m       # dim of objective space
    ##construct x0
    x0 = np.zeros(n)
    x0[range(0, n, 2)] = 0.3
    x0[range(1, n, 2)] = -0.3
    eps_set = [0.2, 0.4, 0.6, 0.8]
    color_set = ["c", "g", "orange", "b"]
    latexify(fig_width=2.2, fig_height=1.8)
    l0, _ = concave_fun_eval(x0)
    max_iters = args.max_iters
    relax = True
    pf = create_pf()
    fig = plt.figure()
    fig.subplots_adjust(left=.12, bottom=.12, right=.9, top=.9)
    label = 'Pareto\nFront' if relax else ''
    plt.plot(pf[:, 0], pf[:, 1], lw=2.0, c='k', label=label)
    label = r'$l(\theta^0)$' 
    plt.scatter([l0[0]], [l0[1]], c='r', s=40)
    plt.annotate(label, xy = (l0[0]+0.03, l0[1]), xytext = (l0[0]+0.03, l0[1]))
    for idx, eps0 in  enumerate(eps_set):
        if eps0 == 0.2:
            eps_plot = np.array([[ i*0.1 * 0.903, 0.2] for i in range(11)])
            plt.plot(eps_plot[:,0], eps_plot[:,1], color = "gray", label = r'$\epsilon$', lw=1, ls='--')
        elif eps0 == 0.4:
            eps_plot = np.array([[ i*0.1 * 0.807, 0.4] for i in range(11)])
            plt.plot(eps_plot[:,0], eps_plot[:,1], color = "gray",  lw=1, ls='--')
        elif eps0 == 0.6:
            eps_plot = np.array([[ i*0.1* 0.652, 0.6] for i in range(11)])
            plt.plot(eps_plot[:,0], eps_plot[:,1], color = "gray",  lw=1, ls='--')
        elif eps0 == 0.8:
            eps_plot = np.array([[ i*0.1 * 0.412, 0.8] for i in range(11)])
            plt.plot(eps_plot[:,0], eps_plot[:,1], color = "gray",  lw=1, ls='--')
        else:
            print("error eps0")
            exit()
        c = color_set[idx]
        _, res = hco_search(concave_fun_eval, x=x0, deltas = [0.5, 0.5], para_delta = 0.5, lr_delta = args.lr_delta, relax=False, eps=[eps0, args.eps1, args.eps2], max_iters=max_iters,
                n_dim=args.n, step_size=args.step_size, grad_tol=args.grad_tol, store_xs=args.store_xs)
        ls = res['ls']
        alpha = 1.0
        zorder = 1 
        plt.plot(ls[:, 0], ls[:, 1], c=c, lw=2.0, alpha=alpha, zorder=zorder)
        print(ls[-1])
        plt.scatter(ls[[-1], 0], ls[[-1], 1], c=c, s=40)
    plt.xlabel(r'$l_1$')
    plt.ylabel(r'$l_2$', rotation = "horizontal")
    plt.legend(loc='lower left', handletextpad=0.3, framealpha=0.9)
    ax = plt.gca()
    ax.xaxis.set_label_coords(1.05, -0.02)
    ax.yaxis.set_label_coords(-0.02, 1.02)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.savefig('figures/violating'  + '.pdf')
    plt.close()


if __name__ == '__main__':
    case1_satisfyingMCF()
    case2_violatingMCF()
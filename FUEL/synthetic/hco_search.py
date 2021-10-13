import numpy as np

from hco_lp import HCO_LP

def hco_search(multi_obj_fg, x=None, deltas = None, para_delta = 0.5, lr_delta = 0.1, relax=False, eps=[1e-4, 1e-4, 1e-4], max_iters=100,
               n_dim=20, step_size=.1, grad_tol=1e-4, store_xs=False):
    # r = [0.98, 0.15]
    if relax:
        print('relaxing')
    else:
        print('Restricted')
    # randomly generate one solution
    x = np.random.randn(n_dim) if x is None else x
    deltas = [0.5, 0.5]
           # number of objectives
    lp = HCO_LP( n_dim, eps) # eps [eps_disparity,]
    lss, gammas, d_nds = [], [], []
    if store_xs:
        xs = [x]

    # find the Pareto optimal solution
    desc, asce = 0, 0
    for t in range(max_iters):
        x = x.reshape(-1)
        ls, d_ls = multi_obj_fg(x)
        alpha, deltas = lp.get_alpha(ls, d_ls, deltas, para_delta, lr_delta, relax=relax)
        if lp.last_move == "dom":
            desc += 1
        else:
            asce += 1
        lss.append(ls)
        gammas.append(lp.gamma)
        d_nd = alpha @ d_ls
        d_nds.append(np.linalg.norm(d_nd, ord=np.inf)) 


        if np.linalg.norm(d_nd, ord=np.inf) < grad_tol:
            print('converged, ', end=',')
            break
        x = x - 10. * max(ls[1], 0.1) * step_size * d_nd
        if store_xs:
            xs.append(x)

    print(f'# iterations={asce+desc}; {100. * desc/(desc+asce)} % descent')
    res = {'ls': np.stack(lss),
           'gammas': np.stack(gammas)}
    if store_xs:
        res['xs': xs]
    return x, res

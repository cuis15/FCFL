import numpy as np
import cvxpy as cp
import cvxopt
import pdb
# from cvxpylayers.torch import CvxpyLayer

class PO_LP(object): # hard-constrained optimization

    def __init__(self, n_theta, n_alpha,  eps):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        # self.objs = objs # the two objs [l, g].
        self.n_theta = n_theta # the dimension of \theta
        self.n_alpha = n_alpha
        self.eps = eps # the error bar of the optimization process eps1 < g
        self.grad_d = cp.Parameter((n_alpha, n_theta))       # [d_l, d_g] * d_l or [d_l, d_g] * d_g.
        self.l = cp.Parameter(( n_theta, 1))
        self.l_g = cp.Parameter(( n_theta, n_alpha))
        self.alpha = cp.Variable((1,n_alpha))    # Variable to optimize
         # disparities has been satisfies, in this case we only maximize the performance
        
        obj_dom = cp.Maximize(cp.sum((self.alpha @  self.grad_d) @ self.l))
        constraints_dom = [self.alpha >= 0, cp.sum(self.alpha) == 1,
                            (self.alpha @ self.grad_d) @ self.l_g >=0]

        self.prob_dom = cp.Problem(obj_dom, constraints_dom)  # LP balance

        self.gamma = 0     # Stores the latest Optimum value of the LP problem
        self.disparity = 0     # Stores the latest maximum of selected K disparities


    # pytorch version
    # def get_alpha(self, grads, grad_l, l_g):
        
    #     self.cvxpy = CvxpyLayer(self.prob_dom, parameters = [self.grad_d, self.l, self.l_g], variables = [self.alpha], gp = True)
        
    #     alpha, = self.cvxpy(grads, grad_l, l_g)
    #     return alpha.clone().detach()

    # numpy version

    def get_alpha(self, grads, grad_l, l_g):
        
        self.grad_d.value = grads.cpu().numpy()
        self.l.value = grad_l.cpu().numpy()
        self.l_g.value = l_g.cpu().numpy()
        self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
        return self.alpha.value, self.gamma



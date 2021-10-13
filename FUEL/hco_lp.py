import numpy as np
import cvxpy as cp
import cvxopt
import pdb
#from cvxpylayers.torch import CvxpyLayer
import torch
class HCO_LP(object): # hard-constrained optimization

    def __init__(self, n, eps):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        # self.objs = objs # the two objs [l, g].
        self.n = n # the dimension of \theta
        self.eps = eps # the error bar of the optimization process [eps1 < g, eps2 < delta1, eps3 < delta2]
        self.deltas = cp.Parameter(2) # the two deltas of the objectives [l1, l2]
        self.Ca1 = cp.Parameter((2,1))       # [d_l, d_g] * d_l or [d_l, d_g] * d_g.
        self.Ca2 = cp.Parameter((2,1))

        self.alpha = cp.Variable((1,2))     # Variable to optimize
         # disparities has been satisfies, in this case we only maximize the performance
        obj_dom = cp.Maximize(self.alpha @  self.Ca1) 
        obj_fair = cp.Maximize(self.alpha @ self.Ca2)


        constraints_dom = [self.alpha >= 0, cp.sum(self.alpha) == 1]
        constraints_fair = [self.alpha >= 0, cp.sum(self.alpha) == 1,
                            self.alpha @ self.Ca1 >= 0]

        self.prob_dom = cp.Problem(obj_dom, constraints_dom)  # LP balance
        self.prob_fair = cp.Problem(obj_fair, constraints_fair)

        self.gamma = 0     # Stores the latest Optimum value of the LP problem
        self.disparity = 0     # Stores the latest maximum of selected K disparities



    # # pytorch version
    # def get_alpha(self, dis_max, d_l1, d_l2, deltas, factor_delta, lr_delta):

    #     d_ls = torch.cat((d_l1, d_l2))
    #     if dis_max[1]<= self.eps[0]: # [l, g] disparities < eps0
    #         # self.Ca1.value = d_ls @ d_l1 
    #         self.cvxpy = CvxpyLayer(self.prob_dom, parameters = [self.Ca1], variables = [self.alpha])

    #         Ca1_value = d_ls @ d_l1.t() 
    #         alpha, = self.cvxpy(Ca1_value)

    #         self.last_move = "dom"
    #         return alpha.clone().detach(), deltas

    #     else:
    #         self.cvxpy = CvxpyLayer(self.prob_fair, parameters = [self.Ca1, self.Ca2], variables = [self.alpha])
    #         Ca1_value = d_ls @ d_l1.t()
    #         Ca2_value = d_ls @ d_l2.t()

    #         alpha, = self.cvxpy(Ca1_value, Ca2_value)
    #         if self.eps[1] < deltas[0] and np.linalg.norm(d_l1.cpu()) * factor_delta <= deltas[0]:
    #             deltas[0] = lr_delta * deltas[0]
    #         if self.eps[2] < deltas[1] and np.linalg.norm(d_l2.cpu()) * factor_delta <= deltas[1]:
    #             deltas[1] = lr_delta * deltas[1]
    #             self.last_move = "fair"
    #         return alpha.clone().detach(), deltas



    def get_alpha(self, dis_max, d_l1, d_l2, deltas, factor_delta, lr_delta):


        d_ls = torch.cat((d_l1, d_l2))
        if dis_max[1]<= self.eps[0]: # [l, g] disparities < eps0
            # self.Ca1.value = d_ls @ d_l1 
            self.Ca1.value = (d_ls @ d_l1.t()).cpu().numpy() 
            self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "dom"
            if self.eps[1] < deltas[0] and np.linalg.norm(d_l1.cpu()) * factor_delta <= deltas[0]:
                deltas[0] = lr_delta * deltas[0]
            if self.eps[2] < deltas[1] and np.linalg.norm(d_l2.cpu()) * factor_delta <= deltas[1]:
                deltas[1] = lr_delta * deltas[1]
            return self.alpha.value, deltas

        else:
            self.Ca1.value = (d_ls @ d_l1.t()).cpu().numpy() 
            self.Ca2.value = (d_ls @ d_l2.t()).cpu().numpy() 
            self.gamma = self.prob_fair.solve(solver=cp.GLPK, verbose=False)
            if self.eps[1] < deltas[0] and np.linalg.norm(d_l1.cpu()) * factor_delta <= deltas[0]:
                deltas[0] = lr_delta * deltas[0]
            if self.eps[2] < deltas[1] and np.linalg.norm(d_l2.cpu()) * factor_delta <= deltas[1]:
                deltas[1] = lr_delta * deltas[1]
            self.last_move = "fair"
            return self.alpha.value, deltas
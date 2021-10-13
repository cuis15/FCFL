# lenet base model for Pareto MTL
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataload import LoadDataset
import os
import numpy as np
import argparse
import json
from hco_lp import HCO_LP
import torch.nn.functional as F 
import pdb
from po_lp import PO_LP
import pickle
from sklearn.metrics import roc_auc_score, classification_report


class RegressionTrain(torch.nn.Module):

    def __init__(self, model, disparity_type = "DP", dataset  = "adult"):
        super(RegressionTrain, self).__init__()
        self.model = model
        self.loss = nn.BCELoss()
        self.disparity_type = disparity_type
        self.dataset = dataset


    def forward(self, x, y, A):
        ys_pre = self.model(x).flatten()
        ys = torch.sigmoid(ys_pre)
        hat_ys = (ys >=0.5).float()
        task_loss = self.loss(ys, y)
        accs = torch.mean((hat_ys == y).float()).item()
        aucs = roc_auc_score(y.cpu(), ys.clone().detach().cpu())
        if True:

            # pred_dis = torch.abs(torch.sum(ys * A)/torch.sum(A) - torch.sum(ys * (1-A))/torch.sum(1-A))
            # pred_dis = torch.sum(F.sigmoid(10 * (ys - 0.5) + 0.5 ) * A)/torch.sum(A) - torch.sum(F.sigmoid(10 * (ys - 0.5) + 0.5) * (1-A))/torch.sum(1-A)
            if self.disparity_type == "DP":
                pred_dis = torch.sum(torch.sigmoid(10 * ys_pre) * A)/torch.sum(
                    A) - torch.sum(torch.sigmoid(10 * ys_pre) * (1-A))/torch.sum(1-A)
                disparitys = torch.sum(hat_ys * A)/torch.sum(A) - \
                    torch.sum(hat_ys * (1-A))/torch.sum(1-A)

            elif self.disparity_type == "Eoppo":
                if "eicu_d" in self.dataset:
                    pred_dis = torch.sum(torch.sigmoid(10 * (1-ys_pre)) * A * (1-y))/torch.sum(
                        A * (1-y)) - torch.sum(torch.sigmoid(10 * (1-ys_pre)) * (1-A) * (1-y))/torch.sum((1-A)*(1-y))

                    disparitys = torch.sum((1-hat_ys) * A * (1-y))/torch.sum(A * (1-y)) - \
                        torch.sum((1-hat_ys) * (1-A) * (1-y)) / \
                        torch.sum((1-A) * (1-y))
                else:
                    pred_dis = torch.sum(torch.sigmoid(10 * ys_pre) * A * y)/torch.sum(
                        A * y) - torch.sum(torch.sigmoid(10 * ys_pre) * (1-A) * y)/torch.sum((1-A)*y)
                    disparitys = torch.sum(hat_ys * A * y)/torch.sum(A * y) - \
                        torch.sum(hat_ys * (1-A) * y)/torch.sum((1-A) * y)
            disparitys = disparitys.item()
            return task_loss, accs, aucs, pred_dis, disparitys, ys

        else:
            print("error model in forward")
            exit()
            # if self.dataset == "adult" and self.disparity_type == "DP" and self.sensitive_attr == "sex":
            #     specific_eps = self.weight_eps * np.array([-0.24607987 - 0.14961589])
            # elif self.dataset == "adult" and self.disparity_type == "DP" and self.sensitive_attr == "race":
            #     specific_eps = self.weight_eps * np.array([-0.24607987 - 0.14961589])
            
    def randomize(self):
        self.model.apply(weights_init)





def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.weight.data *= 0.1


class RegressionModel(torch.nn.Module):
    def __init__(self, n_feats, n_hidden):
        super(RegressionModel, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(n_feats, 1))
        # self.layers.append(nn.Linear(n_hidden, 1))

    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            y_temp = self.layers[i](y)
            if i < len(self.layers) - 1:
                y = torch.tanh(y_temp)
            else:
                y = y_temp
                # y = torch.sigmoid(y_temp)
        return y


class MODEL(object):
    def __init__(self, args, logger, writer):
        super(MODEL, self).__init__()


        self.dataset = args.dataset
        self.uniform_eps = bool(args.uniform_eps)
        if self.uniform_eps:
            self.eps = args.eps
        else:
            args.eps[0] = 0.0
            self.eps = [0.0, args.eps[1], args.eps[2]]

        self.max_epoch1 = args.max_epoch_stage1
        self.max_epoch2 = args.max_epoch_stage2
        self.ckpt_dir = args.ckpt_dir
        self.global_epoch = args.global_epoch
        self.log_pickle_dir = args.log_dir
        self.per_epoches = args.per_epoches
        self.factor_delta = args.factor_delta
        self.lr_delta = args.lr_delta
        self.deltas = np.array([0.,0.])
        self.deltas[0] = args.delta_l
        self.deltas[1] = args.delta_g
        self.eval_epoch = args.eval_epoch
        self.data_load(args)
        self.logger = logger
        self.logger.info(str(args))
        self.n_linscalar_adjusts = 0
        self.done_dir = args.done_dir
        self.FedAve = args.FedAve
        self.writer = writer
        self.uniform = args.uniform
        self.performence_only = args.uniform
        self.policy = args.policy
        self.disparity_type = args.disparity_type
        self.model = RegressionTrain(RegressionModel(args.n_feats, args.n_hiddens), args.disparity_type, args.dataset)
        self.log_train = dict()
        self.log_test = dict()
        self.baseline_type = args.baseline_type
        self.weight_fair = args.weight_fair
        self.sensitive_attr = args.sensitive_attr
        self.weight_eps = args.weight_eps

        if torch.cuda.is_available():
            self.model.cuda()
        self.optim = torch.optim.SGD(self.model.parameters(), lr=args.step_size, momentum=0., weight_decay=1e-4)

        _, n_params = self.getNumParams(self.model.parameters())
        self.hco_lp = HCO_LP(n=n_params, eps = self.eps)
        self.po_lp = PO_LP(n_theta=n_params, n_alpha = 1+ self.n_clients,  eps = self.eps[0])
        if int(args.load_epoch) != 0:
            self.model_load(str(args.load_epoch))

        self.commandline_save(args)

    def commandline_save(self, args):
        with open(args.commandline_file, "w") as f:
            json.dump(args.__dict__, f, indent =2)

    def getNumParams(self, params):
        numParams, numTrainable = 0, 0
        for param in params:
            npParamCount = np.prod(param.data.shape)
            numParams += npParamCount
            if param.requires_grad:
                numTrainable += npParamCount
        return numParams, numTrainable


    def model_load(self, ckptname='last'):

        if ckptname == 'last':
            ckpts = os.listdir(self.ckpt_dir)
            if not ckpts:
                self.logger.info("=> no checkpoint found")
                exit()
            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])
        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            # self.global_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model'])
            self.optim.load_state_dict(checkpoint['optim'])
            self.logger.info("=> loaded checkpoint '{} (epoch {})'".format(filepath, self.global_epoch))
        
        else:
            self.logger.info("=> no checkpoint found at '{}'".format(filepath))


    def model_save(self, ckptname = None):
        states = {'epoch':self.global_epoch,
                  'model':self.model.state_dict(),
                  'optim':self.optim.state_dict()}
        if ckptname == None:
            ckptname = str(self.global_epoch)
        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        os.makedirs(self.ckpt_dir, exist_ok = True)
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        self.logger.info("=> saved checkpoint '{}' (epoch {})".format(filepath, self.global_epoch))


    def data_load(self, args):
        self.client_train_loaders, self.client_test_loaders = LoadDataset(args)
        self.n_clients = len(self.client_train_loaders)
        self.iter_train_clients = [enumerate(i) for i in self.client_train_loaders]
        self.iter_test_clients = [enumerate(i) for i in self.client_test_loaders]


    def valid_stage1(self,  if_train = False, epoch = -1):
        with torch.no_grad():
            losses = []
            accs = []
            diss = []
            pred_diss = []
            aucs = []
            if if_train:
                loader = self.client_train_loaders
            else:
                loader = self.client_test_loaders
            for client_idx, client_test_loader in enumerate(loader):
                valid_loss = []
                valid_accs = []
                valid_diss = []
                valid_pred_dis = []
                valid_auc = []
                for it, (X, Y, A) in enumerate(client_test_loader):  
                    X = X.float()
                    Y = Y.float()
                    A = A.float()
                    if torch.cuda.is_available():
                        X = X.cuda()
                        Y = Y.cuda()
                        A = A.cuda()
                    loss, acc, auc, pred_dis, disparity, pred_y = self.model(X, Y, A)
                    valid_loss.append(loss.item())
                    valid_accs.append(acc)  
                    valid_diss.append(disparity)
                    valid_pred_dis.append(pred_dis.item())
                    valid_auc.append(auc)
                assert len(valid_auc)==1
                losses.append(np.mean(valid_loss))
                accs.append(np.mean(valid_accs))
                diss.append(np.mean(valid_diss))
                pred_diss.append(np.mean(valid_pred_dis))
                aucs.append(np.mean(valid_auc))
            self.logger.info("is_train: {}, epoch: {}, loss: {}, accuracy: {}, auc: {}, disparity: {}, pred_disparity: {}".format(if_train, self.global_epoch, losses, accs, aucs, diss, pred_diss))
            self.log_test[str(epoch)] = { "client_losses": losses, "pred_client_disparities": pred_diss, "client_accs": accs, "client_aucs": aucs, "client_disparities": diss, "max_losses": [max(losses), max(diss)]}

            if if_train:
                for i, item in enumerate(losses):
                    self.writer.add_scalar("valid_train/loss_:"+str(i),  item , epoch)
                    self.writer.add_scalar("valid_trains/acc_:"+str(i),  accs[i], epoch)
                    self.writer.add_scalar("valid_trains/auc_:"+str(i),  aucs[i], epoch)
                    self.writer.add_scalar("valid_trains/disparity_:"+str(i),  diss[i], epoch)
                    self.writer.add_scalar("valid_trains/pred_disparity_:"+str(i),  pred_diss[i], epoch)

            else:
                for i, item in enumerate(losses):
                    self.writer.add_scalar("valid_test/loss_:"+str(i),  item , epoch)
                    self.writer.add_scalar("valid_test/acc_:"+str(i),  accs[i], epoch)   
                    self.writer.add_scalar("valid_test/auc_:"+str(i),  aucs[i], epoch) 
                    self.writer.add_scalar("valid_test/disparity_:"+str(i),  diss[i], epoch)   
                    self.writer.add_scalar("valid_test/pred_disparity_:"+str(i),  pred_diss[i], epoch)
            return losses, accs, diss, pred_diss, aucs
   

    def soften_losses(self, losses, delta):


        losses_list = torch.stack(losses)
        loss = torch.max(losses_list)

        alphas = F.softmax((losses_list - loss)/delta)
        alpha_without_grad = (Variable(alphas.data.clone(), requires_grad=False)) 
        return alpha_without_grad, loss


    def train(self):
        
        if self.baseline_type == "none":
            if self.policy == "alternating":
                start_epoch = self.global_epoch
                for epoch in range(start_epoch , self.max_epoch1 + self.max_epoch2):
                    if int(epoch/self.per_epoches) %2 == 0:
                        self.train_stage1(epoch)
                    else:
                        self.train_stage2(epoch)

                    # if self.uniform:
                    #     pass
                    # else:
                    #     self.performence_only = bool(1-self.performence_only)

            elif self.policy == "two_stage":
                if self.uniform:
                    self.performence_only  = True
                else:
                    self.performence_only  = False
                start_epoch = self.global_epoch
                for epoch in range(start_epoch, self.max_epoch1):
                    self.train_stage1(epoch)

                for epoch in range(self.max_epoch1, self.max_epoch2 + self.max_epoch1):
                    self.train_stage2(epoch)


        elif self.baseline_type == "fedave_fair":
            start_epoch = self.global_epoch
            for epoch in range(start_epoch, self.max_epoch2 + self.max_epoch1):
                self.train_fed(epoch)



    def save_log(self):
        with open(os.path.join(self.log_pickle_dir, "train_log.pkl"), "wb") as f:
            pickle.dump(self.log_train, f)
        with open(os.path.join(self.log_pickle_dir, "test_log.pkl"), "wb") as f:
            pickle.dump(self.log_test, f)    
        os.makedirs(self.done_dir, exist_ok = True) 
        self.logger.info("logs have been saved")   



    def train_fed(self, epoch):
        # start_epoch = self.global_epoch
        # for epoch in range(start_epoch, self.max_epoch1):
        # # scheduler.step()

        self.model.train()
        self.optim.zero_grad()
        losses_data = []
        disparities_data = []
        pred_disparities_data = []
        accs_data = []
        aucs_data = []
        client_losses = []
        client_disparities = []
        for client_idx in range(self.n_clients):
            try:
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            except StopIteration:
                self.iter_train_clients[client_idx] = enumerate(
                    self.client_train_loaders[client_idx])
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            X = X.float()
            Y = Y.float()
            A = A.float()
            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
                A = A.cuda()

            loss, acc, auc, pred_dis, dis, pred_y = self.model(X, Y, A)


############################################################## GPU version

            client_losses.append(loss)
            client_disparities.append(torch.abs(pred_dis))
            losses_data.append(loss.item())
            disparities_data.append(dis)
            pred_disparities_data.append(pred_dis.item())
            accs_data.append(acc)
            aucs_data.append(auc)

        loss_max_performance = max(losses_data)
        loss_max_disparity = disparities_data[np.argmax(np.abs(disparities_data))]
        self.logger.info("fedave_fair, epoch: {}, all client loss: {}, all pred client disparities: {}, all client disparities: {}, all client accs: {}, all client aucs: {},  all max loss: {}".format(
                    self.global_epoch, losses_data, pred_disparities_data, disparities_data, accs_data, aucs_data,  [loss_max_performance, loss_max_disparity]))
        
        self.log_train[str(epoch)] = {"stage": 1, "client_losses": losses_data, "pred_client_disparities": pred_disparities_data, "client_disparities": disparities_data,
                                      "client_accs": accs_data, "client_aucs": aucs_data, "max_losses": [loss_max_performance, loss_max_disparity]}

        for i, loss in enumerate(losses_data):
            self.writer.add_scalar("train/1_loss_" + str(i), loss, epoch)
            self.writer.add_scalar(
                "train/disparity_" + str(i), disparities_data[i], epoch)
            self.writer.add_scalar(
                "train/pred_disparity_" + str(i), pred_disparities_data[i], epoch)
            self.writer.add_scalar(
                "train/acc_" + str(i), accs_data[i], epoch)
            self.writer.add_scalar(
                "train/auc_" + str(i), aucs_data[i], epoch)


        self.optim.zero_grad()
        weighted_loss1 = torch.sum(torch.stack(client_losses))
        weighted_loss2 = torch.sum(torch.stack(client_disparities)) * self.weight_fair
        weighted_loss = weighted_loss1 + weighted_loss2
        weighted_loss.backward()
        self.optim.step()

        # 2. apply gradient dierctly
        ############################
        # grads_and_vars = opt.compute_gradients(loss, parameter_list)
        # my_grads_and_vars = [(g*C, v) for g, v in grads_and_vars]
        # opt.apply_gradients(my_grads_and_vars)

        # Calculate and record performance
        if epoch == 0 or (epoch + 1) % self.eval_epoch == 0:
            self.model.eval()
            losses, accs, client_disparities, pred_dis, aucs = self.valid_stage1(
                if_train=False, epoch=epoch)
            if epoch != 0:
                self.model_save()
        self.global_epoch += 1



    def train_stage1(self, epoch):
        # start_epoch = self.global_epoch
        # for epoch in range(start_epoch, self.max_epoch1):
        # # scheduler.step()

        self.model.train()
        self.optim.zero_grad()
        grads_performance = []
        grads_disparity = []
        losses_data = []
        disparities_data = []
        pred_disparities_data = []
        accs_data = []
        aucs_data = []
        client_losses = []
        client_disparities = []
        for client_idx in range(self.n_clients):
            try:
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            except StopIteration:
                self.iter_train_clients[client_idx] = enumerate(self.client_train_loaders[client_idx])
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            X = X.float()
            Y = Y.float()
            A = A.float()
            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
                A = A.cuda()

            loss, acc, auc, pred_dis, dis, pred_y = self.model(X, Y, A)
                
 


############################################################## GPU version
            loss.backward(retain_graph=True)
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.extend(Variable(param.grad.data.clone().flatten(), requires_grad=False)) 
            grad = torch.stack(grad)
            grads_performance.append(grad)
            self.optim.zero_grad()



            torch.abs(pred_dis).backward(retain_graph=True)
            if self.performence_only:
                self.optim.zero_grad()
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.extend(Variable(param.grad.data.clone().flatten(), requires_grad=False)) 
            grad = torch.stack(grad)
            grads_disparity.append(grad)
            self.optim.zero_grad()   

            if self.uniform_eps:
                client_disparities.append(torch.abs(pred_dis))
                specific_eps = 0

            else:
                if self.dataset == "adult" and self.disparity_type == "DP" and self.sensitive_attr == "sex":
                    specific_eps = self.weight_eps * \
                        np.array([0.12938917 , 0.14046744])

                elif self.dataset == "adult" and self.disparity_type == "DP" and self.sensitive_attr == "race":
                    specific_eps = self.weight_eps * \
                        np.array([0.15663486, 0.07555133])

                elif self.dataset == "adult" and self.disparity_type == "Eoppo" and self.sensitive_attr == "race":
                    specific_eps = self.weight_eps * \
                        np.array([0.13454545, 0.09585903])

                elif self.dataset == "adult" and self.disparity_type == "Eoppo" and self.sensitive_attr == "sex":
                    specific_eps = self.weight_eps * \
                        np.array([0.00064349,  0.11690249])
                
                elif "eicu_los" in self.dataset and self.disparity_type == "DP" and self.sensitive_attr == "race":
                    specific_eps = self.weight_eps * np.array([0.052, 0.22, 0.035, 0.070, 0.094, 0.008, 0.047, 0.089, 0.078, 0.008, 0.108])

                elif "eicu_los" in self.dataset and self.disparity_type == "Eoppo" and self.sensitive_attr == "race":
                    specific_eps = self.weight_eps * np.array([0.187, 0.297, 0.021, 0.020, 0.103, 0.170, 0.138, 0.029, 0.065, 0.027, 0.087])


                client_disparities.append(torch.abs(pred_dis) - specific_eps[client_idx])


            client_losses.append(loss)
            losses_data.append(loss.item())
            disparities_data.append(dis)
            pred_disparities_data.append(pred_dis.item())
            accs_data.append(acc)
            aucs_data.append(auc)


        alphas_l, loss_max_performance = self.soften_losses(client_losses, self.deltas[0])
        loss_max_performance = loss_max_performance.item()
        alphas_g, loss_max_disparity = self.soften_losses(client_disparities, self.deltas[1])
        loss_max_disparity = loss_max_disparity.item()

        losses = np.array(losses_data)
            # a batch of [loss_c1, loss_c2, ... loss_cn], [grad_c1, grad_c2, grad_cn]
        if self.FedAve:
            preference = np.array([1 for i in range(self.n_clients)])
            alpha = preference / preference.sum()
            self.n_linscalar_adjusts += 1

        else:
            try:
                    # Calculate the alphas from the LP solver
                alphas_l = alphas_l.view(1, -1)
                grad_l = alphas_l @ torch.stack(grads_performance)
                alphas_g = alphas_g.view(1, -1)
                grad_g = alphas_g @  torch.stack(grads_disparity)
                alpha, deltas = self.hco_lp.get_alpha([loss_max_performance, loss_max_disparity], grad_l, grad_g, self.deltas, self.factor_delta, self.lr_delta)
                if torch.cuda.is_available():
                    alpha = torch.from_numpy(alpha.reshape(-1)).cuda()
                else:
                    alpha = torch.from_numpy(alpha.reshape(-1))
                self.deltas = deltas

                alpha = alpha.view(-1)
            except Exception as e:
                print(e)
                exit()
############################################################## GPU version

        self.logger.info("1, epoch: {}, all client loss: {}, all pred client disparities: {}, all client disparities: {}, all client accs: {}, all client aucs: {},  all max loss: {}, specific eps: {},  all Alpha: {}, all Deltas: {}".format(self.global_epoch, losses_data, pred_disparities_data, disparities_data, accs_data, aucs_data,  [loss_max_performance, loss_max_disparity] , specific_eps, alpha.cpu().numpy(), self.deltas))
        self.log_train[str(epoch)] = { "stage": 1, "client_losses": losses_data, "pred_client_disparities": pred_disparities_data, "client_disparities": disparities_data, "client_accs": accs_data, "client_aucs": aucs_data, "max_losses": [loss_max_performance, loss_max_disparity], "alpha": alpha.cpu().numpy(), "deltas": self.deltas}

        for i, loss in enumerate(losses_data):
            self.writer.add_scalar("train/1_loss_" + str(i), loss, epoch)
            self.writer.add_scalar("train/1_disparity_" + str(i), disparities_data[i], epoch)
            self.writer.add_scalar("train/1_pred_disparity_" + str(i), pred_disparities_data[i], epoch)
            self.writer.add_scalar("train/1_acc_" + str(i), accs_data[i], epoch)
            self.writer.add_scalar("train/1_auc_" + str(i), aucs_data[i], epoch)

        for i, a in enumerate(alpha):
            self.writer.add_scalar("train/1_alpha_" +str(i), a.item(), epoch)
        for i, delta in enumerate(self.deltas):
            self.writer.add_scalar("train/1_delta_" + str(i), delta, epoch)

            # 1. Optimization step
            # self.optim.zero_grad()
            # weighted_loss = torch.sum(torch.stack(client_losses) * alpha)  # * 5. * max(epo_lp.mu_rl, 0.2)
            # weighted_loss.backward()
            # self.optim.step() 

            # 2. Optimization step
        self.optim.zero_grad()
        weighted_loss1 = torch.sum(torch.stack(client_losses)*alphas_l)
        weighted_loss2 = torch.sum(torch.stack(client_disparities)*alphas_g)
        weighted_loss = torch.sum(torch.stack([weighted_loss1, weighted_loss2]) * alpha)
        weighted_loss.backward()
        self.optim.step()

        # 2. apply gradient dierctly
        ############################
        # grads_and_vars = opt.compute_gradients(loss, parameter_list)
        # my_grads_and_vars = [(g*C, v) for g, v in grads_and_vars]
        # opt.apply_gradients(my_grads_and_vars)

        # Calculate and record performance
        if epoch == 0 or (epoch + 1) % self.eval_epoch == 0:
            self.model.eval()
            losses, accs, client_disparities, pred_dis, aucs = self.valid_stage1(if_train = False, epoch = epoch)
            if epoch != 0:
                self.model_save()   
        self.global_epoch+=1



    def train_stage2(self, epoch):
        self.model.train()
        grads_performance = []
        grads_disparity = []
        disparities_data = []
        client_losses = []
        client_disparities = []
        losses_data = []
        accs_data = []
        pred_diss_data = []
        aucs_data = []

        for client_idx in range(self.n_clients):

            try:
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            except StopIteration:
                self.iter_train_clients[client_idx] = enumerate(self.client_train_loaders[client_idx])
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            X = X.float()
            Y = Y.float()
            A = A.float()
            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
                A = A.cuda()

            loss, acc, auc, pred_dis, dis, pred_y = self.model(X, Y, A)

            loss.backward(retain_graph=True)
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.extend(Variable(param.grad.data.clone().flatten(), requires_grad=False)) 
            grad = torch.stack(grad)
            grads_performance.append(grad)
            self.optim.zero_grad()


            torch.abs(pred_dis).backward(retain_graph=True)
            if self.performence_only:
                self.optim.zero_grad() 
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.extend(Variable(param.grad.data.clone().flatten(), requires_grad=False)) 
            grad = torch.stack(grad)
            grads_disparity.append(grad)
            self.optim.zero_grad()  

            client_losses.append(loss)
            client_disparities.append(torch.abs(pred_dis))
            disparities_data.append(dis) 
            accs_data.append(acc)
            losses_data.append(loss.item())
            pred_diss_data.append(pred_dis.item())
            aucs_data.append(auc)


        alpha_disparity, max_disparity = self.soften_losses(client_disparities, self.deltas[1])

        client_pred_disparity = torch.sum(alpha_disparity * torch.stack(client_disparities))

        grad_disparity = alpha_disparity.view(1, -1) @ torch.stack(grads_disparity)
        grads_performance = torch.stack(grads_performance)

        
        if max_disparity.item() < self.eps[0]:
            grad_disparity = torch.zeros_like(grad_disparity, requires_grad= False)
        grad_performance = torch.mean(grads_performance, dim = 0, keepdim=True)

        grads = torch.cat((grads_performance, grad_disparity), dim = 0)

##########################################GPU()
        grad_performance = grad_performance.t()
        ###

        alpha, gamma = self.po_lp.get_alpha(grads, grad_performance, grads.t())
        if torch.cuda.is_available():
            alpha = torch.from_numpy(alpha.reshape(-1)).cuda()
        else:
            alpha = torch.from_numpy(alpha.reshape(-1))
##########################################GPU()

        client_losses.append(client_pred_disparity)
        weighted_loss = torch.sum(torch.stack(client_losses) * alpha)
        weighted_loss.backward()
        self.optim.step()
        self.optim.zero_grad()

        self.logger.info("2, epoch: {}, all client loss: {}, all pred client disparities: {}, all client disparities: {}, all client accs: {}, all client aucs: {},  max disparity: {},  alpha: {}, deltas: {}".format(epoch,         
        losses_data, pred_diss_data, disparities_data, accs_data, aucs_data, max_disparity.item(), alpha.cpu().numpy(), self.deltas))
        
        self.log_train[str(epoch)] = { "stage": 2, "client_losses": losses_data, "pred_client_disparities": pred_diss_data, "client_disparities": disparities_data, "client_accs": accs_data, "client_aucs": aucs_data, "max_losses": [max(losses_data), max(disparities_data)], "alpha": alpha.cpu().numpy(), "deltas": self.deltas}

        for i, loss in enumerate(losses_data):
            self.writer.add_scalar("train/2_loss_" + str(i), loss, epoch)
            self.writer.add_scalar("train/2_disparity_" + str(i), disparities_data[i], epoch)
            self.writer.add_scalar("train/2_pred_disparity_" + str(i), pred_diss_data[i], epoch)
            self.writer.add_scalar("train/2_acc_" + str(i), accs_data[i], epoch)
            self.writer.add_scalar("train/2_auc_" + str(i), aucs_data[i], epoch)

        if epoch == 0 or (epoch + 1) % self.eval_epoch == 0:
            self.model.eval()
            losses, accs, client_disparities, pred_dis, aucs = self.valid_stage1(if_train = False, epoch = epoch)
            if epoch != 0:
                self.model_save()   
        self.global_epoch+=1

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from latex_utils import latexify
from utils import setup_seed, construct_log, get_random_dir_name
from hco_model import MODEL
import time
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
# parser.add_argument('--other_comment', type = str, default="the_non_phd", help="the dim of the solution")
parser.add_argument('--use_saved_args', type = bool, default=False, help="the dim of the solution")
parser.add_argument('--exp-dir', type = str, default="", help="the dim of the solution")
parser.add_argument('--FedAve', type = bool, default=False, help="the dim of the solution")
parser.add_argument('--target_dir_name', type = str, default="", help="the dim of the solution")
parser.add_argument('--commandline_file', type = str, default="results/args.json", help="the dim of the solution")
parser.add_argument('--eps_g', type = float, default=0.1, help="max_epoch for unbiased_moe")
parser.add_argument('--weight_eps', type=float, default=0.5,
                    help="eps weight for specific eps")
parser.add_argument('--uniform_eps', action="store_true", help="max_epoch for unbiased_moe")
parser.add_argument('--eps_delta_l', type = float, default=1e-4, help="max_epoch for predictor")
parser.add_argument('--eps_delta_g', type = float, default=1e-4, help="iteras for printing the loss info")
parser.add_argument('--factor_delta', type = float, default=0.1, help="max_epoch for unbiased_moe")
parser.add_argument('--lr_delta', type = float, default=0.01, help="max_epoch for predictor")
parser.add_argument('--delta_l', type = float, default=0.5, help="max_epoch for predictor")
parser.add_argument('--delta_g', type = float, default=0.5, help="max_epoch for predictor")
parser.add_argument('--step_size', type = float, default=0.01, help="iteras for printing the loss info")
parser.add_argument('--max_epoch_stage1', type = int, default=800, help="iteras for printing the loss info")
parser.add_argument('--max_epoch_stage2', type = int, default=800, help="iteras for printing the loss info")
parser.add_argument('--per_epoches', type = int, default=50, help="iteras for printing the loss info")
parser.add_argument('--eval_epoch', type = int, default=20, help="iteras for printing the loss info")
parser.add_argument('--grad_tol', type = float, default=1e-4, help="iteras for printing the loss info")
parser.add_argument('--ckpt_dir', type = str, default= "results/models", help="iteras for printing the loss info")
parser.add_argument('--log_dir', type = str, default= "results", help="iteras for printing the loss info")
parser.add_argument('--log_name', type = str, default= "log", help="iteras for printing the loss info")
parser.add_argument('--board_dir', type = str, default= "results/board", help="iteras for printing the loss info")
parser.add_argument('--store_xs', type = bool, default=False, help="iteras for printing the loss info")
parser.add_argument('--seed', type = int, default=1, help="iteras for printing the loss info")
parser.add_argument('--batch_size', type = list, default=[100, 100], help="iteras for printing the loss info")
parser.add_argument('--shuffle', type = bool, default=True, help="iteras for printing the loss info")
parser.add_argument('--drop_last', type = bool, default=False, help="iteras for printing the loss info")
parser.add_argument('--data_dir', type = str, default="data", help="iteras for printing the loss info")
parser.add_argument('--dataset', type = str, default="adult", help="[adult, eicu_d, eicu_los]")
parser.add_argument('--load_epoch', type = str, default=0, help="iteras for printing the loss info")
parser.add_argument('--global_epoch', type = int, default=0, help="iteras for printing the loss info")
parser.add_argument('--num_workers', type = int, default=0, help="iteras for printing the loss info")
parser.add_argument('--n_feats', type = int, default=10, help="iteras for printing the loss info")
parser.add_argument('--n_hiddens', type = int, default=40, help="iteras for printing the loss info")
parser.add_argument('--sensitive_attr', type = str, default="race", help="iteras for printing the loss info")
parser.add_argument('--valid', type = bool, default=False, help="iteras for printing the loss info")
parser.add_argument('--policy', type = str, default="two_stage", help="[alternating, two_stage]")
parser.add_argument('--uniform', action="store_true",  help="uniform mode, without any fairness contraints")
parser.add_argument('--disparity_type', type= str, default= "DP",  help="uniform mode, without any fairness contraints")
parser.add_argument('--baseline_type', type= str, default= "none",  help="fedave_fair, individual_fair")
parser.add_argument('--weight_fair', type= float, default= 1.0,  help="weight for disparity")
args = parser.parse_args()


args.eps = [args.eps_g, args.eps_delta_l, args.eps_delta_g]
args.train_dir = os.path.join(args.data_dir, args.dataset, "train")
args.test_dir = os.path.join(args.data_dir, args.test, "test")
args.ckpt_dir = os.path.join(args.target_dir_name, args.ckpt_dir)
args.log_dir = os.path.join(args.target_dir_name, args.log_dir)
args.board_dir = os.path.join(args.target_dir_name, args.board_dir)
args.done_dir = os.path.join(args.target_dir_name, "done")
args.commandline_file = os.path.join(args.target_dir_name, args.commandline_file)





if __name__ == '__main__':

    writer = SummaryWriter(log_dir = args.board_dir)
    if args.use_saved_args:
        with open(args.commandline_file, "r") as f:
            args.__dict__ = json.load(f)
    else:
        pass
    os.makedirs(args.log_dir, exist_ok = True)
    os.system("cp *.py " + args.target_dir_name)
    logger = construct_log(args)
    setup_seed(seed = args.seed)
    model = MODEL(args, logger, writer)
    if args.valid:
        losses, accs, diss, pred_diss = model.valid_stage1(False, args.max_epoch_stage1)
    else:
        model.train()
    model.save_log()


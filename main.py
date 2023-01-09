import torch
import argparse
import os
import scipy.sparse as sp
import numpy as np

from dataset import TwitterDataset, BillDataset
from model_trainer import InfoVGAETrainer
from config import update_arg_with_config_name
from evaluate import Evaluator

parser = argparse.ArgumentParser()
# Quick experiment, no need to specify other parameters after using config
parser.add_argument('--config_name', type=str, default=None, help="Use existing config to reproduce experiment quickly")

# General
parser.add_argument('--model', type=str, default="InfoVGAE", help="model to use")
parser.add_argument('--epochs', type=int, default=500, help='epochs (iterations) for training')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate of model')
parser.add_argument('--device', type=str, default="0", help='cpu/gpu device')
parser.add_argument('--num_process', type=int, default=40, help='num_process for pandas parallel')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--use_cuda', action="store_true", help='whether to use cuda device')

# Data
parser.add_argument('--dataset', type=str, help='dataset to use')
parser.add_argument('--add_self_loop', type=bool, default=True, help='add self loop for adj matrix')
parser.add_argument('--directed', type=bool, default=False, help='use directed adj matrix')
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--data_json_path', type=str, default=None)
parser.add_argument('--follow_path', type=str, default=None)
parser.add_argument('--use_follow', type=bool, default=False)
parser.add_argument('--stopword_path', type=str, default=None)
parser.add_argument('--keyword_path', type=str, default="N")
parser.add_argument('--kthreshold', type=int, default=5, help='minimum keyword count to keep the sample')
parser.add_argument('--uthreshold', type=int, default=3, help='minimum user tweet count to keep the sample')

# For GAE/VGAE model
parser.add_argument('--hidden1_dim', type=int, default=32, help='graph conv1 dim')
parser.add_argument('--hidden2_dim', type=int, default=2, help='graph conv2 dim')
parser.add_argument('--use_feature', type=bool, default=True, help='Use feature')
parser.add_argument('--num_user', type=int, default=None, help='Number of users, usually no need to specify.')
parser.add_argument('--num_assertion', type=int, default=None, help='Number of assertions, usually no need to specify.')
parser.add_argument('--pos_weight_lambda', type=float, default=1.0, help='Lambda for positive sample weight')

# For Discriminator
parser.add_argument('--gamma', type=float, default=1e-3, help='weight for tc loss')
parser.add_argument('--lr_D', type=float, default=1e-3, help='learning rate for discriminator')
parser.add_argument('--beta1_D', type=float, default=0.5, help='beta1 for discriminator optimizer')
parser.add_argument('--beta2_D', type=float, default=0.9, help='beta2 for discriminator optimizer')

# Result
parser.add_argument('--output_path', type=str, default="./output", help='Path to save the output')

args = parser.parse_args()

# Setting the device
if torch.cuda.is_available() and args.device != "":
    device = "cuda:{}".format(args.device)
else:
    device = "cpu"
setattr(args, "device", device)
print("Device: {}".format(device))

# Update the arg if config_name is set
if args.config_name is not None:
    update_arg_with_config_name(args, args.config_name, phrase="train")
os.makedirs(args.output_path, exist_ok=True)

# Setting the random seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Prepare dataset
if args.config_name.find("_bill_") != -1:
    # Use this for bill
    dataset = BillDataset(args)
    adj_matrix = dataset.build()
    setattr(args, "num_user", dataset.num_user)
    setattr(args, "num_assertion", dataset.num_assertion)
    dataset.dump_label()
else:
    dataset = TwitterDataset(csv_path=args.data_path,
                            keyword_path=args.keyword_path, stopword_path=args.stopword_path,
                            mode="multiply", kthreshold=args.kthreshold,
                            uthreshold=args.uthreshold, num_process=args.num_process,
                            add_self_loop=True, directed=False, args=args)
    adj_matrix = dataset.build()
    setattr(args, "num_user", dataset.num_user)
    setattr(args, "num_assertion", dataset.num_assertion)
    # dump label and namelist for evaluation
    dataset.dump_label()

# Start Training
feature = sp.diags([1.0], shape=(dataset.num_nodes, dataset.num_nodes))
setattr(args, "input_dim", dataset.num_nodes)
trainer = InfoVGAETrainer(adj_matrix, feature, args, dataset)
trainer.train()
trainer.save()

# Start Evaluation (Twitter dataset)
print("Running Evaluation ...")
evaluator = Evaluator()
evaluator.init_from_value(trainer.result_embedding, dataset.user_label, dataset.asser_label,
                          dataset.name_list, dataset.asserlist,
                          output_dir=args.output_path)
evaluator.plot(show=False, save=True)
evaluator.run_clustering()
evaluator.plot_clustering(show=False)
evaluator.numerical_evaluate(verbose=False)
evaluator.dump_topk_json()

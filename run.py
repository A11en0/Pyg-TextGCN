import gc
import torch
import argparse
from model import PygGCN, GAT
from train_eval import TextGCNTrainer
from utils import return_seed, LoadData

parser = argparse.ArgumentParser(description='TextGCN')
parser.add_argument('--model', type=str, default='TextGCN', help='choose a model')
args = parser.parse_args()


def run(dataset, times):
    args.dataset = dataset
    # args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.device = torch.device('cpu')
    args.nhid = 100
    args.max_epoch = 200
    args.dropout = 0.5
    args.val_ratio = 0.1
    args.early_stopping = 10
    args.lr = 0.02
    model = PygGCN
    # model = GAT
    print(args)

    predata = LoadData(args)
    seed_lst = list()
    for ind, seed in enumerate(return_seed(times)):
        # print(f"\n\n==> {ind}, seed:{seed}")
        args.seed = seed
        seed_lst.append(seed)

        framework = TextGCNTrainer(model=model, args=args, pre_data=predata)
        framework.fit()

        # framework.test()
        # del framework
        # gc.collect()
        #
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

    # print("==> seed set:")
    # print(seed_lst)

if __name__ == '__main__':
    # run("mr", 1)
    run("R8", 1)


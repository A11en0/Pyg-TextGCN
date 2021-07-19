import torch
import numpy as np
from time import time
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from utils import get_time_dif


class TextGCNTrainer:
    def __init__(self, args, model, pre_data):
        self.args = args
        self.model = model
        self.device = args.device
        self.max_epoch = self.args.max_epoch
        self.set_seed()
        self.dataset = args.dataset
        self.predata = pre_data
        self.earlystopping = EarlyStopping(args.early_stopping)
        self.criterion = torch.nn.CrossEntropyLoss()

    def set_seed(self):
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

    def fit(self):
        self.prepare_data()
        self.model = self.model(nfeat=self.predata.nfeat_dim,
                                nhid=self.args.nhid,
                                nclass=self.nclass,
                                dropout=self.args.dropout)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        print(self.model.parameters)
        self.model_param = sum(param.numel() for param in self.model.parameters())
        print('model parameters:', self.model_param)
        self.convert_tensor()
        self.train()
        self.test()

    def prepare_data(self):
        self.target = self.predata.target
        self.nclass = self.predata.nclass
        self.data = self.predata.graph

        self.train_lst, self.val_lst = train_test_split(self.predata.train_lst,
                                                        test_size=self.args.val_ratio,
                                                        shuffle=True,
                                                        random_state=self.args.seed)
        self.test_lst = self.predata.test_lst

    def convert_tensor(self):
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.target = torch.tensor(self.target).long().to(self.device)
        self.train_lst = torch.tensor(self.train_lst).long().to(self.device)
        self.val_lst = torch.tensor(self.val_lst).long().to(self.device)

    def train(self):
        start_time = time()
        for epoch in range(self.max_epoch):
            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model.forward(self.data)
            loss = self.criterion(logits[self.train_lst],
                                  self.target[self.train_lst])

            loss.backward()
            self.optimizer.step()
            pred = torch.max(logits[self.train_lst].data, 1)[1].cpu().numpy()
            target = self.target[self.train_lst].data.cpu().numpy()
            train_acc = accuracy_score(pred, target)
            val_loss, val_acc, val_f1 = self.val(self.val_lst)
            time_dif = get_time_dif(start_time)
            msg = 'Epoch: {:>2},  Train Loss: {:>6.3}, Train Acc: {:>6.2%}, Val Loss: {:>6.3},  Val Acc: {:>6.2%},  Time: {}'
            print(msg.format(epoch, loss.item(), train_acc, val_loss, val_acc, time_dif))
            if self.earlystopping(val_loss):
                break

    @torch.no_grad()
    def val(self, x, test=False):
        self.model.eval()
        with torch.no_grad():
            logits = self.model.forward(self.data)
            loss = self.criterion(logits[x],
                                  self.target[x])

            pred = torch.max(logits[x].data, 1)[1].cpu().numpy()
            target = self.target[x].data.cpu().numpy()
            acc = accuracy_score(pred, target)
            f1 = f1_score(pred, target, average='macro')
        if test:
            report = metrics.classification_report(pred, target, digits=4)
            # report = metrics.classification_report(pred, target, target_names=config.class_list, digits=4)
            confusion = metrics.confusion_matrix(pred, target)
            return acc, report, confusion
        return loss.item(), acc, f1

    @torch.no_grad()
    def test(self):
        self.test_lst = torch.tensor(self.test_lst).long().to(self.device)
        acc, report, confusion = self.val(self.test_lst, test=True)
        msg = '\nTest Acc: {:>6.2%}'
        print(msg.format(acc))
        print("Precision, Recall and F1-Score...")
        print(report)
        print("Confusion Matrix...")
        print(confusion)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_path = "hdd_data/prepare_dataset/model/model.pt"

    def __call__(self, val_loss, model=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_path)
        self.val_loss_min = val_loss

    def load_model(self):
        return torch.load(self.model_path)
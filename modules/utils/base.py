import torch,torch.backends.cudnn
import dgl
import os
import random,copy
import numpy as np

def setup_seed(seed, cuda):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if cuda is True:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


# author: Bjarten
# url: https://github.com/Bjarten/early-stopping-pytorch
# MIT license
# modified
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, trace_func=None):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.info_dict = None
        self.trace_func = trace_func
    def __call__(self, score, info_dict=None):
        if self.best_score is None:
            self.best_score = score
            self.info_dict=copy.deepcopy(info_dict)
        elif score < self.best_score:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}') if self.trace_func else None
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.info_dict=copy.deepcopy(info_dict)
            self.counter = 0
    def couldStop(self):return self.early_stop
pass

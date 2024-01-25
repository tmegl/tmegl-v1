from logging import Logger
from typing import Tuple
from torch.utils.data import Dataset
import torch
import torch, numpy as np, sys
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from tqdm import tqdm,trange
from dgl.dataloading import GraphDataLoader
from modules.utils import EarlyStopping
from numba import jit
# =============================================================================

def test(model:torch.nn.Module,dataset:Dataset,batchsize:int,device:str,withAUC=False)->Tuple[np.float32,np.float32,np.float32]:
    model.eval()
    model=model.to(device=device)
    survtime_all, lbl_all, lbl_pred_all = torch.tensor([]), torch.tensor([]), torch.tensor([]).to(device=device)
    loss_nn_sum = 0
    dataloader = GraphDataLoader(dataset=dataset, batch_size=batchsize, shuffle=False)
    for tags,lbl,survtime,gs in dataloader:
        gs=gs.to(device)
        with torch.no_grad():
            _,_,lbl_pred,_=model(gs)
        survtime_all = torch.cat([survtime_all, survtime])
        lbl_all = torch.cat([lbl_all, lbl])
        lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
        current_batch_len = len(survtime)
        R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_matrix_train[i, j] = survtime[j] >= survtime[i]
        test_R = torch.FloatTensor(R_matrix_train).to(device)
        test_ystatus = lbl.to(device)
        theta = lbl_pred.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_nn = -torch.mean((theta - torch.log(torch.sum(exp_theta * test_R, dim=1))) * test_ystatus.float())
        loss_nn_sum = loss_nn_sum + loss_nn.data.item()
    c_index = CIndex(
        lbl_pred_all.detach().cpu().numpy(), 
        lbl_all.detach().cpu().numpy(), 
        survtime_all.detach().cpu().numpy()
    )
    auc=AUC(
        lbl_pred_all.detach().cpu().numpy(),
        lbl_all.detach().cpu().numpy(),
        survtime_all.detach().cpu().numpy()
    ) if withAUC else None
    return c_index,auc,loss_nn_sum

def train(train_set:Dataset, epochs:int, lr:float, batchsize:int, device:str, model: torch.nn.Module, lambda_1:float,log:Logger,valid_set:Dataset|None=None, earlyStop=None)->Tuple[torch.nn.Module,int]:
    model.train()
    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    if earlyStop is None:
        earlyStop = EarlyStopping(patience=epochs,trace_func=None)
    else:
        earlyStop = EarlyStopping(patience=earlyStop)
    # train
    for epoch in trange(epochs,leave=False,ncols=50):
        # vars
        survtime_all, lbl_all, lbl_pred_all = torch.tensor([]), torch.tensor([]), torch.tensor([]).to(device)
        loss_nn_sum = 0
        model.train()
        dataloader = GraphDataLoader(dataset=train_set, batch_size=batchsize, shuffle=False)
        for tags, lbl, survtime, gs in dataloader:
            optimizer.zero_grad()
            _,_,lbl_pred,_ = model(gs.to(device))
            survtime_all = torch.cat([survtime_all, survtime])
            lbl_all = torch.cat([lbl_all, lbl])
            lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
            # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
            # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
            current_batch_len = len(survtime)
            R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
            for i in range(current_batch_len):
                for j in range(current_batch_len):
                    R_matrix_train[i, j] = survtime[j] >= survtime[i]
            train_R = torch.FloatTensor(R_matrix_train).to(device)
            train_ystatus = lbl.to(device)
            theta = lbl_pred.reshape(-1)
            exp_theta = torch.exp(theta)
            loss_nn = -torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus.float())

            l1_reg = None
            for W in model.parameters():
                if l1_reg is None:
                    l1_reg = torch.abs(W).sum()
                else:
                    l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)

            loss = loss_nn + lambda_1 * l1_reg
            loss_nn_sum = loss_nn_sum + loss_nn.data.item()
            # backward
            loss.backward()
            optimizer.step()
        
        message=[f"epoch:{epoch:4}"]
        if valid_set:
            c_index,auc,loss_nn_sum = test(model,valid_set,batchsize,device)
            message.append(f"[valid] c_index: {c_index:.5f}") if log else None
            earlyStop(score=c_index,info_dict={"model":model,"epoch":epoch})
        log.info("    ".join(message)) if log else None
        if earlyStop.couldStop()==True:
            break
        pass
    return earlyStop.info_dict["model"],earlyStop.info_dict["epoch"]

# =====================Survive Prediction======================================
@jit(nopython=True)
def AUC(Hazard:np.ndarray,Status:np.ndarray,Time:np.ndarray):
    total=0
    correct=0
    maxTime=max(Time)
    minTime=min(Time)
    lenTime=len(Time)
    for i in range(lenTime):# 
        if Time[i]!=maxTime and Time[i]!=minTime:
            for j in range(lenTime):
                for k in range(lenTime):
                    if j!=i and k!=i:
                        if Status[j]==1 and Time[j]<Time[i] and Time[k]>Time[i]:
                            total+=1
                            if Hazard[j]>Hazard[k]:
                                correct+=1
    return correct/total


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_cox(hazards, labels):
    # This accuracy is based on estimated survival events against true survival events
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    labels = labels.data.cpu().numpy()
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def cox_log_rank(hazards, labels, survtime_all):
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    survtime_all = survtime_all.data.cpu().numpy().reshape(-1)
    idx = hazards_dichotomize == 0
    labels = labels.data.cpu().numpy()
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return pvalue_pred

def CIndex(hazards:np.ndarray, labels:np.ndarray, survtime_all:np.ndarray):
    # labels = labels.data.cpu().numpy()
    concord = 0.0
    total = 0.0
    N_test = labels.shape[0]
    labels = np.asarray(labels, dtype=bool)
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total = total + 1
                    if hazards[j] < hazards[i]:
                        concord = concord + 1
                    elif hazards[j] == hazards[i]:
                        concord = concord + 0.5

    return concord / total


def CIndex_lifeline(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy()
    hazards = hazards.cpu().numpy().reshape(-1)
    survtime_all = survtime_all.data.cpu().numpy()
    return concordance_index(survtime_all, -hazards, labels)


def frobenius_norm_loss(a, b):
    loss = torch.sqrt(torch.sum(torch.abs(a - b) ** 2))
    return loss

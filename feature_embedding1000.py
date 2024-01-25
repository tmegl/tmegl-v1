#%%
import torch,dgl,torch.nn as nn,os
import numpy as np
from modules import utils as sut
from modules.dataset.dataset import BRCADataset
from modules.dataset import embedding
import argparse

#%%
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-c","--cuda",default=None,help="cuda:?")
    parser.add_argument("-s","--seed",default=666,type=int,help="seed")
    parser.add_argument("--dataset",required=True,help="xxx.bin")

    parser.add_argument("--train",action="store_true",help="embedding train")
    parser.add_argument("-e","--epoch",default=100,type=int,help="epoch")
    parser.add_argument("-l","--lr",type=float,default=0.0001,help="learning rate")
    parser.add_argument("--train_output",default="",help="save folder: ./xxx/")

    parser.add_argument("--test",action="store_true",help="embedding test")
    parser.add_argument("--test_model",default="",help="embedding test model")
    parser.add_argument("--test_output",default="",help="test save pth: xxx.bin")

    args=parser.parse_args()
    #====================================================================================
    if args.cuda is None:
        args.cuda="cpu"
    elif args.cuda.isdigit():
        args.cuda="cuda:"+args.cuda
    
    if args.train:
        if args.train_output=="":
            print("miss --train_output")
            exit
        print("[train] seed:{} device:{} epochs:{} lr:{}".format(args.seed,args.cuda,args.epoch,args.lr))
        sut.base.setup_seed(args.seed,True)
        print("load{}".format(args.dataset))
        datset=BRCADataset(graph_bin_path=args.dataset)
        save_path=args.train_output
        print("embedding model will save to:{}".format(save_path))
        os.makedirs(save_path,exist_ok=True)
        log=sut.setLogger(
            log_file=os.path.join(save_path,"embedding_train.log"),
            console=False
        )
        embedding.train(
            data_set=datset,
            device=args.cuda,
            epochs=args.epoch,
            save_path=save_path,
            lr=args.lr,
            lst_nets=[
                nn.Sequential(nn.Linear(1000,512),nn.ELU(),nn.Linear(512,128)),
                nn.Sequential(nn.Linear(1000,512),nn.ELU(),nn.Linear(512,128)),
                nn.Sequential(nn.Linear(1000,512),nn.ELU(),nn.Linear(512,128))
            ],
            log=log
        )
        print("model save to:{}".format(save_path))
    #=============================================================================
    if args.test:
        if args.test_model=="":
            print("miss --test_model")
            exit
        if args.test_output=="":
            assert(args.test_model.endswith(".pt"))
            args.test_output=args.test_model[:-3]+".pt"
            print("auto set output pth:{}".format(args.test_output))
            
        print("[test] seed:{} device:{}".format(args.seed,args.cuda))
        sut.base.setup_seed(args.seed,True)
        # embedding
        print("load {}".format(args.dataset))
        dataset=BRCADataset(graph_bin_path=args.dataset)
        print("load {}".format(args.test_model))
        lst_nets=torch.load(args.test_model)
        print("model will save to:{}".format(args.test_output))
        embedding.test(
            data_set=dataset,
            device=args.cuda,
            lst_nets=lst_nets,
            graph_save_path=args.test_output
        )
        print("model save to:{}".format(args.test_output))
        pass
        
        

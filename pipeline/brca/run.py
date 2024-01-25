import os,sys,numpy,torch,pandas as pd,argparse,json,time
import torch.nn as nn,torch.utils.data,numpy as np
from tqdm import trange
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../..")
from modules.dataset.dataset import BRCADataset
from modules.models.net import GruGatNets
from modules.utils import setup_seed,setLogger
from modules.models.train import train,test

parser=argparse.ArgumentParser()
parser.add_argument("-f","--fold",type=int,required=True,help="fold in (0,1,2,3,4)")
parser.add_argument("-d","--device",type=str,required=True,help="device: cpu|cuda:0|cuda:1|...")
parser.add_argument("-e","--epochs",type=int,required=True,help="epochs")
parser.add_argument("-s","--earlyStop",type=int,default=100000,help="earlyStop, set it very large to make it dont work if necessary")
args=parser.parse_args()

SEED=210
FOLD=args.fold
DEVICE=args.device
EPOCHS=args.epochs
earlyStop=args.earlyStop
PARAMS=json.loads(open("./params.json","r").read())['fold']
"""
file: params.json like that format:
{
    "fold":{
        "0":{
            "batchsize": 32,
            "lr": 1e-5,
            "lambda_1": 1e-6,
            "hiddens": [
                [128, 1, 0.8, 0.2, 0.2, 0.4, 5], [128, 1, 0.4, 0.5, 0.2, 0.2, 1], [128, 1, 0.6, 0.5, 0.3, 0.6, 3]
            ]
        },
        "1":{
            // same format like "0"
        },
        "2":{
            // same format like "0"
        },
        "3":{
            // same format like "0"
        },
        "4":{
            // same format like "0"
        }
    }
}
"""
BATCHSIZE=PARAMS[f"{FOLD}"]["batchsize"]
LR=PARAMS[f"{FOLD}"]["lr"]
LAMBDA_1=PARAMS[f"{FOLD}"]["lambda_1"]

def get_dataset_from_excel(data_set:BRCADataset,fold:int,excel_path:str):
    assert(fold>=0 and fold<=4)
    train_df=pd.read_excel(excel_path,sheet_name="train")
    valid_df=pd.read_excel(excel_path,sheet_name="valid")
    test_df=pd.read_excel(excel_path,sheet_name="test")

    train_ind=train_df[f"train_ind"].values
    valid_ind=valid_df[f"valid_ind"].values
    test_ind=test_df[f"test_ind"].values

    train_set=np.array(data_set, dtype="object")[train_ind].tolist()
    valid_set=np.array(data_set, dtype="object")[valid_ind].tolist()
    test_set = np.array(data_set, dtype="object")[test_ind].tolist()

    return train_set,valid_set,test_set

def define_model()->GruGatNets:
    hiddens=PARAMS[f"{FOLD}"]["hiddens"]
    return GruGatNets(
        input=128,
        hiddens=hiddens,
        classifier=nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()),
    )

def just_test(model_path:str):
    # fix seed
    setup_seed(SEED,True)
    # dataset
    train_set,valid_set,test_set=get_dataset_from_excel(
        data_set=BRCADataset(
            graph_bin_path="./dataset/embeded.bin",
            info_pkl_path="./dataset/info.pkl"
        ),
        fold=FOLD,
        excel_path=f"./dataset/fold/fold{FOLD}.xlsx"
    )
    # model
    model=define_model()
    model.load_state_dict(torch.load(model_path))
    # just test with the trained model
    setup_seed(SEED,True)
    test_c_index_pred,test_auc,_=test(
        model=model,
        dataset=test_set,
        batchsize=BATCHSIZE,
        device=DEVICE,
        withAUC=True
    )
    print(f"test cindex:{test_c_index_pred} auc:{test_auc}")
    
    pass

def run():
    setup_seed(SEED,True)
    train_set,valid_set,test_set=get_dataset_from_excel(
        data_set=BRCADataset(
            graph_bin_path="./dataset/embeded.bin",
            info_pkl_path="./dataset/info.pkl"
        ),
        fold=FOLD,
        excel_path=f"./dataset/fold/fold{FOLD}.xlsx"
    )
    # model
    model=define_model()
    # logger
    logger=setLogger(
        log_file=f"./log/fold{FOLD}.log",
        console=False,
        rewrite=True
    )
    logger.setLevel("INFO")
    # train
    setup_seed(SEED,True)
    _model,_=train(
        train_set=train_set,
        epochs=EPOCHS,
        batchsize=BATCHSIZE,
        lr=LR,
        lambda_1=LAMBDA_1,
        model=model,
        device=DEVICE,
        log=logger,
        earlyStop=earlyStop,
        valid_set=valid_set
    )
    os.makedirs("./model",exist_ok=True)
    torch.save(_model.state_dict(),f"./model/fold{FOLD}.pth")
    test_c_index,test_auc,_=test(
        model=_model,
        dataset=test_set,
        batchsize=BATCHSIZE,
        device=DEVICE,
        withAUC=True
    )
    msg=f"test cindex:{test_c_index} auc:{test_auc}"
    logger.info(msg)
    print(msg)


#================================================================================================
if os.path.exists(f"./model/fold{FOLD}.pth"):just_test(f"./model/fold{FOLD}.pth")
else:run()
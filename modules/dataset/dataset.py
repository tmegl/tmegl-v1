from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch, dgl, os
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import save_info, load_info
from torch.utils.data import Dataset
from scipy.io import loadmat
from scipy.spatial import distance
from tqdm.auto import tqdm,trange


def handleXls(excel_path: str, column: Optional[dict] = None) -> Tuple[list, list, list]:
    column = column if column is not None else {"tag": "tag", "lbl": "lbl", "survtime": "survtime"}
    xls = pd.read_excel(excel_path)
    xls[column["tag"]] = xls[column["tag"]].str.upper()
    xls = xls.dropna(subset=[column["lbl"]])
    xls[column["lbl"]] = np.int32(xls[column["lbl"]] != "alive")
    xls = xls[xls[column["survtime"]] > 0]
    return xls[column["tag"]].tolist(), xls[column["lbl"]].tolist(), xls[column["survtime"]].tolist()


class BRCADataset(Dataset):
    def __init__(self,data_folder=None,graph_bin_path=None,info_pkl_path=None):
        graph_bin_path=(os.path.join(data_folder,"graphs.bin") if data_folder else None) if graph_bin_path is None else graph_bin_path
        info_pkl_path=(os.path.join(data_folder,"info.pkl") if data_folder else None) if info_pkl_path is None else info_pkl_path
        self.graphs=[]
        self.lbls=[]
        self.survtimes=[]
        self.tags=[]
        self.ls=[]
        if graph_bin_path:
            graphs,labels=dgl.load_graphs(graph_bin_path)
            self.graphs=graphs
            self.lbls=labels["lbls"]
            self.survtimes=labels["survtimes"]
        if info_pkl_path:
            info=load_info(info_pkl_path)
            self.tags=info["tags"]
            self.ls=info["ls"]
        # fill with None
        if graph_bin_path is None:
            self.graphs=[None for i in range(len(self.tags))]
            self.lbls=[None for i in range(len(self.tags))]
            self.survtimes=[None for i in range(len(self.tags))]
        if info_pkl_path is None:
            self.tags=[None for i in range(len(self.lbls))]
            self.ls=[None for i in range(len(self.lbls))]

    def __getitem__(self, ind):
        return self.tags[ind],self.lbls[ind],self.survtimes[ind],self.graphs[ind]
    def __len__(self):return len(self.graphs)
    def info(self):
        print(f"nums:{len(self.tags)}")
        print(f"tags[:3]      : {self.tags[:3]}")
        print(f"lbls[:3]      : {self.lbls[:3]}")
        print(f"survtimes[:3] : {self.survtimes[:3]}")
        if len(self.tags)>0:
            print(f"graphs[0]     : {self.graphs[0]}")
    def save(self,graph_path:str=None,info_path:str=None):
        if graph_path is not None:
            os.makedirs(os.path.dirname(graph_path),exist_ok=True)
            dgl.save_graphs(
                filename=graph_path, 
                g_list=self.graphs, 
                labels={"lbls": self.lbls, "survtimes": self.survtimes}
            )
        if info_path is not None:
            os.makedirs(os.pardir(info_path),exist_ok=True)
            save_info(path=info_path, info={"tags": self.tags, "ls": self.ls})
        pass
    

pass
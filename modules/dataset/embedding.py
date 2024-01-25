# %%
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__),"../"))
import torch,dgl,numpy as np
import utils as sut
from .dataset import BRCADataset
from tqdm import tqdm
from torch import nn
from sklearn.cluster import SpectralClustering

# %%
def train(data_set:BRCADataset,device,epochs=10,lr=0.0001,lst_nets=None,log=None,save_interval=5,save_path=None):
    if lst_nets is None:
        lst_nets = [
            nn.Sequential(nn.Linear(128,128)),
            nn.Sequential(nn.Linear(128,128)),
            nn.Sequential(nn.Linear(128,128))
        ]
    for net in lst_nets:
        net=net.to(device=device)
    # loss
    kl_loss = torch.nn.KLDivLoss(reduction = "batchmean")
    # 
    trainers = [
        torch.optim.SGD(lst_nets[0].parameters(), lr=lr),
        torch.optim.SGD(lst_nets[1].parameters(), lr=lr),
        torch.optim.SGD(lst_nets[2].parameters(), lr=lr),
    ]
    def deleteDiagonalElements(matrix):
        l=len(matrix)
        ans=torch.tensor([]).to(device=device)
        for i in range(l):
            ans=torch.concat( (ans,torch.concat((matrix[i][0:i],matrix[i][i+1:]),dim=0).reshape(1,-1)),dim=0 )
        return ans
    pass
    def cluster_g(g: dgl.DGLGraph, n_clusters: int = 5):
        lst_center_nodes = [None for i in [0, 1, 2]]
        sample_loss=[0,0,0]
        for lst in [0, 1, 2]:
            # loss
            loss_in_array=torch.tensor([]).to(device=device)
            sub_g = dgl.node_subgraph(g, g.filter_nodes(lambda nodes: nodes.data['lst'] == lst)).to(device=device)
            # embedding
            sub_g.ndata['embedding'] = lst_nets[lst](sub_g.ndata['feats'])
            trainers[lst].zero_grad()
            Sim_G = sut.calc.cos_sim(sub_g.ndata['adjv'])
            sc = SpectralClustering(n_clusters=n_clusters,affinity="precomputed", random_state=128,n_init=1,degree=1)
            sc.fit(Sim_G.cpu())
            clusters_inds = [[] for i in range(n_clusters)]
            for ind, k in enumerate(sc.labels_):
                clusters_inds[k].append(ind)
            center_nodes = [None for i in range(n_clusters)]
            for k, node_inds in enumerate(clusters_inds):
                node_sim_sum = np.zeros_like(node_inds)
                node_sim_sum_max, center_node = -1, None
                for nodei in node_inds:
                    node_sim_sum = Sim_G[nodei][node_inds].sum()
                    if node_sim_sum_max < node_sim_sum:
                        node_sim_sum_max = node_sim_sum
                        center_node = nodei
                center_nodes[k] = center_node
                sub_g_by_cluster = dgl.node_subgraph(sub_g,nodes=node_inds).to(device=device)
                Sim_P_In = sut.calc.cos_sim(sub_g_by_cluster.ndata['adjv'])
                Sim_E_In = sut.calc.cos_sim(sub_g_by_cluster.ndata['embedding'])
                sub_center_node = None
                for _i,_v in enumerate(sub_g_by_cluster.ndata[dgl.NID].tolist()):
                    if _v == center_node:
                        sub_center_node = _i
                        break
                assert(sub_center_node is not None)
                real_Sim_E_In = torch.log_softmax(deleteDiagonalElements(Sim_E_In)[sub_center_node],dim=0)
                real_Sim_P_In = torch.softmax(deleteDiagonalElements(Sim_P_In)[sub_center_node],dim=0)
                loss_in = kl_loss(real_Sim_E_In,real_Sim_P_In)
                loss_in_array=torch.concat((loss_in_array,loss_in.reshape(1,-1)),dim=-1)

            lst_center_nodes[lst]=center_nodes

            sub_g_by_center_nodes=dgl.node_subgraph(sub_g,nodes=center_nodes)
            Sim_P_Between = sut.calc.cos_sim( sub_g_by_center_nodes.ndata['adjv'] )
            Sim_E_Between = sut.calc.cos_sim( sub_g_by_center_nodes.ndata['embedding'] )
            real_Sim_E_Between = torch.log_softmax(deleteDiagonalElements(Sim_E_Between),dim=1)
            real_Sim_P_Between = torch.softmax(deleteDiagonalElements(Sim_P_Between),dim=1)
            for i in range(len(real_Sim_E_Between)):
                loss_between = kl_loss(real_Sim_E_Between[i],real_Sim_P_Between[i])
                loss_in_array=torch.concat((loss_in_array,loss_between.reshape(-1,1)),dim=-1)
            pass

            __sum=loss_in_array.sum()
            if __sum.isnan():pass
            else:
                __sum.backward()
                trainers[lst].step()
                sample_loss[lst]=__sum.tolist()
        
        return sample_loss

    with tqdm(total=epochs,leave=False,ncols=100) as epoch_bar:
        for epoch in range(epochs):
            epoch_bar.set_description("epoch:{}".format(epoch))
            epoch_loss=[0,0,0]
            with tqdm(total=len(data_set),leave=False,ncols=100) as sample_bar:
                for sample_ind,[tag, lbl, survtime, g] in enumerate(data_set):
                    sample_bar.set_description("sample:{}".format(sample_ind))
                    if g.ndata['feats'].shape[0]<100:
                        sample_bar.update()
                        continue
                    sample_loss=cluster_g(g,n_clusters=4)
                    epoch_loss=[epoch_loss[i]+sample_loss[i] for i in range(len(sample_loss))]
                    sample_bar.update()
            log.info("epoch: {:3}    loss: {}".format(epoch,epoch_loss)) if log is not None else None
            if save_path is not None:
                if (save_interval >0 and epoch%save_interval==0) or (epoch+1==epochs):
                    model_save_path=os.path.join(save_path,"embed_nets_epoch_{}.pt".format(epoch))
                    torch.save(lst_nets,model_save_path)
            epoch_bar.update()
    return lst_nets

def test(data_set:BRCADataset,device,lst_nets,graph_save_path=None):
    def cluster_g(g:dgl.DGLGraph):
        for lst in [0,1,2]:
            lst_nets[lst]=lst_nets[lst].to(device).eval()
            pass
        sub_gs=[None,None,None]
        for lst in [0,1,2]:
            sub_gs[lst] = dgl.node_subgraph(g, g.filter_nodes(lambda nodes: nodes.data['lst'] == lst)).to(device=device)
            sub_gs[lst].ndata['feats']=lst_nets[lst](sub_gs[lst].ndata['feats'])
        for lst in [0,1,2]:
            sut.gut.update_graph_feats_by_subgraph(g,sub_gs[lst],"feats",force_update=True)
        return g
    with tqdm(total=len(data_set),ncols=100) as sample_bar:
        for ind in range(len(data_set)):
            sample_bar.set_description("sample:{}".format(ind))
            g=cluster_g(data_set[ind][3].to(device))
            data_set[ind][3].ndata['feats']=g.ndata['feats'].detach().to(data_set[ind][3].ndata['feats'].device)
            sample_bar.update()
    if graph_save_path : data_set.save(graph_path=graph_save_path)
    pass

pass
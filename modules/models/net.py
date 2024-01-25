import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GATConv, GatedGraphConv
from sagpool import SAGPool
from dgl.nn.pytorch.glob import GlobalAttentionPooling
import graph_utils as gut

class GruGatNets(nn.Module):
    def __init__(self, input, hiddens, classifier):
        super().__init__()
        self.grus = [ nn.ModuleList(), nn.ModuleList(), nn.ModuleList() ]
        self.gats = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(len(hiddens)):
            if i == 0:
                # gru
                for j in range(3):
                    self.grus[j].append(
                        GatedGraphConv(in_feats=input, out_feats=hiddens[i][0], n_steps=hiddens[i][6], n_etypes=1)
                    )
                # gat
                self.gats.append(
                    GATConv(
                        in_feats=input, 
                        out_feats=hiddens[i][0],
                        feat_drop=hiddens[i][3],
                        attn_drop=hiddens[i][4],
                        activation=nn.ELU(), 
                        num_heads=hiddens[i][1], 
                        negative_slope=hiddens[i][5], 
                        allow_zero_in_degree=True
                    )
                )
                # sagpool
                self.pools.append(
                    SAGPool(hiddens[i][0]*hiddens[i][1], ratio=hiddens[i][2])
                )
            else:
                # gru
                for j in range(3):
                    self.grus[j].append(
                        GatedGraphConv(in_feats=hiddens[i-1][0]*hiddens[i-1][1], out_feats=hiddens[i][0], n_steps=hiddens[i][6], n_etypes=1)
                    )
                # gat
                self.gats.append(
                    GATConv(
                        in_feats=hiddens[i-1][0]*hiddens[i-1][1], 
                        out_feats=hiddens[i][0],
                        feat_drop=hiddens[i][3],
                        attn_drop=hiddens[i][4],
                        activation=nn.ELU(), 
                        num_heads=hiddens[i][1], 
                        negative_slope=hiddens[i][5], 
                        allow_zero_in_degree=True
                    )
                )
                # sagpool
                self.pools.append(
                    SAGPool(hiddens[i][0]*hiddens[i][1], ratio=hiddens[i][2])
                )

        self.globalPool = GlobalAttentionPooling(nn.Linear(hiddens[-1][0], 1))
        self.classifier = classifier

    def forward(self, g):
        res =  g.ndata['feats']
        for i in range(len(self.gats)):
            lst00_sub_edges=g.filter_edges(lambda edges:edges.data["lst"]==0)
            lst11_sub_edges=g.filter_edges(lambda edges:edges.data["lst"]==11)
            lst22_sub_edges=g.filter_edges(lambda edges:edges.data["lst"]==22)
            sub_g00=dgl.edge_subgraph(g, edges=lst00_sub_edges, relabel_nodes=True)
            sub_g11=dgl.edge_subgraph(g, edges=lst11_sub_edges, relabel_nodes=True)
            sub_g22=dgl.edge_subgraph(g, edges=lst22_sub_edges, relabel_nodes=True)
            self.grus[0][i].to(res.device)
            sub_g00.ndata["feats"]=self.grus[0][i](sub_g00,sub_g00.ndata['feats'])
            self.grus[1][i].to(res.device)
            sub_g11.ndata["feats"]=self.grus[1][i](sub_g11,sub_g11.ndata['feats'])
            self.grus[2][i].to(res.device)
            sub_g22.ndata["feats"]=self.grus[2][i](sub_g22,sub_g22.ndata['feats'])
            gut.update_graph_feats_by_subgraph(g,sub_g00,"feats")
            gut.update_graph_feats_by_subgraph(g,sub_g11,"feats")
            gut.update_graph_feats_by_subgraph(g,sub_g22,"feats")
            lstDiff_sub_edges=g.filter_edges(
                lambda edges:(edges.data["lst"]!=0) & (edges.data["lst"]!=11) & (edges.data["lst"]!=22)
            )
            sub_gDiff=dgl.edge_subgraph(g, edges=lstDiff_sub_edges, relabel_nodes=True)
            if i!=len(self.gats)-1:
                sub_gDiff.ndata['feats'] = self.gats[i](sub_gDiff, sub_gDiff.ndata['feats']).flatten(1)
                gut.update_graph_feats_by_subgraph(g,sub_gDiff,"feats")
                g, res, _ = self.pools[i](g, g.ndata['feats'])
                g.ndata['feats']=res 
            else:
                res, atten = self.gats[-1](sub_gDiff, sub_gDiff.ndata['feats'], get_attention=True)
                res=res.flatten(1)
                sub_gDiff.ndata['feats']=res
                gut.update_graph_feats_by_subgraph(g,sub_gDiff,"feats")
                pass
            pass

        res = self.globalPool(g, g.ndata['feats'])
        lbl_pred = self.classifier(res)
        return g, res, lbl_pred, (atten, sub_gDiff)

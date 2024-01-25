import dgl,torch

def update_graph_feats_by_subgraph(g: dgl.DGLGraph, sub_g: dgl.DGLGraph, feat_key: str,force_update=False):
    assert(dgl.NID in sub_g.ndata)
    if g.ndata[feat_key].shape[1]!=sub_g.ndata[feat_key].shape[1]:
        if force_update==False:
            raise Exception("shape not same, cannot update. if necessary, set param:force_update")
        pass
        g.ndata[feat_key]=torch.zeros(
            size=(g.ndata[feat_key].shape[0],sub_g.ndata[feat_key].shape[1])
        ).to(g.device)
    origin_nodes = sub_g.ndata[dgl.NID][sub_g.nodes()]
    g.ndata[feat_key][origin_nodes]=sub_g.ndata[feat_key]
    pass

import torch,torch.nn as nn
def cos_sim(vector)->torch.Tensor:
    return  \
        nn.functional.cosine_similarity(
            vector.unsqueeze(1),
            vector.unsqueeze(0),
            dim=2
        )
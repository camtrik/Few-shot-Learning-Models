import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .make_models import register_model, make_model
from utils.functions import compute_logits


@register_model('protonet')
class ProtoNet(nn.Module):
    def __init__(self, encoder, method='sqr', temp=1.0, temp_Learnable=True):
        """
        method: method to compute distance 
        temp: scale parameter, to scale the result of loss
        """
        super().__init__()
        self.encoder = make_model(encoder)
        self.method = method
        if temp_Learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp 

    def forward(self, x_support, x_query):
        support_shape = x_support.shape[:-3]
        query_shape = x_query.shape[:-3]
        image_shape = x_support.shape[-3:]

        x_support = x_support.view(-1, *image_shape)
        x_query = x_query.view(-1, *image_shape)
        x_emb = self.encoder(torch.cat([x_support, x_query], dim=0))
        x_support, x_query = x_emb[:len(x_support)], x_emb[len(x_support):]
        x_support = x_support.view(*support_shape, -1)
        x_query = x_query.view(*query_shape, -1)        
        # prototype
        x_support = x_support.mean(dim=-2)
            
        logits = compute_logits(x_query, x_support, metric=self.method, temp=self.temp)
        return logits
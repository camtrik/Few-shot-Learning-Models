import torch
import torch.nn as nn
import torch.nn.functional as F
from .make_models import register_model, make_model

class MatchingNetwork(nn.Module):
    def __init__(self, encoder):
        super(MatchingNetwork, self).__init__()
        self.encoder = make_model(encoder)

    def forward(self, x_support, x_query):
        support_shape = x_support.shape[:-3]
        query_shape = x_query.shape[:-3]
        image_shape = x_support.shape[-3:]

        x_support = x_support.view(-1, *image_shape)
        x_query = x_query.view(-1, *image_shape)
        
        # Encode support and query images
        x_emb = self.encoder(torch.cat([x_support, x_query], dim=0))
        x_support, x_query = x_emb[:len(x_support)], x_emb[len(x_support):]
        
        x_support = x_support.view(*support_shape, -1)
        x_query = x_query.view(*query_shape, -1)

        # Calculate similarity (cosine similarity)
        x_support_norm = torch.norm(x_support, dim=-1, keepdim=True) + 1e-8
        x_query_norm = torch.norm(x_query, dim=-1, keepdim=True) + 1e-8
        x_support_normalized = x_support / x_support_norm
        x_query_normalized = x_query / x_query_norm

        similarity_matrix = torch.bmm(x_query_normalized, x_support_normalized.transpose(-1, -2))
        
        # Normalize similarity scores
        attention_weights = F.softmax(similarity_matrix, dim=-1)
        
        # Calculate predictions
        logits = torch.bmm(attention_weights, x_support)

        return logits

@register_model('matchnet')
def matchingnet(encoder):
    return MatchingNetwork(encoder)

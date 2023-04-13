import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .make_models import register_model, make_model

class RelationModule(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(RelationModule, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64 * (5 // 4) * (5 // 4), hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, 5),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

@register_model('relationnet')
class RelationNet(nn.Module):
    def __init__(self, encoder, hidden_dim, temp=1.0, temp_Learnable=True):
        super().__init__()
        self.encoder = make_model(encoder)
        self.relation_module = RelationModule(self.encoder.out_dim*2, hidden_dim)
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
        # print("before encoder: ", x_support.shape, x_query.shape)
        x_emb = self.encoder(torch.cat([x_support, x_query], dim=0))
        x_support, x_query = x_emb[:len(x_support)], x_emb[len(x_support):]
        # print("after encoder: ", x_support.shape, x_query.shape)
        x_query = x_query.unsqueeze(1)
        # x_query dim=1 copy 5 times
        x_query = x_query.expand(x_query.size(0), x_support.size(0), x_query.size(2), x_query.size(3), x_query.size(4))

        x_query = x_query[0]
        pairwise_comparison = torch.cat([x_support, x_query], dim=1)
        # print('pairwise_comparison: ', pairwise_comparison.shape)
        logits = self.relation_module(pairwise_comparison)
        # print('logits: ', logits.shape)
        # logits = logits.squeeze(-1).mean(dim=-1) * self.temp

        return logits
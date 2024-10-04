import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, zeros_, ones_
import torch.nn.functional as F

class NeuMF(nn.Module):
    def __init__(self, config, dataset):
        super(NeuMF, self).__init__()
        self.config = config
        self.n_users, self.n_items = config["n_users"], config["n_items"]
        self.embedding_size = config['dim']
        self.layer_sizes = config['layer_sizes']
        self.mf_user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.mf_item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.mlp_user_embedding = nn.Embedding(self.n_users, self.layer_sizes[0] // 2)
        self.mlp_item_embedding = nn.Embedding(self.n_items, self.layer_sizes[0] // 2)
        self.mlp_layers = []
        for layer_idx in range(1, len(self.layer_sizes)):
            dense_layer = nn.Linear(self.layer_sizes[layer_idx - 1], self.layer_sizes[layer_idx])
            self.mlp_layers.append(dense_layer)
        self.mlp_layers = nn.ModuleList(self.mlp_layers)
        self.output_layer = nn.Linear(self.layer_sizes[-1] + self.embedding_size, 1, bias=False)

        kaiming_uniform_(self.mf_user_embedding.weight)
        kaiming_uniform_(self.mf_item_embedding.weight)
        kaiming_uniform_(self.mlp_user_embedding.weight)
        kaiming_uniform_(self.mlp_item_embedding.weight)
        self.init_mlp_layers()
        self.arch = 'gmf'
        self.to(device=self.config["device"])

    def init_mlp_layers(self):
        for layer in self.mlp_layers:
            kaiming_uniform_(layer.weight)
            zeros_(layer.bias)
        ones_(self.output_layer.weight)

    def forward(self, user_list, pos_items, neg_items):
        users_mf = self.mf_user_embedding(torch.LongTensor(user_list).to(self.config["device"]))
        users_mlp = self.mlp_user_embedding(torch.LongTensor(user_list).to(self.config["device"]))
        posI_mf = self.mf_item_embedding(torch.LongTensor(pos_items).to(self.config["device"]))
        posI_mlp = self.mlp_item_embedding(torch.LongTensor(pos_items).to(self.config["device"]))
        negI_mf = self.mf_item_embedding(torch.LongTensor(neg_items).to(self.config["device"]))
        negI_mlp = self.mlp_item_embedding(torch.LongTensor(neg_items).to(self.config["device"]))

        pos_vectors = users_mf * posI_mf
        neg_vectors = users_mf * negI_mf
        mlp_P_vectors = torch.cat([users_mlp, posI_mlp], dim=1)
        mlp_N_vectors = torch.cat([users_mlp, negI_mlp], dim=1)
        for layer in self.mlp_layers:
            mlp_P_vectors = F.leaky_relu(layer(mlp_P_vectors))
            mlp_N_vectors = F.leaky_relu(layer(mlp_N_vectors))

        pos_vec = torch.cat([pos_vectors, mlp_P_vectors], dim=1)
        neg_vec = torch.cat([neg_vectors, mlp_N_vectors], dim=1)

        pos_score = pos_vec * self.output_layer.weight
        neg_score = neg_vec * self.output_layer.weight

        reg = (torch.norm(pos_score, p=2, dim=1) ** 2 + torch.norm(neg_score, p=2, dim=1) ** 2).mean()

        return pos_score, neg_score, users_mlp, posI_mlp, negI_mlp, reg
    
    def predict(self, user_list):
        users_mf = self.mf_user_embedding(torch.LongTensor(user_list).to(self.config["device"]))
        users_mlp = self.mlp_user_embedding(torch.LongTensor(user_list).to(self.config["device"]))
        
        items_mf = self.mf_item_embedding.weight.to(self.config["device"])
        items_mlp = self.mlp_item_embedding.weight.to(self.config["device"])
        
        
        mf_output = users_mf.unsqueeze(1).expand(-1, self.n_items, -1) * items_mf
        
        mlp_output = torch.cat([users_mlp.unsqueeze(1).expand(-1, self.n_items, -1), items_mlp.unsqueeze(0).expand(len(user_list), -1, -1)], dim=2)
        
        for layer in self.mlp_layers:
            mlp_output = F.leaky_relu(layer(mlp_output))
        
        combined_output = torch.cat([mf_output, mlp_output], dim=2).view(-1, self.layer_sizes[-1] + self.embedding_size)
        prediction = self.output_layer(combined_output).view(len(user_list), self.n_items)
        

        return prediction

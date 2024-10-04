import json
import os
import time
import numpy as np
import torch
import torch.optim as optim
import random
from torch.utils.data import DataLoader
from datetime import datetime
import torch.nn.functional as F
from utls.mydataset import CFDataset
from model.MF.MF import *
from model.LightGCN.LightGCN import *
from model.NeuMF.NeuMF import *
from utls.utilize import calculate_f1, bpr_loss, custom_loss, slice_lists, batch_split


class BasicTrainer:
    def __init__(self, trainer_config) -> None:
        self.config = trainer_config
        self.device = trainer_config['device']
        self.n_epochs = trainer_config['n_epochs']
        self.min_epochs = trainer_config['min_epochs']
        self.max_patience = trainer_config.get('patience', 50)
        self.val_interval = trainer_config.get('val_interval', 1)
    
    def _create_dataset(self, path):
        raise NotImplementedError
    
    def _create_dataloader(self):
        self.dataloader = DataLoader(self.dataset, batch_size=int(self.config["batch_size"]), shuffle=True)
        print(f"Create Dataloader with batch_size:{int(self.config['batch_size'])}")

    def _create_model(self):
        raise NotImplementedError
    
    def _create_opt(self):
        raise NotImplementedError

    def _train_epoch(self, epoch):
        raise NotImplementedError
    
    def _eval_model(self, epoch):
        raise NotImplementedError

    def _save_model(self, best_model_path):
        torch.save({
            'model': self.model.state_dict(),
        }, best_model_path)
    
    def _load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model'])

    def train(self, path=None):
        patience = self.config["patience"]
        best_metrics = -1
        
        best_model_path = f"{self.config['checkpoints']}/{self.config['model']}/{self.config['method']}/{self.config['dataset']}"
        if self.config["main_file"] != "":
            best_model_path = os.path.join(best_model_path, self.config["main_file"])
        if path is not None:
            best_model_path = path
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path, exist_ok=True)
        best_model_path = os.path.join(best_model_path, f"{self.config['noise']}_{self.config['seed']}.pth")

        self.update_flag = False
        for epoch in range(self.n_epochs):
            self._train_epoch(epoch)
            if (epoch + 1) % self.config["val_interval"] == 0:
                metrics_list, _ = self._eval_model(epoch)
                metrics = metrics_list[0]
                if (epoch + 1) >= self.config["min_epochs"]:
                    if metrics > best_metrics:
                        best_metrics = metrics
                        # Save the best model
                        self._save_model(best_model_path)
                        patience = self.config["patience"]
                    else:
                        patience -= self.config["val_interval"]
                        if patience <= 0:
                            print('Early stopping!')
                            break

        self._load_model(best_model_path)
        # Test
        avg_hr, avg_ndcg = self._eval_model(eval_type='test')

        return avg_hr, avg_ndcg


class CFTrainer(BasicTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)
    
        self._create_dataset(f"data/{trainer_config['dataset']}")
        self._create_dataloader()
        self._create_model()
        self._create_opt()

    def _create_dataset(self, path):
        self.dataset = CFDataset(path, self.config)
    
    def _create_dataloader(self):
        return super()._create_dataloader()

    def _create_opt(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.config["lr"])

    def _create_model(self):
        glo = globals()
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        self.model = glo[f'{self.config["model"]}'](self.config["model_config"], self.dataset)
        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()
    
    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0

        for batch_data in self.dataloader:
            self.opt.zero_grad()
            self.model.train()
            user_id_list, pos_item_list, neg_item_list = self.dataset.get_train_batch(batch_data)
            if self.config["model"] in ["NeuMF"]:
                pos_logits, neg_logits, _, _, _, l2_norm_sq = self.model(user_id_list, pos_item_list, neg_item_list)
                pos_logits = torch.sum(pos_logits, dim=1)
                neg_logits = torch.sum(neg_logits, dim=1)
            else:
                users_emb, pos_items_emb, neg_items_emb, l2_norm_sq = self.model(user_id_list, pos_item_list, neg_item_list)
                pos_logits = torch.sum(users_emb * pos_items_emb, dim=1)
                neg_logits = torch.sum(users_emb * neg_items_emb, dim=1)
            loss = self._rec_loss(pos_logits, neg_logits).mean() + self.config["weight_decay"] * l2_norm_sq
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
        
        end_t = time.time()
        print(f"Epoch {epoch}: Rec Loss: {epoch_loss/len(self.dataloader):.4f}, Time: {end_t-start_t:.2f}")

    def _rec_loss(self, pos, neg):
        return F.softplus(neg - pos)

    def _eval_model(self, epoch=0, eval_type='val'):
        start_t = time.time()

        assert eval_type in ['val', 'test']
        self.model.eval()
        top_ks = self.config["rec_top_k"]

        recall_list = [0.0 for _ in self.config["rec_top_k"]]
        ndcg_list = [0.0 for _ in self.config["rec_top_k"]]
    

        user_list = list(range(self.dataset.n_users))
        for batch_data in batch_split(users=user_list, batch_size=self.config["test_batch_size"]):
            if eval_type == 'val':
                user_id_list, user_inter_list, user_train_list = self.dataset.get_val_batch(batch_data)
            else:
                user_id_list, user_inter_list, user_train_list = self.dataset.get_test_batch(batch_data)
            
            with torch.no_grad():
                score_list = self.model.predict(user_id_list)

            for idx, user_train_items in enumerate(user_train_list):
                score_list[idx, user_train_items] = float('-inf')
            max_k = max(top_ks)

            for user_idx, user_inter_items in enumerate(user_inter_list):
                _, top_indices = torch.topk(score_list[user_idx], max_k)
                for idx, k in enumerate(top_ks):
                    top_indices_k = top_indices[:k]
                    
                    num_hits = sum([1 for item in user_inter_items if item in top_indices_k])

                    # Recall@k
                    recall_k = num_hits / len(user_inter_items) if user_inter_items else 0.0

                    # NDCG@k
                    dcg = sum([1 / np.log2(i + 2) for i, item in enumerate(top_indices_k) if item in user_inter_items])
                    idcg = sum([1.0 / np.log2(i + 2) for i in range(len(user_inter_items))])
                    idcg = 1.0 if idcg == 0 else idcg
                    ndcg_k = dcg / idcg

                    recall_list[idx] += recall_k
                    ndcg_list[idx] += ndcg_k


        avg_hr = [hr / self.dataset.n_users for hr in recall_list]
        avg_ndcg = [ndcg / self.dataset.n_users for ndcg in ndcg_list]

        end_t = time.time()
        print(("Validation - " if eval_type == 'val' else "Test - ") + f"Time: {end_t - start_t:.2f}")

        epoch_text = f"at Epoch {epoch}" if eval_type == 'val' else ""
        self._print_performance("Recommendation Performance" + epoch_text, ("Recall", "NDCG"), avg_hr, avg_ndcg, self.config["rec_top_k"])
    
        return recall_list, ndcg_list


    
    def _print_performance(self, title, metrics, m1_list, m2_list, top_k_list):
        out_text = f"{title}:"
        for i, k in enumerate(top_k_list):
            out_text += f"\n{metrics[0]}@{k}: {m1_list[i]:.4f}, {metrics[1]}@{k}: {m2_list[i]:.4f};"
        print(out_text)


class SelfDenoiseCFTrainer(CFTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)
        
        self.begin_adv = trainer_config["model_config"]['denoise_config']['begin_adv']
        self.temp = self.config["model_config"]['denoise_config']["temperature"]
        
    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0
        

        for batch_data in self.dataloader:
            self.opt.zero_grad()
            self.model.train()
            
            if epoch >= self.begin_adv:
                user_id_list, pos_item_list, neg_item_list = self.dataset.get_train_batch(batch_data, multi_sample=True, k=self.config["model_config"]['denoise_config']["item_num"])
                
                # pre-selection
                with torch.no_grad():
                    pos_item_list_flat = torch.tensor(pos_item_list).flatten()
                    user_id_list_repeated = torch.tensor(user_id_list).repeat_interleave(self.config["model_config"]['denoise_config']["item_num"])
                    neg_item_list_repeated = torch.tensor(neg_item_list).repeat_interleave(self.config["model_config"]['denoise_config']["item_num"])
                    
                    if self.config["model"] in ["NeuMF"]:
                        pos_logits, neg_logits, _, _, _, _ = self.model(user_id_list_repeated, pos_item_list_flat, neg_item_list_repeated)
                        pos_logits = torch.sum(pos_logits, dim=1)
                        neg_logits = torch.sum(neg_logits, dim=1)
                    else:
                        users_emb, pos_items_emb, neg_items_emb, _ = self.model(user_id_list_repeated, pos_item_list_flat, neg_item_list_repeated)
                        pos_logits = torch.sum(users_emb * pos_items_emb, dim=1)
                        neg_logits = torch.sum(users_emb * neg_items_emb, dim=1)
                    
         
                    temp_loss = self._rec_loss(pos_logits, neg_logits).detach()

                    temp_loss = temp_loss.view(len(user_id_list), self.config["model_config"]['denoise_config']["item_num"])

                    sampling_probabilities = torch.nn.functional.softmax(-temp_loss / self.temp, dim=1)

                    sampled_indices = torch.multinomial(sampling_probabilities, 1, replacement=False).squeeze()

                    pos_item_list = torch.gather(torch.tensor(pos_item_list).to(sampled_indices.device), 1, sampled_indices.unsqueeze(-1)).squeeze().cpu().tolist()
                    
            else:
                user_id_list, pos_item_list, neg_item_list = self.dataset.get_train_batch(batch_data)
            
            if self.config["model"] in ["NeuMF"]:
                pos_logits, neg_logits, _, _, _, l2_norm_sq = self.model(user_id_list, pos_item_list, neg_item_list)
                pos_logits = torch.sum(pos_logits, dim=1)
                neg_logits = torch.sum(neg_logits, dim=1)
            else:
                users_emb, pos_items_emb, neg_items_emb, l2_norm_sq = self.model(user_id_list, pos_item_list, neg_item_list)
                pos_logits = torch.sum(users_emb * pos_items_emb, dim=1)
                neg_logits = torch.sum(users_emb * neg_items_emb, dim=1)
            loss = self._rec_loss(pos_logits, neg_logits).mean() + self.config["weight_decay"] * l2_norm_sq
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
        
        end_t = time.time()
        print(f"Epoch {epoch}: Rec Loss: {epoch_loss/len(self.dataloader):.4f}, Time: {end_t-start_t:.2f}")
import torch as tr
from torch import nn
import os
import json
from domCNN import domCNN
from dataset import PFamDataset

class domCNNe(nn.Module):
    def __init__(self, models_path, emb_path, data_path, cat_path, voting_method):
        super(domCNNe, self).__init__()

        # Load model paths
        model_dirs = [os.path.join(models_path, d) for d in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, d))]
        
        # Load categories
        with open(cat_path, 'r') as f:
            categories = [item.strip() for item in f]
        
        self.categories = categories
        self.emb_path = emb_path
        self.data_path = data_path
        self.voting_method = voting_method

        # Initialize the ensemble of models
        self.models = nn.ModuleList()
        self.model_configs = []

        for model_dir in model_dirs:
            # Load the config.json to get the parameters
            config_path = os.path.join(model_dir, 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            lr = config['lr']
            batch_size = config['batch_size']
            win_len = config['window_len']
            label_win_len = config['label_win_len']
            only_seeds = config.get('only_seeds', False)

            self.model_configs.append({
                'lr': lr,
                'batch_size': batch_size,
                'win_len': win_len,
                'label_win_len': label_win_len,
                'only_seeds': only_seeds
            })

            # Load the model weights
            weights_path = os.path.join(model_dir, 'weights.pk')
            print("loading weights from", model_dir)
            model = domCNN(len(categories), lr=lr, device="cuda")
            model.load_state_dict(tr.load(weights_path))
            model.eval()
            self.models.append(model)
        
        if self.voting_method in ['weighted_mean','all']:
            self.model_weights = nn.Parameter(tr.rand(len(model_dirs)))
        if self.voting_method in ['weighted_families','all']:
            self.family_weights = nn.Parameter(tr.rand(len(model_dirs), len(categories)))

    def fit(self):
        if self.voting_method in ['weighted_mean', 'weighted_families', 'all']:
            all_preds = []
            for i, net in enumerate(self.models):
                config = self.model_configs[i]
                dev_data = PFamDataset(
                    f"{self.data_path}dev.csv",
                    self.emb_path,
                    self.categories,
                    win_len=config['win_len'],
                    label_win_len=config['label_win_len'],
                    only_seeds=config['only_seeds'],
                    is_training=False
                )
                dev_loader = tr.utils.data.DataLoader(dev_data, batch_size=config['batch_size'], num_workers=1)

                with tr.no_grad():
                    _, _, pred, ref, _, _, _, _, _ = net.pred(dev_loader)
                    all_preds.append(pred)
            stacked_preds = tr.stack(all_preds)

        if self.voting_method in ['weighted_mean', 'all']:
            criterion = nn.CrossEntropyLoss()
            optimizer = tr.optim.Adam([self.model_weights], lr=0.01)

            for epoch in range(500):
                pred_avg = tr.sum(stacked_preds * self.model_weights.view(-1, 1, 1), dim=0)
                loss = criterion(pred_avg, tr.argmax(ref, dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f'Epoch {epoch+1}, Loss: {loss.item()}')

            if self.voting_method == 'all':
                pred_mean = tr.argmax(tr.mean(stacked_preds, dim=0), dim=1)
                pred_weighted_mean = tr.argmax(pred_avg, dim=1)
                self.final_preds = {
                    'mean': pred_mean,
                    'weighted_mean': pred_weighted_mean
                }
            else:
                print('Final model weights:', self.model_weights)

        if self.voting_method in ['weighted_families', 'all']:
            criterion = nn.CrossEntropyLoss()
            optimizer = tr.optim.Adam([self.family_weights], lr=0.01)

            for epoch in range(500):
                pred_avg = tr.sum(stacked_preds * self.family_weights.view(len(self.models), 1, len(self.categories)), dim=0)
                loss = criterion(pred_avg, tr.argmax(ref, dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f'Epoch {epoch+1}, Loss: {loss.item()}')

            print('Final family weights:', self.family_weights)

    def pred(self):
        all_preds = []
        for i, net in enumerate(self.models):
            config = self.model_configs[i]
            test_data = PFamDataset(
                f"{self.data_path}test.csv",
                self.emb_path,
                self.categories,
                win_len=config['win_len'],
                label_win_len=config['label_win_len'],
                only_seeds=config['only_seeds'],
                is_training=False
            )
            test_loader = tr.utils.data.DataLoader(test_data, batch_size=config['batch_size'], num_workers=1)

            net_preds = []
            with tr.no_grad():
                test_loss, test_errate, pred, _, _, _, _, _, _ = net.pred(test_loader)
                net_preds.append(pred)
            print(f"win_len = {config['win_len']} - label_win_len = {config['label_win_len']} - lr = {config['lr']} - test_loss {test_loss:.5f} - test_errate {test_errate:.5f}")
            net_preds = tr.cat(net_preds)
            all_preds.append(net_preds)

        stacked_preds = tr.stack(all_preds)

        # Calculate predictions for each method
        if self.voting_method == 'all':
            pred_mean = tr.argmax(tr.mean(stacked_preds, dim=0), dim=1)
            pred_weighted_mean = tr.argmax(tr.sum(stacked_preds * self.model_weights.view(-1, 1, 1), dim=0), dim=1)
            pred_weighted_fam = tr.argmax(tr.sum(stacked_preds * self.family_weights.view(len(self.models), 1, len(self.categories)), dim=0), dim=1)
            pred_majority = tr.mode(tr.argmax(stacked_preds, dim=2), dim=0)[0]
            return pred_mean, pred_weighted_mean, pred_weighted_fam, pred_majority
        elif self.voting_method == 'mean':
            pred_mean = tr.argmax(tr.mean(stacked_preds, dim=0), dim=1)
            return pred_mean
        elif self.voting_method == 'weighted_mean':
            pred_weighted_mean = tr.argmax(tr.sum(stacked_preds * self.model_weights.view(-1, 1, 1), dim=0), dim=1)
            return pred_weighted_mean
        elif self.voting_method == 'weighted_families':
            pred_weighted_fam = tr.argmax(tr.sum(stacked_preds * self.family_weights.view(len(self.models), 1, len(self.categories)), dim=0), dim=1)
            return pred_weighted_fam
        elif self.voting_method == 'majority':
            pred_majority = tr.mode(tr.argmax(stacked_preds, dim=2), dim=0)[0]
            return pred_majority


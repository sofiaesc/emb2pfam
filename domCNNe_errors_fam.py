import torch as tr
from torch import nn
import os
import json
from domCNN import domCNN
from dataset import PFamDataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

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
        
        if self.voting_method == 'weighted_mean':
            self.model_weights = nn.Parameter(tr.rand(len(model_dirs)))

        if self.voting_method == 'weighted_families':
            self.family_weights = nn.Parameter(tr.rand(len(model_dirs), len(categories)))

        if self.voting_method == 'weighted_families_f1':
            self.family_weights = tr.zeros(len(self.models), len(self.categories))

    def fit(self):
        if self.voting_method == 'weighted_mean':
            criterion = nn.CrossEntropyLoss()
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

            optimizer = tr.optim.Adam([self.model_weights], lr=0.01)

            for epoch in range(200):
                pred_avg = tr.sum(stacked_preds * self.model_weights.view(-1, 1, 1), dim=0)
                loss = criterion(pred_avg, tr.argmax(ref, dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f'Epoch {epoch+1}, Loss: {loss.item()}')

            print('Final model weights:', self.model_weights)

        elif self.voting_method == 'weighted_families':
            criterion = nn.CrossEntropyLoss()
            all_dev_preds = []
            all_test_preds = []
            
            # Datos de validación (dev)
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
                    _, _, dev_pred, dev_ref, _, _, _, _, _ = net.pred(dev_loader)
                    all_dev_preds.append(dev_pred)
            stacked_dev_preds = tr.stack(all_dev_preds)

            # Datos de prueba (test)
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

                with tr.no_grad():
                    _, _, test_pred, test_ref, _, _, _, _, _ = net.pred(test_loader)
                    all_test_preds.append(test_pred)
            stacked_test_preds = tr.stack(all_test_preds)

            optimizer = tr.optim.Adam([self.family_weights], lr=0.01)
            dev_error_values = []
            test_error_values = []

            for epoch in range(1000):
                # Predicción y pérdida en dev
                dev_pred_avg = tr.sum(stacked_dev_preds * self.family_weights.view(len(self.models), 1, len(self.categories)), dim=0)
                dev_loss = criterion(dev_pred_avg, tr.argmax(dev_ref, dim=1))

                dev_accuracy = tr.sum(tr.argmax(dev_pred_avg, dim=1) == tr.argmax(dev_ref, dim=1)).item() / len(dev_ref)
                dev_error = 1 - dev_accuracy
                dev_error_values.append(dev_error)

                # Predicción y pérdida en test
                test_pred_avg = tr.sum(stacked_test_preds * self.family_weights.view(len(self.models), 1, len(self.categories)), dim=0)
                test_loss = criterion(test_pred_avg, tr.argmax(test_ref, dim=1))

                test_accuracy = tr.sum(tr.argmax(test_pred_avg, dim=1) == tr.argmax(test_ref, dim=1)).item() / len(test_ref)
                test_error = 1 - test_accuracy
                test_error_values.append(test_error)

                # Actualizar pesos de las familias
                optimizer.zero_grad()
                dev_loss.backward()
                optimizer.step()

                print(f'Epoch {epoch+1}, Dev Error: {dev_error:.5f}, Test Error: {test_error:.5f}')

            print('Final family weights:', self.family_weights)

            # Graficar Dev Error y Test Error
            plt.figure(figsize=(10, 6))
            plt.plot(dev_error_values, label='Dev Error', color='blue', linewidth=2)
            plt.plot(test_error_values, label='Test Error', color='magenta', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.title('Dev and Test Error over Epochs')
            plt.legend()
            plt.grid(True)
            plt.ylim(0, 0.3)
            plt.savefig('error_plot.png', dpi=300)
            plt.close()

        elif self.voting_method == 'weighted_families_f1':
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
                    pred_classes = tr.argmax(pred, dim=1).cpu().numpy()
                    ref_classes = tr.argmax(ref, dim=1).cpu().numpy()

                    f1_scores = f1_score(ref_classes, pred_classes, average=None, labels=list(range(len(self.categories))))

                self.family_weights[i] = tr.tensor(f1_scores)

            print('Final family weights based on F1 score:', self.family_weights)

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

        if self.voting_method == 'mean':
            pred_avg = tr.mean(stacked_preds, dim=0)
            pred_avg_bin = tr.argmax(pred_avg, dim=1)
        elif self.voting_method == 'weighted_mean':
            pred_avg = tr.sum(stacked_preds * self.model_weights.view(-1, 1, 1), dim=0)
            pred_avg_bin = tr.argmax(pred_avg, dim=1)
        elif self.voting_method == 'weighted_families':
            pred_avg = tr.sum(stacked_preds * self.family_weights.view(len(self.models), 1, len(self.categories)), dim=0)
            pred_avg_bin = tr.argmax(pred_avg, dim=1)
        elif self.voting_method == 'weighted_families_f1':
            pred_avg = tr.sum(stacked_preds * self.family_weights.view(len(self.models), 1, len(self.categories)), dim=0)
            pred_avg_bin = tr.argmax(pred_avg, dim=1)
        elif self.voting_method == 'majority':
            pred_bin = tr.argmax(stacked_preds, dim=2)
            pred_avg_bin = tr.mode(pred_bin, dim=0)[0]
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")

        return pred_avg_bin


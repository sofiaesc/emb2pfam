import torch as tr
from torch import nn
import os
import json
import matplotlib.pyplot as plt
from domCNN import domCNN
import sklearn.metrics as mt

class domCNNe(nn.Module):
    def __init__(self, models_path, emb_path, data_path, cat_path, voting_method):
        super(domCNNe, self).__init__()
        
        # Load model paths
        model_dirs = [os.path.join(models_path, d) for d in os.listdir(models_path)]
        
        # Load categories
        with open(cat_path, 'r') as f:
            categories = [item.strip() for item in f]
        
        self.categories = categories
        self.emb_path = emb_path
        self.data_path = data_path
        self.voting_method = voting_method
        
        # Initialize the ensemble of models
        self.models = nn.ModuleList()
        self.num_models = len(model_dirs)
        for model_dir in model_dirs:
            # Load the config.json to get the learning rate
            config_path = os.path.join(model_dir, 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            lr = config['lr']
            
            # Load the model weights
            weights_path = os.path.join(model_dir, 'weights.pk')
            print("loading weights from", model_dir)
            model = domCNN(len(categories), lr=lr, device="cuda")
            model.load_state_dict(tr.load(weights_path))
            model.eval()
            self.models.append(model)
        
        # Initialize learnable weights for each model
        self.model_weights = nn.Parameter(tr.rand(self.num_models))

    # def forward(self, dataloader):
        #llamas al pred de cada uno de los modelitos

    def fit(self, dev_loader, test_loader):
        if self.voting_method == 'weighted_mean':  # adjusting weights only for weighted mean option
            criterion = nn.CrossEntropyLoss() 
            optimizer = tr.optim.Adam([self.model_weights], lr=0.01) 
            
            dev_error_values = []  
            test_error_values = []  
            dev_loss_values = []  
            test_loss_values = []  

            # Collect predictions for dev_loader
            all_dev_preds = []  
            for net in self.models:
                with tr.no_grad():
                    _, _, pred_dev, ref_dev, _, _, _, _, _ = net.pred(dev_loader)
                    all_dev_preds.append(pred_dev)
            
            # Collect predictions for test_loader
            all_test_preds = []
            for net in self.models:
                with tr.no_grad():
                    _, _, pred_test, ref_test, _, _, _, _, _ = net.pred(test_loader)
                    all_test_preds.append(pred_test)
            
            stacked_dev_preds = tr.stack(all_dev_preds) 
            stacked_test_preds = tr.stack(all_test_preds)

            for epoch in range(200):
                # dev
                dev_pred_avg = tr.sum(stacked_dev_preds * self.model_weights.view(-1, 1, 1), dim=0)  
                dev_loss = criterion(dev_pred_avg, tr.argmax(ref_dev, dim=1)) 
                dev_loss_values.append(dev_loss.item()) 

                dev_accuracy = mt.accuracy_score(tr.argmax(ref_dev, dim=1).cpu().numpy(), tr.argmax(dev_pred_avg,dim=1).cpu().numpy())
                dev_error = 1 - dev_accuracy
                dev_error_values.append(dev_error)

                # test
                test_pred_avg = tr.sum(stacked_test_preds * self.model_weights.view(-1, 1, 1), dim=0)  
                test_loss = criterion(test_pred_avg, tr.argmax(ref_test, dim=1)) 
                test_loss_values.append(test_loss.item())

                test_accuracy = mt.accuracy_score(tr.argmax(ref_test, dim=1).cpu().numpy(), tr.argmax(test_pred_avg,dim=1).cpu().numpy())
                test_error = 1 - test_accuracy
                test_error_values.append(test_error)

                optimizer.zero_grad()
                dev_loss.backward()
                optimizer.step()

                print(f'Epoch {epoch+1}, Dev Error: {dev_loss.item()}, Test Error: {test_loss.item()}')

            print('Final model weights:', self.model_weights)
            
            # Plot the loss
            plt.plot(range(1, 201), dev_loss_values, label='Dev Loss')
            plt.plot(range(1, 201), test_loss_values, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Test Loss')
            plt.legend()
            plt.savefig(os.path.join('ensembles/results', 'training_test_loss.png'))
            plt.show()
            plt.close()

            # Plot the errors
            plt.plot(range(1, 201), dev_error_values, label='Dev Error')
            plt.plot(range(1, 201), test_error_values, label='Test Error')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.title('Training and Test Error')
            plt.legend()
            plt.savefig(os.path.join('ensembles/results', 'training_test_error.png'))
            plt.show()
            plt.close()

    def pred(self, test_loader):
        all_preds = []
        for net in self.models:
            net_preds = []
            with tr.no_grad():
                _, _, pred, _, _, _, _, _, _ = net.pred(test_loader)
                net_preds.append(pred)
            net_preds = tr.cat(net_preds)  
            all_preds.append(net_preds)

        stacked_preds = tr.stack(all_preds)  

        if self.voting_method == 'mean':
            pred_avg = tr.mean(stacked_preds, dim=0)  
            pred_avg_bin = tr.argmax(pred_avg, dim=1) 
        elif self.voting_method == 'weighted_mean':
            pred_avg = tr.sum(stacked_preds * self.model_weights.view(-1, 1, 1), dim=0) 
            pred_avg_bin = tr.argmax(pred_avg, dim=1) 
        elif self.voting_method == 'majority':
            pred_bin = tr.argmax(stacked_preds, dim=2)  
            pred_avg_bin = tr.mode(pred_bin, dim=0)[0]  
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")

        return pred_avg_bin

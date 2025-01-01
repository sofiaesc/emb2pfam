import torch as tr
from torch import nn
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from tqdm import tqdm
#from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

class domCNN(nn.Module):
    def __init__(self, nclasses, emb_size=1280, lr=1e-3, device="cuda", logger=None,# max_train_len=20000,
    filters=1100, kernel_size=9, num_layers=5, first_dilated_layer=2, dilation_rate=3, 
    resnet_bottleneck_factor=.5):
        # nclasses includes not-domain
        super().__init__()

        #self.max_train_len = max_train_len # max number of batches per epoch
        self.emb_size = emb_size # esm-1b

        self.logger = logger
        self.train_steps = 0
        self.dev_steps = 0

        self.cnn = [nn.Conv1d(self.emb_size, filters, kernel_size, padding="same")]
        for k in range(num_layers):
            self.cnn.append(ResidualLayer(k, first_dilated_layer, dilation_rate, resnet_bottleneck_factor, filters, kernel_size))
        self.cnn.append(nn.AdaptiveMaxPool1d(1))
        self.cnn = nn.Sequential(*self.cnn)

        self.fc = nn.Linear(filters, nclasses)

        self.loss = nn.CrossEntropyLoss()
        self.optim = tr.optim.Adam([{"params": self.cnn.parameters(), "lr": lr},
                                    {"params": self.fc.parameters(), "lr": lr}])

        #self.scheduler = ReduceLROnPlateau(self.optim, patience=3)
        #scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

        self.to(device)
        self.device = device

    def forward(self, emb):
        """emb is the embedded sequence batch with shape [N, EMBSIZE, L]"""
        y = self.cnn(emb.to(self.device))
        y = self.fc(y.squeeze(2))
        return y

    def fit(self, dataloader):

        avg_loss = 0
        self.cnn.train(), self.fc.train()
        self.optim.zero_grad()
        for k,(x, y, _, _, _, _, _) in enumerate(tqdm(dataloader)):
            yhat = self(x)
            y = y.to(self.device)

            loss = self.loss(yhat, y)
            loss.backward()
            avg_loss += loss.item()
            self.optim.step()
            self.optim.zero_grad()

            if self.logger is not None:
                self.logger.add_scalar("Loss/train", loss, self.train_steps)
            self.train_steps+=1
            #if k>self.max_train_len:
            #    break

        avg_loss /= len(dataloader)

        return avg_loss

    def pred(self, dataloader):
        test_loss = 0
        pred, ref, names, starts, ends, label_starts, label_ends = [], [], [], [], [], [], []
        self.eval()

        for seq, y, name, start, end, label_start, label_end in tqdm(dataloader):
            with tr.no_grad():
                yhat = self(seq)
                y = y.to(self.device)
                test_loss += self.loss(yhat, y).item()

            names += name
            starts.append(start)
            ends.append(end)
            label_starts.append(label_start)
            label_ends.append(label_end)
            pred.append(yhat.detach().cpu())
            ref.append(y.cpu())

        pred = tr.cat(pred)
        pred_bin = tr.argmax(pred, dim=1)
        
        ref = tr.cat(ref)
        ref_bin = tr.argmax(ref, dim=1)

        self.dev_steps += 1
        test_loss /= len(dataloader)

        #self.scheduler.step(test_loss)

        acc = accuracy_score(ref_bin, pred_bin)
        if self.logger is not None:
            self.logger.add_scalar("Loss/dev", test_loss, self.dev_steps)
            balacc = balanced_accuracy_score(ref_bin, pred_bin)
            self.logger.add_scalar("Error rate/dev", 1-acc, self.dev_steps)
            self.logger.add_scalar("Balanced acc/dev", balacc, self.dev_steps)

        return test_loss, 1-acc, pred, ref, names, starts, ends, label_starts, label_ends

class ResidualLayer(nn.Module):
    def __init__(self, layer_index, first_dilated_layer, dilation_rate, resnet_bottleneck_factor, filters, kernel_size):
        super().__init__()

        shifted_layer_index = layer_index - first_dilated_layer + 1
        dilation_rate = max(1, dilation_rate**shifted_layer_index)

        num_bottleneck_units = math.floor(
            resnet_bottleneck_factor * filters)

        self.layer = nn.Sequential(nn.BatchNorm1d(filters),
        nn.ReLU(),
        nn.Conv1d(filters, num_bottleneck_units, kernel_size, dilation=dilation_rate, padding="same"), 
        nn.BatchNorm1d(num_bottleneck_units),
        nn.ReLU(),
        nn.Conv1d(num_bottleneck_units, filters, kernel_size=1, padding="same"))
        # The second convolution is purely local linear transformation across
        # feature channels, as is done in
        # tensorflow_models/slim/nets/resnet_v2.bottleneck

    def forward(self, x):

        return x + self.layer(x)


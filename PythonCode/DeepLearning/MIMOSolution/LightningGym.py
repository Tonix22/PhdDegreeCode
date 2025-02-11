import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from MIMOTransformer import MIMOTransformer
from MIMONetV2 import MIMONetV2

# Luego definimos la clase LightningModule que envuelve la red
class LitModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(LitModel, self).__init__()
        self.save_hyperparameters()  # guarda hyperparams en checkpoints, etc.
        self.model = MIMONetV2()
        self.lr = lr
        # Podríamos usar cross entropy
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  # (N, 64), y -> (N,) con las clases
        preds = self(x)  # (N, 4)
        loss = self.criterion(preds, y)
        # Registrar la métrica de entrenamiento
        self.log("train_loss", loss, prog_bar=True)
        
        acc = (preds.argmax(dim=1) == y).float().sum()/y.shape[0]
        #self.log("val_acc", acc, prog_bar=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        
        # Calcular accuracy como ejemplo
        acc = (preds.argmax(dim=1) == y).float().sum()/y.shape[0]
        
        # Registrar la métrica de validación
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        # Definimos el optimizador
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
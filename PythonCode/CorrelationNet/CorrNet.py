import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class CorrelationNetLightning(pl.LightningModule):
    def __init__(self, num_classes=4, output_dim=48, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.lr = lr
        
        # Convolution layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),  # 48 -> 24
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2)   # 24 -> 12
        )
        
        # Flattened size: 64 * 12 * 12 = 9216
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.GELU(),
            nn.Linear(128, self.output_dim * self.num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        x shape: (B, 4, 48, 48)
        Returns: (B, 48, 4)
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x.view(-1, self.output_dim, self.num_classes)

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (B, 4, 48, 48), y: (B, 48)
        logits = self.forward(x)  # (B, 48, 4)
        
        # Flatten for cross entropy:
        loss = self.criterion(logits.view(-1, self.num_classes), y.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch  # x: (B, 4, 48, 48), y: (B, 48)
        logits = self.forward(x)  # (B, 48, 4)
        
        # Compute the loss (flatten for cross-entropy):
        loss = self.criterion(logits.view(-1, self.num_classes), y.view(-1))
        
        # Predictions:
        preds = torch.argmax(logits, dim=-1)  # shape (B, 48)
        
        # Symbol-level accuracy:
        correct = (preds == y).sum()
        total = y.numel()  # B * 48
        batch_acc = correct.float() / total
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", batch_acc, prog_bar=True)
        
        return {"val_loss": loss, "val_acc": batch_acc}

    
    """
    def validation_step(self, batch, batch_idx):
        
        Perform an ensemble-like evaluation by shifting the input 48 times.
        We'll sum the logits (after unshifting them back) and then do argmax.

        x, y = batch  # x shape: (B,4,48,48), y shape: (B,48)
        B = x.size(0)
        num_shifts = 48
        
        # We'll accumulate logits in 'logits_sum'
        # shape: (B, 48, 4)
        logits_sum = None

        for s in range(num_shifts):
            # 1) Shift the input x by s positions along dimension=3
            #    (the 48-wide column dimension)
            x_shift = torch.roll(x, shifts=s, dims=3)

            # 2) Forward pass
            #    shape: (B, 48, 4)
            logits_s = self.forward(x_shift)

            # 3) "Unshift" these logits by -s along dimension=1
            #    so they line up with the original label y
            #    i.e. logits_s_unshift[i,0] now corresponds to the same
            #    subcarrier index as y[i,0], etc.
            logits_s_unshift = torch.roll(logits_s, shifts=-s, dims=1)

            # 4) Accumulate
            if logits_sum is None:
                logits_sum = logits_s_unshift
            else:
                logits_sum += logits_s_unshift

        # 5) Average the logits across all shifts
        logits_ensemble = logits_sum / num_shifts  # (B,48,4)

        # 6) Final predictions
        preds = torch.argmax(logits_ensemble, dim=-1)  # (B,48)

        # 7) Compute loss using the *averaged* logits and the original y
        loss = self.criterion(logits_ensemble.view(-1, self.num_classes),
                            y.view(-1))

        # 8) Symbol-level Accuracy
        correct = (preds == y).sum()
        total = y.numel()  # B*48
        batch_acc = correct.float() / total

        # 9) Bit Error Rate (BER)
        # Each label/pred in [0..3] => 2 bits: 0->00, 1->01, 2->10, 3->11
        # We'll XOR them, then count bit mismatches
        xordiff = preds ^ y  # shape (B,48)
        bit_errors = ((xordiff >> 1) & 1).sum() + (xordiff & 1).sum()
        bits_total = 2 * total
        ber = bit_errors.float() / bits_total

        # 10) Log final metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", batch_acc, prog_bar=True)
        self.log("val_BER", ber, prog_bar=True)

        return {"val_loss": loss, "val_acc": batch_acc, "val_BER": ber}
    """



    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits.view(-1, self.num_classes), y.view(-1))
        self.log("test_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

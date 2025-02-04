from pytorch_lightning import Trainer
from LightningGym import LitModel
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader, random_split
from RowDataset import *

EPOCHS = 1
BATCHSIZE = 256
LEARNINGRATE = 1e-3
TRAINPERCENT = 0.7
SNR = 35

basePath = "/home/tonix/Documents/PhdDegreeCode/MatlabCode/PuertoSourceCode/MIMODataSet/"
# Load dataset
data = np.load(basePath+f"Signal_SNR_Rx_{SNR}.npy")  # Shape: (num_samples, num_rows, num_cols)
targets = np.load(basePath+f"Signal_SNR_Tx_{SNR}.npy")  # Shape: (num_samples, labels)

# Convert to PyTorch tensors
data_tensor = torch.tensor(data, dtype=torch.float32)  # Shape: (num_samples, num_rows, num_cols)
target_tensor = torch.tensor(targets, dtype=torch.long)  # Shape: (num_samples, labels)

# Init model
model = LitModel(lr = LEARNINGRATE)

# Init logger
logger = TensorBoardLogger("lightning_logs", name=f"MIMO_{SNR}_Epochs_{EPOCHS}_BS{BATCHSIZE}_LR{LEARNINGRATE}")

# Trainner Config
trainer = Trainer(
    max_epochs=EPOCHS,       # Epoch Number
    logger=logger,           # TensorBoard Logger
    accelerator="gpu",       # GPU
    devices=1,                # GPU num
)

train_dataset = RowDataset(data_tensor, target_tensor)

# Train size
train_size = int(TRAINPERCENT * len(train_dataset))  # 80% trainning
val_size = len(train_dataset) - train_size  # 20% validation

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE,num_workers=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE,num_workers=16, shuffle=False)

# Ejecutar el entrenamiento
trainer.fit(model, train_loader, val_loader)

torch.save(model.state_dict(), f"model_MIMO_DPSK_{SNR}.pth")
print(f"model_MIMO_DPSK_{SNR}.pth")

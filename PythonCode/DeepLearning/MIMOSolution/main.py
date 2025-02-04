from pytorch_lightning import Trainer
from LightningGym import LitModel
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader, random_split
from RowDataset import *

EPOCHS = 5
BATCHSIZE = 512
LEARNINGRATE = 1e-4
TRAINPERCENT = 0.7

basePath = "/home/tonix/Documents/PhdDegreeCode/MatlabCode/PuertoSourceCode/MIMODataSet/"
# Load dataset
data = np.load(basePath+"Signal_SNR_Rx_10.npy")  # Shape: (num_samples, num_rows, num_cols)
targets = np.load(basePath+"Signal_SNR_Tx_10.npy")  # Shape: (num_samples, labels)

# Convert to PyTorch tensors
data_tensor = torch.tensor(data, dtype=torch.float32)  # Shape: (num_samples, num_rows, num_cols)
target_tensor = torch.tensor(targets, dtype=torch.long)  # Shape: (num_samples, labels)

# Init model
model = LitModel(lr = LEARNINGRATE)

# Init logger
logger = TensorBoardLogger("lightning_logs", name="my_experiment")

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

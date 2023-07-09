import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Timer
from pytorch_lightning.callbacks import Callback


class MySlurmCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        slurm_id = os.getenv('SLURM_JOB_ID')
        slurm_rank = os.getenv('SLURM_PROCID')
        device_id = torch.cuda.current_device()
        print(f"SLURM_JOB_ID: {slurm_id}, SLURM_PROCID: {slurm_rank}, CUDA Device ID: {device_id}")


# Define the Lightning Module
class MNISTModel(LightningModule):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Prepare data
transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST("../../DDP/001/data/", train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64)
#, num_workers=4, pin_memory=True)

# Initialize a model
model = MNISTModel()

# Initialize a trainer
trainer = Trainer(
    max_epochs=10, 
    num_nodes=2,
    gpus=2, 
    callbacks=[
        EarlyStopping(monitor='train_loss'), 
        ModelCheckpoint(dirpath='checkpoints/', filename='{epoch}-{train_loss:.2f}'), 
        LearningRateMonitor(logging_interval='step'), 
        Timer(),
        MySlurmCallback(),
    ]
)

# Train the model 
trainer.fit(model, train_loader)

import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchsummary import summary
import torchmetrics
import visualtorch
from collections import defaultdict

from PIL import Image

# train_transforms = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.RandomHorizontalFlip(p=0.1),
#     transforms.RandomVerticalFlip(p=0.1),
#     transforms.RandomRotation(degrees=10),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
#
#
# val_transforms = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# train_dataset = ImageFolder("Lung Disease Dataset/train", transform=train_transforms)
# val_dataset = ImageFolder("Lung Disease Dataset/val", transform=val_transforms)
# test_dataset = ImageFolder("Lung Disease Dataset/test", transform=val_transforms)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity=None, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels*4,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(out_channels*4)
        )

        self.identity = identity
        self.relu = nn.ReLU()
        self.stride = stride



    def forward(self, x):
        identity_1 = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.identity is not None:
            identity_1 = self.identity(identity_1)
        x += identity_1
        x = self.relu(x)
        return x

class ResidualNetwork(pl.LightningModule):
    def __init__(self, learning_rate, batch_size, block, layers):
        super().__init__()

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=3)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.2)
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
          )

        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((16, 16))
        self.fc = nn.Linear(512*16*16*4, 3)

    def _make_layer(self, block, num_blocks, out_channels, stride):
        identity = None
        layers = []

        if stride != 1 or self.in_channels != out_channels*4:
              identity = nn.Sequential(
                  nn.Conv2d(
                      in_channels=self.in_channels,
                      out_channels=out_channels*4,
                      kernel_size=1,
                      stride=stride,
                      bias=False
                  ),
                  nn.BatchNorm2d(out_channels*4)
              )

        layers.append(block(self.in_channels, out_channels, identity, stride))
        self.in_channels = out_channels*4

        for i in range(num_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    # def train_dataloader(self):
    #     train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    #     return train_loader
    #
    # def val_dataloader(self):
    #     val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
    #     return val_loader
    #
    # def test_dataloader(self):
    #     test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
    #     return test_loader

    def forward(self, x):
      x = self.conv1(x)
      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)
      x = self.avgpool(x)
      x = nn.Flatten()(x)
      x = self.fc(x)
      return x

    def _common_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = self.accuracy(logits, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch)
        self.log_dict({'train_loss': loss, 'train_acc': acc}, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch)
        self.log_dict({'val_loss': loss, 'val_acc': acc}, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch)
        self.log_dict({'test_loss': loss, 'test_acc': acc}, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

# device = torch.device('cuda')
# #
bacha_size = 32
# #
# model = ResidualNetwork(
#     learning_rate=0.01,
#     batch_size=bacha_size,
#     block=ResidualBlock,
#     layers=[3, 5, 21, 5]
# ).to(device)
# #
# logger = TensorBoardLogger('lightning_logs', name='tensorboard')
# #
# print(train_dataset.classes)
# #
# # # summary(model, (3, 256, 256))
# #
# checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", filename='best-checkpoint', save_top_k=1)
#
class MyPrintingCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Starting training!")

    def on_train_end(self, trainer, pl_module):
        print("Finished training")
#
# trainer = pl.Trainer(
#
#     accelerator='gpu',
#
#     devices=1,
#
#     max_epochs=20,
#
#     logger=logger,
#
#     callbacks=[MyPrintingCallback(), checkpoint, EarlyStopping('val_loss')],
#
#     profiler='simple'
#
# )
#
# checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", filename='best-checkpoint', save_top_k=1)

# trainer.fit(model)
#
# input_shape = (1, 3, 512, 512)
#
#
#
# color_map: dict = defaultdict(dict)
#
# color_map[nn.Conv2d]["fill"] = "LightSlateGray"  # Light Slate Gray
#
# color_map[nn.ReLU]["fill"] = "#87CEFA"  # Light Sky Blue
#
# color_map[nn.MaxPool2d]["fill"] = "LightSeaGreen"  # Light Sea Green
#
# color_map[nn.Flatten]["fill"] = "#98FB98"  # Pale Green
#
# color_map[nn.Linear]["fill"] = "LightSteelBlue"  # Light Steel Blue

# img = visualtorch.layered_view(model, input_shape=input_shape, color_map=color_map,
#                                one_dim_orientation="x", spacing=40, legend=True)

# plt.figure(figsize=(25, 20))
#
# plt.axis("off")
#
# plt.tight_layout()
#
# plt.imshow(img)
#
# plt.show()

# start tensorboard


# print(trainer.callback_metrics)

# pretrain_model = ResidualNetwork.load_from_checkpoint(batch_size=bacha_size, learning_rate=0.001,
#
#                                                       block=ResidualBlock,
#
#                                             layers=[3, 5, 21, 5],
#
#                                           checkpoint_path=checkpoint.best_model_path)
#
# pretrain_model = pretrain_model.to('cuda')
#
# pretrain_model.eval()
#
# pretrain_model.freeze()
#
# model = ResidualNetwork.load_from_checkpoint(batch_size=bacha_size, learning_rate=0.001,
#
#                                              block=ResidualBlock,
#
#                                             layers=[3, 5, 21, 5],
#
#                                           checkpoint_path="lightning_logs/tensorboard/version_1/checkpoints/best-checkpoint.ckpt")
#
# torch.save(model.state_dict(), 'ResidualNetwork104-pytorch-lightning2.pt')
import logging
import os
from pathlib import Path

import click
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dotenv import find_dotenv, load_dotenv
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

# cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "saved_models/"


class ResNetModule(pl.LightningModule):
    def __init__(self, model):
        """Creates the Module

        Args:
            model (pytorch model): Model object
        """
        super().__init__()
        self.model = model

        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        optimizer_ft = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        return [optimizer_ft], [exp_lr_scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        inputs, labels = batch
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = self.loss_module(outputs, labels)
        acc = (preds == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        wandb.log({"train loss": loss, "train acc": acc})
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)
        wandb.log({"val acc": acc})

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)
        wandb.log({"test acc": acc})


def prepare_loaders(data_dir):
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "test"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=64, shuffle=True, num_workers=4
        )
        for x in ["train", "test"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
    return dataloaders, dataset_sizes


def train_model(model, dataloaders):
    # Create a PyTorch Lightning trainer with the generation callback
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(
        enable_checkpointing=False,
        logger=wandb_logger,
        # We run on a single GPU (if possible)
        gpus=1 if str(device) == "cuda" else 0,
        # How many epochs to train for if no patience is set
        max_epochs=10,
        callbacks=[
            # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
        ],  # Log learning rate every epoch
        progress_bar_refresh_rate=1,
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    trainer.logger._log_graph = (
        True  # If True, we plot the computation graph in tensorboard
    )
    trainer.logger._default_hp_metric = (
        None  # Optional logging argument that we don't need
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    pl.seed_everything(42)
    model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 19)
    model_ft = model_ft.to(device)

    model = ResNetModule(model_ft)
    trainer.fit(model, dataloaders["train"], dataloaders["test"])

    # Test best model on validation and test set
    test_result = trainer.test(model, dataloaders["test"], verbose=False)
    result = {"test": test_result[0]["test_acc"]}
    print(result)
    return model


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
def main(data_dir):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("training model from processed data")

    wandb.init(project="hyperkvasir")
    dataloaders, dataset_sizes = prepare_loaders(data_dir)

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 19)

    model_ft = model_ft.to(device)

    trained_model = train_model(model_ft, dataloaders)

    path = "models/version=2.pth"
    torch.save(trained_model, path)

    logging.info(f"Model was trained and saved: {path}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

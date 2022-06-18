import glob
import logging
from pathlib import Path

import click
import torch.nn as nn
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, models, transforms

from src.models.train_model_light import ResNetModule

# cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_loaders(data_dir):
    data_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_dataset = datasets.ImageFolder(data_dir, data_transform)
    dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=32, shuffle=False, num_workers=8
    )

    class_names = image_dataset.classes
    print(class_names)
    dataset_size = len(image_dataset)
    return dataloader, dataset_size


def predict_model(model, dataloader, dataset_size):
    ls = []
    model.eval()  # Set model to evaluate mode
    i = 0
    logging.info(f"Length in batches: {len(dataloader)}")
    logging.info("Started predicting...")
    # Iterate over data.
    for inputs, labels in dataloader:
        i += 1
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            ls.append(preds)

        if i % 50 == 0:
            logging.info(f"Batch processed {i}...")
    predictions = torch.cat(ls)
    return predictions


def get_lexem(s, pattern):
    lexems = s.split("/")
    for lexem in lexems:
        lexem_patterns = lexem.split("=")
        if pattern in lexem_patterns:
            return lexem.replace(f"{pattern}=", "").replace(".pth", "")
    return None


def get_most_recent_path(path=None):
    if path is None:
        paths = glob.glob("models/*.pth")
    else:
        paths = glob.glob(f"{path}/*.pth")
    versions = [(x, int(get_lexem(x, "version"))) for x in paths]
    versions = sorted(versions, reverse=True, key=lambda x: x[1])
    latest_version = versions[0]
    latest_path = latest_version[0]
    return latest_path


def get_model(model_path):
    num_classes = 19
    model_ft = models.resnet34(pretrained=True)
    model_ft.to(device)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    model = ResNetModule(model_ft)
    model.load_state_dict(torch.load(model_path))
    return model


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
def main(data_dir, model_path=None):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("predicting with the most recent model")

    if model_path is None:
        model_path = get_most_recent_path()
    model = get_model(model_path)

    dataloader, dataset_size = prepare_loaders(data_dir)
    predictions = predict_model(model, dataloader, dataset_size)
    print(predictions.shape)
    print(np.array(predictions))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

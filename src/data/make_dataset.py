# -*- coding: utf-8 -*-
import logging
import os
import shutil as sh
from collections import defaultdict
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split


def move_files(filelist, phase, input_filepath, output_filepath):
    i = 0
    for x_file in filelist:
        i += 1
        splits = str(x_file).split("/")
        splits.insert(3, phase)
        del splits[-3]
        del splits[-3]
        new_file = "/".join(splits)
        new_file = new_file.replace(input_filepath, output_filepath)
        path = os.path.dirname(new_file)
        if not os.path.exists(path):
            os.makedirs(path)
        sh.copy(x_file, new_file)
        if i % 1000 == 0:
            logging.info(f"Processed {i} files...")


def create_dict(data_dir):
    labels = set()
    for path in Path(data_dir).rglob("*.jpg"):
        labels.add(os.path.dirname(path).split("/")[-1])
    label_items_dict = defaultdict(list)
    for label in labels:
        label_items = []
        for path in Path(data_dir).rglob(f"*/{label}/*"):
            label_items.append(path)
        label_items_dict[label] = label_items

    return label_items_dict


def make_df(label_items_dict):
    tuples = []
    for k in label_items_dict:
        values = label_items_dict[k]
        for v in values:
            tuples.append((v, k))

    df = pd.DataFrame(tuples, columns=["path", "label"])
    X_train, X_test, y_train, y_test = train_test_split(
        df["path"], df["label"], test_size=0.2, stratify=df["label"]
    )
    return X_train, X_test


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    data_dir = os.path.join(input_filepath, "labeled-images", "lower-gi-tract")

    label_items_dict = create_dict(data_dir)
    X_train, X_test = make_df(label_items_dict)

    move_files(X_test, "test", input_filepath, output_filepath)
    move_files(X_train, "train", input_filepath, output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

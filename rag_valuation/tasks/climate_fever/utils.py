import datasets
import logging


def filter_binary(dataset: datasets.Dataset) -> datasets.Dataset:
    # Only supports and refutes
    return dataset.filter(lambda example: example["claim_label"] in [0,1])
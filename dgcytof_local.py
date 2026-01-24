"""
Local, trimmed-down copy of the minimal DGCyTOF helpers used by the CLI.
This avoids relying on the package import hierarchy that can break under
different Python path setups.
"""

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable


def get_num_correct(preds, labels):
    """Utility: count correctly predicted labels in a batch."""
    return preds.argmax(dim=1).eq(labels).sum().item()


def validate_model(
    model_fc,
    val_tensor,
    classes,
    params_val={"batch_size": 10000, "shuffle": False, "num_workers": 6},
):
    """
    Runs validation, printing per-class accuracy and returning a zip of results.
    Mirrors the original DGCyTOF.validate_model.
    """
    assert len(set(classes)) > 1, "There must be at least 2 classes"

    labels = len(classes)
    model_fc.eval()
    val_loader = data_utils.DataLoader(dataset=val_tensor, **params_val)

    class_correct = [0.0 for _ in range(labels)]
    class_total = [0.0 for _ in range(labels)]

    val_correct = 0
    val_total = 0

    for data in val_loader:
        val_samples, val_labels = data
        val_outputs = model_fc(Variable(val_samples))
        _, val_predicted = torch.max(val_outputs.data, 1)
        c = (val_predicted == val_labels).squeeze()
        for i in range(val_labels.shape[0]):
            label = val_labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

        val_total += val_labels.size(0)
        val_correct += (val_predicted == val_labels).sum()

    print("Accuracy:", round(100 * val_correct.item() / val_total, 4))
    print("-" * 100)
    for i in range(labels):
        if class_total[i] == 0:
            accuracy = "n/a"
        else:
            accuracy = round(100 * class_correct[i] / class_total[i], 3)
        print("Accuracy of {} : {}".format(classes[i], accuracy))

    return list(zip(val_predicted, val_labels, val_outputs))


def train_model(
    model_fc,
    X_train,
    max_epochs=20,
    params_train={"batch_size": 128, "shuffle": True, "num_workers": 6},
):
    """
    Train the model using CrossEntropyLoss + Adam, matching the original helper.
    """
    train_loader = data_utils.DataLoader(dataset=X_train, **params_train)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_fc.parameters(), lr=0.001)

    for epoch in range(max_epochs):
        total_loss = 0
        total_correct = 0

        for data in train_loader:
            samples, labels = data
            preds = model_fc(samples)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)

        print(
            "epoch",
            epoch,
            "total_correct:",
            total_correct,
            "loss:",
            loss.item(),
            "total_loss:",
            total_loss,
        )
    model_fc.eval()


def preprocessing(dataset, columns_to_remove=[]):
    """
    Prepare labeled/unlabeled splits as in the original helper.
    """
    data = dataset.drop(columns_to_remove, axis=1)
    data_labeled = data[data.label.notnull()]
    X_data_labeled = data_labeled.drop(["label"], axis=1)
    y_data = data_labeled["label"]
    data_unlabeled = data[data.label.isnull()].drop(["label"], axis=1)

    return X_data_labeled, y_data, data_unlabeled

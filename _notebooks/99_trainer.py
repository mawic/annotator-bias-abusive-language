import argparse
import json
import os

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class HateDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    f1_score_micro = f1_score(labels, preds, average="micro")
    f1_score_macro = f1_score(labels, preds, average="macro")
    f1_score_weighted = f1_score(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_micro": f1_score_micro,
        "f1_macro": f1_score_macro,
        "f1_weighted": f1_score_weighted,
    }


def train(input_label, input_dataset, input_path):
    MODEL_NAME = "distilbert-base-uncased"
    SEED = 42

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    dataset_name = input_dataset
    dataset = load_dataset(
        "pandas",
        data_files={
            "train": f"{input_path}_train.pkl",
            "test": f"{input_path}_test.pkl",
            "validation": f"{input_path}_validation.pkl",
        },
    )

    label_field = input_label
    # tokenize data
    train_encodings = tokenizer(dataset["train"]["text"], truncation=True, padding=True)
    val_encodings = tokenizer(
        dataset["validation"]["text"], truncation=True, padding=True
    )
    test_encodings = tokenizer(dataset["test"]["text"], truncation=True, padding=True)

    train_dataset = HateDataset(train_encodings, dataset["train"][label_field])
    val_dataset = HateDataset(val_encodings, dataset["validation"][label_field])
    test_dataset = HateDataset(test_encodings, dataset["test"][label_field])

    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        num_train_epochs=3,  # total # of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        seed=SEED,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
    trainer.model.save_pretrained(f"../tmp/models/{dataset_name}/{label_field}/")


def main():

    parser = argparse.ArgumentParser(description="List the content of a folder")

    parser.add_argument("-l", "--label", help="label")
    parser.add_argument("-d", "--dataset", help="dataset")
    parser.add_argument("-p", "--path", help="path")

    args = parser.parse_args()

    input_label = args.label
    input_dataset = args.dataset
    input_path = args.path

    # train and test model
    train(input_label, input_dataset, input_path)


if __name__ == "__main__":
    main()

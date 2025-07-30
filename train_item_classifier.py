"""
Script to train a text classification model on product names.

This example shows how to fine-tune a Transformer (e.g., BERT) to categorize items based on their name.

Usage:
    python train_item_classifier.py \
        --train_file data/train.csv \
        --model_name_or_path bert-base-uncased \
        --output_dir models/item_classifier \
        --num_train_epochs 3
(This example auto-splits 10% for validation and infers label count)

Requirements:
    pip install pandas datasets transformers torch scikit-learn
"""
import argparse
import os
import pandas as pd
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Transformer for item categorization"
    )
    parser.add_argument("--train_file", type=str, required=True,
                        help="File with columns 'text' and 'label' for training")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased",
                        help="Pretrained model name or path")
    parser.add_argument("--output_dir", type=str, default="./item_classifier_model",
                        help="Where to save the fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Load data
    # Load data (CSV or Excel)
    def load_df(path):
        if path.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(path, engine='openpyxl')
        return pd.read_csv(path)
    df = load_df(args.train_file)
    # Automatically infer number of labels and split dataset
    num_labels = int(df['Libellé produit'].nunique())
    print(f"Detected {num_labels} classes")
    ds = Dataset.from_pandas(df)
    split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split['train']
    valid_ds = split['test']

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels  # num_labels is now inferred from the data
    ).to(device)

    # Preprocessing
    def preprocess(batch):
        enc = tokenizer(
            batch['text'],
            padding='max_length',
            truncation=True,
            max_length=128
        )
        enc['labels'] = batch['label']
        return enc

    train_ds = train_ds.map(preprocess, batched=True)
    valid_ds = valid_ds.map(preprocess, batched=True)

    # Metric
    metric = evaluate.load('accuracy')
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return metric.compute(predictions=preds, references=labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.learning_rate,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy'
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()
    # Save
    trainer.save_model(args.output_dir)

if __name__ == '__main__':
    main()

"""
Script to train a sequence-to-sequence model to extract dimensions from "Libellé produit".

This script reads a CSV file with columns "Libellé produit" and "dimensions_extraites",
prepares a dataset, fine-tunes a pretrained T5 model, and saves the trained model.

Usage:
    python train_dimension_extractor.py \
        --input_csv data/libelle_dimensions.csv \
        --model_name_or_path t5-small \
        --output_dir models/dim_extractor \
        --num_train_epochs 3

Dependencies:
    pip install pandas datasets transformers torch
"""
import argparse
import os
import pandas as pd
from datasets import Dataset
import evaluate
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments
)
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model to extract dimensions from product labels"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the file containing 'Libellé produit' and 'dimensions_extraites' columns"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default="t5-small",
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./dim_extractor_model",
        help="Directory to save trained model and checkpoints"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=16
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=16
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Ensure training uses CUDA GPU if available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected, training on CPU.")

    # Load data
    # Load data from CSV or Excel
    input_path = args.input
    if input_path.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(input_path, engine='openpyxl')
    else:
        df = pd.read_csv(input_path)
    df = df[["Libellé produit", "dimensions_extraites"]]
    df["dimensions_extraites"] = df["dimensions_extraites"].fillna("None")
    df = df.rename(columns={
        "Libellé produit": "input_text",
        "dimensions_extraites": "target_text"
    })

    # Convert to Hugging Face dataset and train-test split
    ds = Dataset.from_pandas(df)
    split = ds.train_test_split(test_size=0.1, seed=42)

    # Load tokenizer and model
    tokenizer = T5TokenizerFast.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

    # Preprocessing function
    def preprocess(batch):
        inputs = ["extract dimensions: " + text for text in batch['input_text']]
        model_inputs = tokenizer(
            inputs,
            max_length=128,
            padding='max_length',
            truncation=True
        )
        labels = tokenizer(
            batch['target_text'],
            max_length=32,
            padding='max_length',
            truncation=True
        )
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    # Tokenize
    tokenized = split.map(preprocess, batched=True)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        # Remove evaluation_steps and save_steps due to transformers version compatibility
        fp16=True,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01
    )

    # Define metric for evaluation
    accuracy_metric = evaluate.load('accuracy')
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Compute simple match accuracy
        matches = [int(p == l) for p, l in zip(decoded_preds, decoded_labels)]
        return {'accuracy': sum(matches) / len(matches)}

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train and save
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()

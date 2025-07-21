from transformers import GPT2Tokenizer
from transformers import GPT2ForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
from datasets import Dataset
import evaluate
import numpy as np

def load_and_prepare_dataset(json_path, csv_path, test_size=0.2, val_size=0.2):
    df = pd.read_json(json_path, lines=True)
    labels_df = pd.read_csv(csv_path, sep=";")

    labels_df["label"] = labels_df["real_news"].replace({"yes": 1, "no": 0})
    adjusted_index = labels_df["index"] - 1
    df.loc[adjusted_index, "labels"] = labels_df["label"].values
    df["labels"] = df["labels"].astype(int)

    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=test_size)
    train_all = dataset['train']
    test = dataset['test']

    dataset_train = train_all.train_test_split(test_size=val_size)
    train = dataset_train['train']
    val = dataset_train['test']

    return train, val, test 


def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  ##all sentences have to be the same length
    return tokenizer


def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(example):
        return tokenizer(example["Text"], padding = 'max_length', truncation = True)
    
    return dataset.map(tokenize_function, batched=True)


def load_model(tokenizer, num_labels=2):
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels = 2)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model



metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)  ##output form raw model is converted to 0 1 0 0 ....
    return metric.compute(predictions=predictions, references=labels)


def get_training_args(output_dir = "./results"):
    return TrainingArguments(
        output_dir = output_dir,
        eval_strategy="epoch",
        save_strategy="epoch", 
        logging_strategy="epoch",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2)
        


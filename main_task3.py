from transformers import Trainer
from llm_model import (get_tokenizer, load_and_prepare_dataset, load_model, tokenize_dataset, get_training_args, compute_metrics)


def train():
    json_path = "without_assessment_updated.jsonl"
    csv_path = "group31_stage1.csv"
    train_dataset, val_dataset, test_dataset = load_and_prepare_dataset(json_path, csv_path)

    tokenizer = get_tokenizer()
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    tokenized_val = tokenize_dataset(val_dataset, tokenizer)

    model = load_model(tokenizer)
    training_args = get_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    train()
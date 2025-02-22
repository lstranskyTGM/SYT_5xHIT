from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import numpy


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def compute_metrics(eval_preds, label_names, metric):
    logits, labels = eval_preds
    predictions = numpy.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[label_id] for label_id in label if label_id != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return (
        f"precision: {all_metrics['overall_precision']}\n"
        f"recall: {all_metrics['overall_recall']}\n"
        f"f1: {all_metrics['overall_f1']}\n"
        f"accuracy: {all_metrics['overall_accuracy']}"
    )


def main():
    # Load the raw datasets
    raw_datasets = load_dataset("conll2003", trust_remote_code=True)

    # Load the tokenizer
    model_checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Extract the label names
    ner_feature = raw_datasets["train"].features["ner_tags"]
    label_names = ner_feature.feature.names

    # Tokenize and align the labels for the entire dataset
    def map_function(examples):
        return tokenize_and_align_labels(examples, tokenizer)

    tokenized_datasets = raw_datasets.map(
        map_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    # Load the data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Load the metric
    metric = evaluate.load("seqeval")

    # Create a mapping from label id to label name
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    # Load the model
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
    )

    # Define the training arguments
    args = TrainingArguments(
        "bert-finetuned-ner",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=True,
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, label_names, metric),
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    save_directory = "bert-ner-conll2003"
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved in {save_directory}")

    # Push the model to the hub
    trainer.push_to_hub(commit_message="Fine-tuned BERT model for NER on CoNLL-2003 dataset")


if __name__ == "__main__":
    main()

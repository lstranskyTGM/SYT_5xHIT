---
library_name: transformers
license: apache-2.0
base_model: bert-base-cased
tags:
- generated_from_trainer
datasets:
- conll2003
metrics:
- precision
- recall
- f1
- accuracy
model-index:
- name: bert-finetuned-ner
  results:
  - task:
      name: Token Classification
      type: token-classification
    dataset:
      name: conll2003
      type: conll2003
      config: conll2003
      split: validation
      args: conll2003
    metrics:
    - name: Precision
      type: precision
      value: 0.932892561983471
    - name: Recall
      type: recall
      value: 0.9498485358465163
    - name: F1
      type: f1
      value: 0.9412941961307538
    - name: Accuracy
      type: accuracy
      value: 0.9860334373344322
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert-finetuned-ner

This model is a fine-tuned version of [bert-base-cased](https://huggingface.co/bert-base-cased) on the conll2003 dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0632
- Precision: 0.9329
- Recall: 0.9498
- F1: 0.9413
- Accuracy: 0.9860

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step | Validation Loss | Precision | Recall | F1     | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:---------:|:------:|:------:|:--------:|
| 0.0759        | 1.0   | 1756 | 0.0716          | 0.8972    | 0.9300 | 0.9133 | 0.9814   |
| 0.0351        | 2.0   | 3512 | 0.0754          | 0.9304    | 0.9424 | 0.9364 | 0.9838   |
| 0.0221        | 3.0   | 5268 | 0.0632          | 0.9329    | 0.9498 | 0.9413 | 0.9860   |


### Framework versions

- Transformers 4.47.1
- Pytorch 2.5.1+cpu
- Datasets 3.2.0
- Tokenizers 0.21.0

#  PEGASUS Summarization on SAMSum Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/156QAcdh8hBxCpIrRqDuAJGY_bPRdsQYl)

---

<p align="center">
  <img src="https://storage.googleapis.com/gweb-cloudblog-publish/images/Google_Cloud_logo.width-400.png" height="70">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="70">
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg" height="70">
  <img src="https://upload.wikimedia.org/wikipedia/en/3/33/NVIDIA_logo.svg" height="70">
</p>

---

##  Project Overview

This project demonstrates how to **fine-tune the PEGASUS model** (`google/pegasus-cnn_dailymail`) for **dialogue summarization** using the **SAMSum dataset**.
The notebook is designed for execution on **Google Colab** and leverages **Transformers**, **Datasets**, **NVIDIA GPU**, and **PyTorch** frameworks.

---

##  Environment Setup

Run the following command to check GPU availability:

```bash
!nvidia-smi
```

Install the required libraries:

```bash
!pip install transformers[sentencepiece] datasets sacrebleu rough_score py7zr -q
!pip install --upgrade accelerate
!pip uninstall -y transformers accelerate
!pip install transformers accelerate
```

---

##  Import Dependencies

```python
from transformers import pipeline, set_seed
from datasets import load_dataset, load_from_disk, load_metric
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import torch

nltk.download('punkt')
```

Select computation device:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

##  Load PEGASUS Model and Tokenizer

```python
model_ckpt = "google/pegasus-cnn_dailymail"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
```

---

##  Load and Explore SAMSum Dataset

```python
dataset_samsum = load_dataset("samsum")
dataset_samsum
```

View sample data:

```python
print(dataset_samsum['train']['dialogue'][1])
print(dataset_samsum['train'][1]['summary'])
```

---

##  Preprocess and Tokenize Data

The following function converts dialogues and summaries into tokenized input features.

```python
def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch['summary'], max_length=128, truncation=True)

    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }
```

Apply mapping:

```python
dataset_samsum_pt = dataset_samsum.map(convert_examples_to_features, batched=True)
```

---

##  Data Collation

```python
from transformers import DataCollatorForSeq2Seq
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
```

---

##  Training Setup

Define training arguments and the trainer:

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="pegasus-samsum",
    warmup_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1e6,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=10,
    gradient_accumulation_steps=16
)

trainer = Trainer(
    model=model_pegasus,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=seq2seq_data_collator,
    train_dataset=dataset_samsum_pt["train"],
    eval_dataset=dataset_samsum_pt["test"]
)
```

---

##  Train the Model

```python
trainer.train()
```

---

##  Evaluate the Model

After training, you can generate and compare summaries using the fine-tuned PEGASUS model.

Example:

```python
dialogue = dataset_samsum["test"][1]["dialogue"]
inputs = tokenizer([dialogue], max_length=1024, truncation=True, return_tensors="pt").to(device)
summary_ids = model_pegasus.generate(**inputs)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
```

---

##  References

* [PEGASUS on Hugging Face](https://huggingface.co/google/pegasus-cnn_dailymail)
* [SAMSum Dataset](https://huggingface.co/datasets/samsum)
* [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai)
* [Transformers Documentation](https://huggingface.co/docs/transformers/index)

---
Developed by **Chiejina Chike Obinna**.

---

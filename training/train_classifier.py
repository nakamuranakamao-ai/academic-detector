"""Fine-tune classifier dari CSV berkolom text,label; label 0=human, 1=ai.
Gunakan dataset yang berizin, representatif, dan seimbang. Hasil tidak otomatis valid untuk semua domain.
Contoh: python training/train_classifier.py --csv training/dataset.csv --base-model xlm-roberta-base
"""
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

p = argparse.ArgumentParser()
p.add_argument('--csv', required=True)
p.add_argument('--base-model', default='xlm-roberta-base')
p.add_argument('--output', default='models/ai-classifier')
p.add_argument('--epochs', type=int, default=3)
a = p.parse_args()
df = pd.read_csv(a.csv).dropna(subset=['text', 'label'])
df['label'] = df['label'].astype(int)
if set(df.label.unique()) != {0, 1}:
    raise SystemExit('CSV harus memiliki label 0 (human) dan 1 (AI).')
train_df, val_df = train_test_split(df[['text','label']], test_size=0.2, random_state=42, stratify=df.label)
tokenizer = AutoTokenizer.from_pretrained(a.base_model)
def tok(batch): return tokenizer(batch['text'], truncation=True, max_length=512)
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).map(tok, batched=True)
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True)).map(tok, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(a.base_model, num_labels=2,
    id2label={0:'human',1:'ai'}, label2id={'human':0,'ai':1})
def metrics(pred):
    y = pred.label_ids; p = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(y,p,average='binary',zero_division=0)
    return {'accuracy':accuracy_score(y,p),'precision_ai':precision,'recall_ai':recall,'f1_ai':f1}
args = TrainingArguments(output_dir='training/checkpoints', num_train_epochs=a.epochs,
    per_device_train_batch_size=8, per_device_eval_batch_size=8,
    learning_rate=2e-5, weight_decay=0.01, eval_strategy='epoch', save_strategy='epoch',
    load_best_model_at_end=True, metric_for_best_model='f1_ai', report_to='none')
trainer = Trainer(model=model,args=args,train_dataset=train_ds,eval_dataset=val_ds,
    tokenizer=tokenizer,data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=metrics)
trainer.train()
print('Validation:', trainer.evaluate())
trainer.save_model(a.output); tokenizer.save_pretrained(a.output)
print('Model saved to', a.output)

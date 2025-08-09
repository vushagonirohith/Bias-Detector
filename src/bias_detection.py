from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
from datasets import Dataset
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# Load the pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=9)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load and process dataset
df = pd.read_csv(r'D:\bias-detection-project\data\crowspairs.csv')
bias_types = ["race-color", "gender", "sexual-orientation", "religion", "age", 
              "nationality", "disability", "physical-appearance", "socioeconomic"]
df['bias_type'] = df['bias_type'].apply(lambda x: bias_types.index(x))

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['sent_more'], padding='max_length', truncation=True, max_length=512)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[['sent_more', 'bias_type']])
dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.rename_column("bias_type", "labels")

# Split dataset
train_dataset, val_dataset = dataset.train_test_split(test_size=0.1).values()

# Compute metrics for evaluation
def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"precision": precision, "recall": recall, "f1": f1}

# Training arguments
training_args = TrainingArguments(
    output_dir='./models',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,  # Load best model at the end
    metric_for_best_model="f1",   # Use F1 to determine best model
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the best model and tokenizer
trainer.save_model('./models/bias_detection_model')
tokenizer.save_pretrained('./models/bias_detection_model')
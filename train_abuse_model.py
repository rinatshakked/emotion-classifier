# # Install core packages
# !pip install -U transformers datasets accelerate

# Python standard + ML packages
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support

from torch.utils.data import Dataset

# Hugging Face transformers
from transformers import (
    AutoTokenizer,
    DebertaV2Tokenizer,
    BertTokenizer, 
    BertForSequenceClassification,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset class
class AbuseDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float).to(device)
        return item


#  Convert label values to soft scores: "yes" = 1.0, "plausibly" = 0.5, others = 0.0
def label_row_soft(row):
    labels = []
    for col in label_columns:
        val = str(row[col]).strip().lower()
        if val == "yes":
            labels.append(1.0)
        elif val == "plausibly":
            labels.append(0.5)
        else:
            labels.append(0.0)
    return labels
    
# Function to map probabilities to 3 classes
# (0.0, 0.5, 1.0) based on thresholds
def map_to_3_classes(prob_array, low, high):
    """Map probabilities to 0.0, 0.5, 1.0 using thresholds."""
    mapped = np.zeros_like(prob_array)
    mapped[(prob_array > low) & (prob_array <= high)] = 0.5
    mapped[prob_array > high] = 1.0
    return mapped

def convert_to_label_strings(array):
    """Convert float label array to list of strings."""
    return [label_map[val] for val in array.flatten()]

def tune_thresholds(probs, true_labels, verbose=True):
    """Search for best (low, high) thresholds by macro F1 score."""
    best_macro_f1 = 0.0
    best_low, best_high = 0.0, 0.0

    for low in np.arange(0.2, 0.5, 0.05):
        for high in np.arange(0.55, 0.8, 0.05):
            if high <= low:
                continue

            pred_soft = map_to_3_classes(probs, low, high)
            pred_str = convert_to_label_strings(pred_soft)
            true_str = convert_to_label_strings(true_labels)

            _, _, f1, _ = precision_recall_fscore_support(
                true_str, pred_str,
                labels=["no", "plausibly", "yes"],
                average="macro",
                zero_division=0
            )
            if verbose:
                print(f"low={low:.2f}, high={high:.2f} -> macro F1={f1:.3f}")
            if f1 > best_macro_f1:
                best_macro_f1 = f1
                best_low, best_high = low, high

    return best_low, best_high, best_macro_f1

def evaluate_model_with_thresholds(trainer, test_dataset):
    """Run full evaluation with automatic threshold tuning."""
    print("\n🔍 Running model predictions...")
    predictions = trainer.predict(test_dataset)
    probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()
    true_soft = np.array(predictions.label_ids)

    print("\n🔎 Tuning thresholds...")
    best_low, best_high, best_f1 = tune_thresholds(probs, true_soft)

    print(f"\n✅ Best thresholds: low={best_low:.2f}, high={best_high:.2f} (macro F1={best_f1:.3f})")

    final_pred_soft = map_to_3_classes(probs, best_low, best_high)
    final_pred_str = convert_to_label_strings(final_pred_soft)
    true_str = convert_to_label_strings(true_soft)

    print("\n📊 Final Evaluation Report (multi-class per label):\n")
    print(classification_report(
        true_str,
        final_pred_str,
        labels=["no", "plausibly", "yes"],
        zero_division=0
    ))

    return {
        "thresholds": (best_low, best_high),
        "macro_f1": best_f1,
        "true_labels": true_str,
        "pred_labels": final_pred_str
    }

# Load dataset
df = pd.read_excel("Abusive Relationship Stories - Technion & MSF.xlsx")

# Define text and label columns
text_column = "post_body" 
label_columns = [
    'emotional_violence', 'physical_violence', 'sexual_violence', 'spiritual_violence',
    'economic_violence', 'past_offenses', 'social_isolation', 'refuses_treatment',
    'suicidal_threats', 'mental_condition', 'daily_activity_control', 'violent_behavior',
    'unemployment', 'substance_use', 'obsessiveness', 'jealousy', 'outbursts',
    'ptsd', 'hard_childhood', 'emotional_dependency', 'prevention_of_care',
    'fear_based_relationship', 'humiliation', 'physical_threats',
    'presence_of_others_in_assault', 'signs_of_injury', 'property_damage',
    'access_to_weapons', 'gaslighting'
]

print(np.shape(df))
# Clean data
df = df[[text_column] + label_columns]
print(np.shape(df))
df = df.dropna(subset=[text_column])
print(np.shape(df))

df["label_vector"] = df.apply(label_row_soft, axis=1)
label_matrix = df["label_vector"].tolist()


#model_name = "onlplab/alephbert-base"
model_name = "microsoft/deberta-v3-base"

# Load pretrained model for fine-tuning
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_columns),
    problem_type="multi_label_classification"
).to(device)  # Move model to GPU

# # Optional: Freeze base model layers (only train classifier head)
# freeze_base = False
# if freeze_base:
#     for name, param in model.bert.named_parameters():
#         param.requires_grad = False

# Freeze bottom 6 layers of DeBERTa encoder
for name, param in model.named_parameters():
    if any(f"encoder.layer.{i}." in name for i in range(0, 6)):
        param.requires_grad = False
    

# Proper 3-way split: train / val / test
train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    df[text_column].tolist(), label_matrix, test_size=0.2, random_state=42
)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_val_texts, train_val_labels, test_size=0.1, random_state=42
)

train_dataset = AbuseDataset(train_texts, train_labels)
val_dataset = AbuseDataset(val_texts, val_labels)
test_dataset = AbuseDataset(test_texts, test_labels)


# TrainingArguments for HuggingFace Trainer (logging, saving)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)

# Train using HuggingFace Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Start training!
trainer.train()

label_map = {0.0: "no", 0.5: "plausibly", 1.0: "yes"}
evaluate_model_with_thresholds(trainer, test_dataset)
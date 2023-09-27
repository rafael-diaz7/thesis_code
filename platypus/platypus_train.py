
# This script is a minimal example of using the Platypus2-7B model from the
# HuggingFace Transformers library to classify movie reviews from the IMDb

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext.data as data
from torchtext.datasets import IMDB
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# Define the tokenizer and model
model_name = "garage-bAInd/Platypus2-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load the IMDb dataset using torchtext
TEXT = data.Field(tokenize=tokenizer.tokenize, batch_first=True, include_lengths=True)
LABEL = data.LabelField(dtype=torch.long)

train_data, test_data = IMDB.splits(TEXT, LABEL)

TEXT.build_vocab(train_data, max_size=10000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

# Create iterators for training and testing
BATCH_SIZE = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),  # Sort by text length to minimize padding
    device=device
)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Move model to the device (GPU, if available)
model = model.to(device)

# Training loop
NUM_EPOCHS = 1

for epoch in range(NUM_EPOCHS):
    print(f"Start of Epoch {epoch}")
    model.train()
    running_loss = 0.0

    for index, batch in enumerate(train_iterator):
        print(f"Start of batch {index}")
        text, text_lengths = batch.text
        labels = batch.label

        optimizer.zero_grad()

        outputs = model(text, attention_mask=(text != 1).float())  # Avoid attention to padding tokens
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {running_loss / len(train_iterator):.4f}")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_iterator:
        text, text_lengths = batch.text
        labels = batch.label

        outputs = model(text, attention_mask=(text != 1).float())
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")
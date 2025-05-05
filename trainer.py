import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformer import SentenceTransformer
from multi_task_transformer import MultiTaskTransformer

# Initialize the base model
base_model = SentenceTransformer('distilbert-base-uncased')
multi_task_model = MultiTaskTransformer(base_model)


sentences = [
    "I love programming in Python.",             # Positive
    "I hate when my favorite team loses.",         # Negative
    "I am passionate about nature.",             # Positive
]

# Labels
labels_a = torch.tensor([1, 0, 1])  # Task A: Sentiment (Positive=1, Negative=0)
labels_b = torch.tensor([1, 1, 2])  # Task B: Topic (Tech=1, Environment=2)

# Tokenize sentences
tokenizer = base_model.tokenizer
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Forward pass
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Training setup
optimizer = optim.Adam(multi_task_model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

multi_task_model.train()
optimizer.zero_grad()

# Forward pass through model
task_a_output, task_b_output = multi_task_model(input_ids, attention_mask)

# Compute losses
loss_a = criterion(task_a_output, labels_a)
loss_b = criterion(task_b_output, labels_b)
total_loss = loss_a + loss_b

# Backpropagation
total_loss.backward()
optimizer.step()

# Inference (for display)
predictions_a = torch.argmax(task_a_output, dim=1)
predictions_b = torch.argmax(task_b_output, dim=1)

print(f"Task A (Sentiment) Predictions: {predictions_a}")
print(f"Task B (Topic) Predictions: {predictions_b}")
print(f"Total Loss: {total_loss.item():.4f}")
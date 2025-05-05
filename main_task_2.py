import torch
from sentence_transformer import SentenceTransformer
from multi_task_transformer import MultiTaskTransformer
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

def train_joint(model, sentences, labels_a, labels_b, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    # Tokenizes sentences
    inputs = model.base_model.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Forward passing for  tasks
    task_a_output, task_b_output = model(input_ids, attention_mask)

    # Computes losses for both tasks
    loss_a = criterion(task_a_output, labels_a)
    loss_b = criterion(task_b_output, labels_b)
    total_loss = loss_a + loss_b

    # Backpropagation
    total_loss.backward()
    optimizer.step()

    return total_loss.item()

def demo_multi_task():
    base_model = SentenceTransformer('distilbert-base-uncased')  
    
    model = MultiTaskTransformer(base_model=base_model)

    
    sentences = [
        "I love programming in Python.",
        "Machine learning is transforming industries."
    ]
    labels_a = torch.tensor([0, 1])  
    labels_b = torch.tensor([1, 2])  

    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(5):
        loss = train_joint(model, sentences, labels_a, labels_b, optimizer, criterion)
        print(f"Epoch {epoch+1}, Loss: {loss}")

    # Forward pass through the model
    tokenizer = model.base_model.tokenizer
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    task_a_output, task_b_output = model(inputs['input_ids'], inputs['attention_mask'])

    print("\nMulti-Task Transformer Demo:")
    print(f"Task A output: {task_a_output}")
    print(f"Task B output: {task_b_output}")

if __name__ == "__main__":
    demo_multi_task()
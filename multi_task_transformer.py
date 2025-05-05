import torch
import torch.nn as nn
from sentence_transformer import SentenceTransformer

class MultiTaskTransformer(nn.Module):
    def __init__(self, base_model, num_classes_a=2, num_classes_b=3):
        super(MultiTaskTransformer, self).__init__()
        self.base_model = base_model
        self.hidden_size = 768  # For distilbert-base-uncased

        self.classification_head_a = nn.Linear(self.hidden_size, num_classes_a)
        self.classification_head_b = nn.Linear(self.hidden_size, num_classes_b)

    def forward(self, input_ids, attention_mask):
        # Get the last hidden state from the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)

        # Use the [CLS]-like token (first token's embedding)
        cls_embedding = last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size)

        task_a_output = self.classification_head_a(cls_embedding)
        task_b_output = self.classification_head_b(cls_embedding)

        return task_a_output, task_b_output

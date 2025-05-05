import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling

class SentenceTransformer(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", embedding_dim=768):
        super().__init__()
        # Loads a pre-trained transformer model and tokenizer
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_dim = embedding_dim
        
        
        if embedding_dim != self.transformer.config.hidden_size:
            self.projection = nn.Linear(self.transformer.config.hidden_size, embedding_dim)
        else:
            self.projection = nn.Identity()

    def mean_pooling(self, model_output, attention_mask):
         # Apply mean pooling over the token embeddings, taking the attention mask into account
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
          # Gets transformer outputs and applies mean pooling
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.mean_pooling(outputs, attention_mask)
        return BaseModelOutputWithPooling(last_hidden_state=outputs.last_hidden_state, pooler_output=self.projection(pooled_output))

    def encode(self, sentences, batch_size=32):
        #Function encodes sentences into vector representations 
        self.eval()
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")

            with torch.no_grad():
                embeddings = self(inputs["input_ids"], inputs["attention_mask"]).pooler_output
                all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)
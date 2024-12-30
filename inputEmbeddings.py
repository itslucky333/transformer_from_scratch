import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int , vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    


d_model = 16  # Embedding dimension
vocab_size = 50  # Vocabulary size

# Creating an instance of the class
embedding_layer = InputEmbeddings(d_model=d_model, vocab_size=vocab_size)

# Input tensor (batch_size=2, seq_len=5)
input_tensor = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

# Forward pass
output = embedding_layer(input_tensor)

print(output.shape)
# print(output)
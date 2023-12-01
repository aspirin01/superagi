
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query*keys for each training example and head
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        for pos in range(max_len):
            for i in range(0, embed_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_size)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / embed_size)))
        self.pe = pe.unsqueeze(0)
        self.register_buffer('pe', self.pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_size):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size, ff_hidden_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, followed by layer normalization
        x = self.norm1(attention + query)
        forward = self.ff(x)

        # Add skip connection, followed by layer normalization
        out = self.norm2(forward + x)
        return out


class GPT2(nn.Module):
    def __init__(self, embed_size, num_layers, heads, ff_hidden_size, vocab_size, max_length):
        super(GPT2, self).__init__()
        self.embed_size = embed_size
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, ff_hidden_size)
                for _ in range(num_layers)
            ]
        )
        self.pos_encoding = PositionalEncoding(embed_size, max_length)
        self.word_embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x, mask):
        x = self.word_embedding(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x, x, x, mask)
        return x

model = GPT2(
    embed_size=768,
    num_layers=12,
    heads=12,
    ff_hidden_size=3072,
    vocab_size=50257,  # Size of GPT-2 vocabulary
    max_length=1024    # Maximum sequence length
)

# Example input (tokenized text)
input_ids = torch.tensor([[50256, 318, 7039, 11, 290, 262, 50256]])  # Sample token IDs

# Forward pass
output = model(input_ids, mask=None)

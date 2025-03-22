import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm

# 🔹 Input Embedding Layer
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)  # Shape: [batch_size, seq_len, embed_dim]

# 🔹 Custom Processing Layer (Insert your own transformation)
class CustomProcessingLayer(nn.Module):
    def __init__(self, embed_dim):
        super(CustomProcessingLayer, self).__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)  # Example processing

    def forward(self, x):
        return F.relu(self.linear(x))  # Example: Applying a non-linearity

# 🔹 Transformer Model (Encoder-Decoder)
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(TransformerModel, self).__init__()

        # Embedding Layers
        self.input_embedding = InputEmbedding(vocab_size, embed_dim)
        self.output_embedding = InputEmbedding(vocab_size, embed_dim)

        # Custom Processing Layer (Add your transformation here)
        self.processing_layer = CustomProcessingLayer(embed_dim)

        # Transformer Encoder-Decoder
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        # Final output layer
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.input_embedding(src)
        tgt = self.output_embedding(tgt)

        # Custom Processing
        src = self.processing_layer(src)
        tgt = self.processing_layer(tgt)

        # Transformer Encoder-Decoder
        output = self.transformer(src, tgt, src_mask, tgt_mask)

        return self.fc_out(output)  # Shape: [batch, seq_len, vocab_size]

# 🔹 Load AG News Dataset
dataset = load_dataset("ag_news")
tokenizer = get_tokenizer("basic_english")

# 🔹 Build Vocabulary
def yield_tokens(data_iter):
    for example in data_iter:
        yield tokenizer(example["text"])

vocab = build_vocab_from_iterator(yield_tokens(dataset["train"]), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])
pad_idx = vocab["<pad>"]

# 🔹 Tokenization Function
def encode_text(text, seq_len=20):
    tokens = tokenizer(text)
    indices = [vocab[token] for token in tokens]
    if len(indices) < seq_len:
        indices += [pad_idx] * (seq_len - len(indices))
    return torch.tensor(indices[:seq_len])

# 🔹 Prepare Training Data
def collate_fn(batch):
    src_texts = [encode_text(item["text"]) for item in batch]
    tgt_texts = [encode_text(item["text"]) for item in batch]
    return torch.stack(src_texts), torch.stack(tgt_texts)

# 🔹 Hyperparameters
vocab_size = len(vocab)
embed_dim = 256
num_heads = 4
num_layers = 3
ff_dim = 512
dropout = 0.1
batch_size = 32
num_epochs = 3
learning_rate = 0.001

# 🔹 Create Dataloader
from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# 🔹 Initialize Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(vocab_size, embed_dim, num_heads, num_layers, ff_dim, dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 🔹 Training Loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for src, tgt in tqdm(train_dataloader):
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()
        output = model(src, tgt)

        loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_dataloader):.4f}")

# 🔹 Inference (Generate Text)
def generate_text(model, seed_text, max_length=20):
    model.eval()
    input_tokens = encode_text(seed_text).unsqueeze(0).to(device)
    output_tokens = input_tokens.clone()

    for _ in range(max_length - len(input_tokens[0])):
        output = model(input_tokens, output_tokens)
        next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
        output_tokens = torch.cat((output_tokens, next_token), dim=1)

    return " ".join([vocab.get_itos()[idx] for idx in output_tokens.squeeze().tolist()])

# 🔹 Test Text Generation
seed_text = "Breaking news: AI is taking over the world"
generated_text = generate_text(model, seed_text)
print("Generated Text:", generated_text)
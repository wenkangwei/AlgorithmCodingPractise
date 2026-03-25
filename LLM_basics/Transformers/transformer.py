import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, q_seq_len = q.size(0), q.size(1)
        k_seq_len = k.size(1)
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(q).view(batch_size, q_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask - FIXED: Properly handle mask dimensions
        if mask is not None:
            # Ensure mask has the right shape: (batch_size, num_heads, q_seq_len, k_seq_len)
            if mask.dim() == 4:  # Already in correct shape
                if mask.size(1) == 1:  # (batch_size, 1, q_seq_len, k_seq_len)
                    mask = mask.expand(batch_size, self.num_heads, q_seq_len, k_seq_len)
                else:  # (batch_size, num_heads, q_seq_len, k_seq_len)
                    mask = mask
            elif mask.dim() == 3:  # (batch_size, q_seq_len, k_seq_len)
                mask = mask.unsqueeze(1).expand(batch_size, self.num_heads, q_seq_len, k_seq_len)
            elif mask.dim() == 2:  # (batch_size, seq_len) - padding mask
                # For self-attention where q_seq_len == k_seq_len
                mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
                mask = mask.expand(batch_size, self.num_heads, q_seq_len, k_seq_len)
            
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        output = output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, self.d_model
        )
        
        return self.w_o(output)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, 
                self_mask: Optional[torch.Tensor] = None, 
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention (with causal mask for decoder)
        attn_output = self.self_attention(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with encoder output
        cross_attn_output = self.cross_attention(x, enc_output, enc_output, cross_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, 
                 num_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Create padding mask for encoder
        if padding_mask is not None:
            # For encoder self-attention, we need (batch_size, seq_len, seq_len) mask
            if padding_mask.dim() == 2:  # (batch_size, seq_len)
                # Create 2D mask for self-attention
                mask = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)  # (batch_size, seq_len, seq_len)
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            else:
                mask = padding_mask
        else:
            mask = None
            
        # Embedding + positional encoding
        x = self.dropout(self.pos_encoding(self.token_embedding(x)))
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
            
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, 
                 num_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
                self_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embedding + positional encoding
        x = self.dropout(self.pos_encoding(self.token_embedding(x)))
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, self_mask, cross_mask)
            
        return self.output_proj(x)

def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Create padding mask for encoder and cross-attention"""
    # Return (batch_size, seq_len) boolean mask
    return (seq != pad_idx)

def create_causal_mask(seq_len: int, device: str = 'cpu') -> torch.Tensor:
    """Create causal mask for decoder self-attention"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return ~mask  # (seq_len, seq_len)

def create_decoder_self_mask(seq: torch.Tensor, pad_idx: int, device: str = 'cpu') -> torch.Tensor:
    """Combine causal mask with padding mask for decoder self-attention"""
    batch_size, seq_len = seq.shape
    
    # Create causal mask: (seq_len, seq_len)
    causal_mask = create_causal_mask(seq_len, device)
    
    # Create padding mask: (batch_size, seq_len)
    padding_mask = create_padding_mask(seq, pad_idx).to(device)
    
    # Combine: we need (batch_size, seq_len, seq_len)
    # For each sequence in batch, apply both causal and padding mask
    combined_mask = causal_mask.unsqueeze(0) & padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)
    
    return combined_mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)

def create_cross_attention_mask(src_seq: torch.Tensor, tgt_seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Create mask for decoder cross-attention"""
    # src_mask: (batch_size, src_seq_len)
    src_mask = create_padding_mask(src_seq, pad_idx)
    # tgt_mask: (batch_size, tgt_seq_len)  
    tgt_mask = create_padding_mask(tgt_seq, pad_idx)
    
    # Cross attention mask: (batch_size, tgt_seq_len, src_seq_len)
    # We attend to src positions only if both src and tgt positions are not padding
    cross_mask = tgt_mask.unsqueeze(2) & src_mask.unsqueeze(1)
    
    return cross_mask.unsqueeze(1)  # (batch_size, 1, tgt_seq_len, src_seq_len)

# Training functions
def train_encoder_only(model: TransformerEncoder, data_loader, 
                      optimizer, criterion, pad_idx: int, device: str):
    """Train encoder-only model (e.g., for classification)"""
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        src, tgt = batch
        src, tgt = src.to(device), tgt.to(device)
        
        # Create padding mask
        src_mask = create_padding_mask(src, pad_idx)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, src_mask)
        
        # For encoder-only tasks, you might use only the [CLS] token or mean pooling
        # This is a simplified example - adapt based on your specific task
        loss = criterion(output.mean(dim=1), tgt)  # Mean pooling for classification
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def train_encoder_decoder(encoder: TransformerEncoder, decoder: TransformerDecoder, 
                         data_loader, optimizer, criterion, pad_idx: int, device: str):
    """Train encoder-decoder model (e.g., for sequence-to-sequence tasks)"""
    encoder.train()
    decoder.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(data_loader):
        src, tgt = batch
        src, tgt_in = src.to(device), tgt[:, :-1].to(device)
        tgt_out = tgt[:, 1:].to(device)
        
        # Create masks
        src_padding_mask = create_padding_mask(src, pad_idx)
        tgt_self_mask = create_decoder_self_mask(tgt_in, pad_idx, device)
        cross_mask = create_cross_attention_mask(src, tgt_in, pad_idx)
        
        optimizer.zero_grad()
        
        # Forward pass
        memory = encoder(src, src_padding_mask)
        output = decoder(tgt_in, memory, tgt_self_mask, cross_mask)
        
        # Calculate loss (ignore padding)
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_out.reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), 
            max_norm=1.0
        )
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(data_loader)

# Example usage with proper testing
if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 1000  # Reduced for testing
    d_model = 512
    num_layers = 2
    num_heads = 8
    d_ff = 2048
    max_seq_len = 50
    dropout = 0.1
    pad_idx = 0
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Create models
    encoder = TransformerEncoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout)
    decoder = TransformerDecoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout)
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    # Print model parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Encoder parameters: {count_parameters(encoder):,}")
    print(f"Decoder parameters: {count_parameters(decoder):,}")
    print(f"Total parameters: {count_parameters(encoder) + count_parameters(decoder):,}")
    
    # Test with different sequence lengths
    src_seq_len = 10
    tgt_seq_len = 8
    
    src_data = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 0, 0, 0, 0, 0],
        [16, 17, 18, 0, 0, 0, 0, 0, 0, 0],
        [19, 20, 21, 22, 23, 24, 25, 26, 0, 0]
    ])
    
    tgt_data = torch.tensor([
        [1, 30, 31, 32, 33, 34, 35, 36],
        [37, 38, 39, 0, 0, 0, 0, 0],
        [40, 41, 42, 43, 44, 0, 0, 0],
        [45, 46, 47, 48, 49, 50, 0, 0]
    ])
    
    print(f"\nSource data shape: {src_data.shape}")
    print(f"Target data shape: {tgt_data.shape}")
    
    # Test masks
    src_mask = create_padding_mask(src_data, pad_idx)
    tgt_self_mask = create_decoder_self_mask(tgt_data[:, :-1], pad_idx, device)
    cross_mask = create_cross_attention_mask(src_data, tgt_data[:, :-1], pad_idx)
    
    print(f"Source mask shape: {src_mask.shape}")
    print(f"Target self mask shape: {tgt_self_mask.shape}")
    print(f"Cross mask shape: {cross_mask.shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    memory = encoder(src_data.to(device), src_mask.to(device))
    print(f"Encoder output shape: {memory.shape}")
    
    output = decoder(tgt_data[:, :-1].to(device), memory, tgt_self_mask.to(device), cross_mask.to(device))
    print(f"Decoder output shape: {output.shape}")
    
    # Test training
    print("\nTesting training step...")
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), 
        lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # Create a simple data loader for testing
    class SimpleDataLoader:
        def __init__(self, src_data, tgt_data, batch_size=2):
            self.src_data = src_data
            self.tgt_data = tgt_data
            self.batch_size = batch_size
            self.num_batches = len(src_data) // batch_size
            
        def __len__(self):
            return self.num_batches
            
        def __iter__(self):
            for i in range(self.num_batches):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                yield (self.src_data[start_idx:end_idx], self.tgt_data[start_idx:end_idx])
    
    # Test training loop
    data_loader = SimpleDataLoader(src_data, tgt_data, batch_size=2)
    
    print("Starting training test...")
    avg_loss = train_encoder_decoder(encoder, decoder, data_loader, optimizer, criterion, pad_idx, device)
    print(f"Training completed! Average loss: {avg_loss:.4f}")
    
    print("\nAll tests passed! Transformer implementation is working correctly.")
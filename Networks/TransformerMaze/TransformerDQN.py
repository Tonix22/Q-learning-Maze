from Networks.TransformerMaze.PositionalEncoding import PositionalEncoding
import torch.nn as nn

class TransformerDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead, num_layers, seq_length, num_actions):
        """
        Args:
            input_dim: Dimension of the state representation (e.g., if one-hot encoded, this is num_states)
            hidden_dim: Dimension of the transformer embeddings.
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            seq_length: Length of the input state sequence.
            num_actions: Number of possible actions.
        """
        super().__init__()
        self.seq_length = seq_length
        
        # Map the raw state input to a hidden dimension.
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout=0.1, max_len=seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer to produce Q-values
        self.fc_out = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_length, input_dim]
        Returns:
            Q-values of shape [batch_size, num_actions]
        """
        # Embed input states
        x = self.input_embedding(x)  # shape: [batch_size, seq_length, hidden_dim]
        x = self.positional_encoding(x)  # add positional info
        
        # The transformer expects input of shape [seq_length, batch_size, hidden_dim]
        x = x.transpose(0, 1)
        transformer_output = self.transformer_encoder(x)  # shape: [seq_length, batch_size, hidden_dim]
        
        # Use the output corresponding to the last element in the sequence (or pool the sequence)
        last_output = transformer_output[-1]  # shape: [batch_size, hidden_dim]
        
        # Map to Q-values for each action
        q_values = self.fc_out(last_output)  # shape: [batch_size, num_actions]
        return q_values

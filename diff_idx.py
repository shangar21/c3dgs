import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableIndexing(nn.Module):
    def __init__(self, input_size, codebook_size, hidden_size=64, temperature=1.0):
        """
        Initializes the DifferentiableIndexing module.
        
        Args:
            input_size (int): The size of the input feature vector.
            codebook_size (int): The number of possible indices (codebook size).
            hidden_size (int): The size of the hidden layer in the MLP. Default is 64.
            temperature (float): The temperature parameter for the Gumbel-Softmax. Default is 1.0.
        """
        super(DifferentiableIndexing, self).__init__()
        self.temperature = temperature
        
        # Define the MLP to map input features to logits
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, codebook_size)
        )

    def forward(self, features):
        """
        Forward pass of the DifferentiableIndexing module.
        
        Args:
            features (Tensor): Input feature tensor of shape (batch_size, input_size).
        
        Returns:
            Tensor: Output tensor representing differentiable indices of shape (batch_size, codebook_size).
            Tensor: Output tensor representing the discrete argmax index for each input.
        """
        # Compute logits from input features using the MLP
        logits = self.mlp(features)
        
        # Apply Gumbel-Softmax to make the index selection differentiable
        gumbel_softmax_output = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        
        # Compute the discrete argmax index for each input in the batch
        argmax_indices = torch.argmax(gumbel_softmax_output, dim=-1)
        
        return gumbel_softmax_output, argmax_indices

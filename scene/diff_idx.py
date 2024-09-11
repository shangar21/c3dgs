import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableIndexing(nn.Module):
    def __init__(self, num_gaussians, codebook_size, hidden_size=64, temperature=1.0):
        """
        Initializes the DifferentiableIndexing module.

        Args:
            num_gaussians (int): Number of possible input indices (0 to N).
            codebook_size (int): Number of possible codebook indices (0 to K).
            hidden_size (int): Size of the hidden layer in the MLP.
            temperature (float): Temperature for the Gumbel-Softmax.
        """
        super(DifferentiableIndexing, self).__init__()
        self.temperature = temperature
        self.codebook_size = codebook_size
        
        # Embedding layer to map Gaussian indices to hidden vectors
        self.embedding = nn.Embedding(num_gaussians, hidden_size)
        
        # Fully connected layer to output logits for the codebook indices
        #self.fc = nn.Sequential(
        #    nn.Linear(hidden_size, hidden_size * 2),
        #    nn.ReLU(),  # Activation function
        #    nn.Linear(hidden_size * 2, hidden_size * 2),
        #    nn.ReLU(),  # Activation function
        #    nn.Linear(hidden_size * 2, codebook_size)  # Final layer to match codebook size
        #)
        self.fc = nn.Linear(hidden_size, codebook_size)

    def _process_chunk(self, gaussian_indices, use_topk, top_k):
        """
        Processes a chunk of Gaussian indices.

        Args:
            gaussian_indices (Tensor): Input tensor of Gaussian indices.
            use_topk (bool): Whether to apply top-k selection.
            top_k (int): Number of top elements to select if use_topk is True.

        Returns:
            Tensor: Output tensor of codebook indices for the chunk.
        """
        # Embed the Gaussian indices to hidden vectors
        embedded = self.embedding(gaussian_indices)  # Shape: (chunk_size, hidden_size)

        # Pass through a linear layer to get logits for the codebook indices
        logits = self.fc(embedded)  # Shape: (chunk_size, codebook_size)

        if use_topk:
            # Select the top-k logits along the codebook dimension
            top_logits, _ = torch.topk(logits, top_k, dim=-1)
            # Apply Gumbel-Softmax on the top-k logits
            gumbel_softmax_out = F.gumbel_softmax(top_logits, tau=self.temperature, hard=True, dim=-1)
            # Map back the indices to the original codebook range using argmax
            codebook_indices = gumbel_softmax_out.argmax(dim=-1)
        else:
            # Apply Gumbel-Softmax on the full logits
            gumbel_softmax_out = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
            codebook_indices = gumbel_softmax_out.argmax(dim=-1)  # Shape: (chunk_size,)

        return logits, codebook_indices 

    def forward(self, gaussian_indices, use_chunking=False, chunk_size=1000, use_topk=False, top_k=100):
        """
        Forward pass of the DifferentiableIndexing module.

        Args:
            gaussian_indices (Tensor): Input tensor of Gaussian indices of shape (batch_size,).
            use_chunking (bool): Whether to use chunking to reduce memory usage.
            chunk_size (int): Size of each chunk when chunking is enabled.
            use_topk (bool): Whether to apply top-k selection on logits before Gumbel-Softmax.
            top_k (int): Number of top elements to select if use_topk is True.

        Returns:
            Tensor: Output tensor of codebook indices of shape (batch_size,).
        """
        if use_chunking:
            outputs = []
            logits = []
            # Process input in chunks
            for i in range(0, len(gaussian_indices), chunk_size):
                chunk = gaussian_indices[i:i+chunk_size]
                lgts, codebook_indices = self._process_chunk(chunk, use_topk, top_k)
                outputs.append(codebook_indices)
                logits.append(lgts)
            return torch.cat(logits, dim=0), torch.cat(outputs, dim=0)  # Concatenate chunks along batch dimension
        else:
            # Process all at once
            return self._process_chunk(gaussian_indices, use_topk, top_k)


if __name__ == "__main__":
    from tqdm import tqdm
    import json
    import matplotlib.pyplot as plt
    N = 300_000
    k = N // 10
    indices = torch.arange(N).cuda()
    gt = torch.randint(0, k, indices.shape).cuda()
    model = DifferentiableIndexing(N, k, hidden_size = 64).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()

    n_epochs = 1000

    progress_bar = tqdm(range(n_epochs))

    losses = []

    for epoch in progress_bar:
        model.train()
        logits, out = model(indices, use_chunking=True, use_topk=True)
        l = loss(logits, gt)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        progress_bar.set_postfix({"Loss": l.item()})
        losses.append(l.item())

    plt.plot(list(range(len(losses))), losses)
    plt.savefig('plot.png')

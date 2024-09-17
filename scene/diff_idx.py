import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pdb

class DifferentiableIndexing(nn.Module):
    def __init__(self, num_gaussians, codebook_size, hidden_size=64, temperature=10.0):
        """
        Initializes the DifferentiableIndexing module.

        Args:
            num_gaussians (int): Number of possible input indices (0 to N).
            codebook_size (int): Number of possible codebook indices (0 to K).
            hidden_size (int): Size of the hidden layer in the MLP.
            temperature (float): Initial temperature for the Gumbel-Softmax.
        """
        super(DifferentiableIndexing, self).__init__()
        self.temperature = temperature
        self.codebook_size = codebook_size
        
        # Embedding layer to map Gaussian indices to hidden vectors
        self.embedding = nn.Embedding(num_gaussians, hidden_size)
        
        # Layer normalization after embedding
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Fully connected layers for outputting logits for the codebook indices
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(), 
            nn.Linear(hidden_size * 2, codebook_size)  # Final layer to match codebook size
        )
        # Apply Xavier initialization to the linear layers
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


    def forward(self, gaussian_indices, anneal_factor=None):
        """
        Forward pass of the DifferentiableIndexing module.

        Args:
            gaussian_indices (Tensor): Input tensor of Gaussian indices of shape (batch_size,).
            anneal_factor (float, optional): Factor to adjust the temperature during annealing.

        Returns:
            logits (Tensor): Output logits of shape (batch_size, codebook_size).
            codebook_indices (Tensor): Output tensor of codebook indices of shape (batch_size,).
        """
        # Embed the Gaussian indices to hidden vectors
        embedded = self.embedding(gaussian_indices)  # Shape: (batch_size, hidden_size)
        embedded = self.layer_norm(embedded)  # Normalize embeddings
        
        # Pass through the fully connected layers to get logits for the codebook indices
        logits = self.fc(embedded)  # Shape: (batch_size, codebook_size)

        # Apply Gumbel-Softmax for differentiable indexing
        if anneal_factor is not None:
            self.temperature *= anneal_factor  # Update temperature with annealing factor

        # Gumbel-Softmax trick for differentiable sampling
        codebook_indices = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)

        return logits, codebook_indices.argmax(dim=-1)  # Return the indices for hard assignment

class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, indices, gt_indices):
        self.indices = indices.cuda()
        self.gt_indices = gt_indices.cuda()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx], self.gt_indices[idx]

class IndexDatasetEval(torch.utils.data.Dataset):
    def __init__(self, indices):
        self.indices = indices.cuda()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx]

def model_inference(model, indices, batch_size=1000):
    indices = indices.cuda()
    dataset = IndexDatasetEval(indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    res = []
    for index in loader:
        logits, out = model(index)
        res.append(out)
    return torch.cat(res, dim=0)

def initial_train(model, gt_indices, batch_size=1000, n_epochs=50):
    indices = torch.arange(gt_indices.shape[0]).cuda()
    gt = gt_indices.cuda()
    
    dataset = IndexDataset(indices, gt)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    progress_bar = tqdm(range(n_epochs))

    count = 0
    for epoch in progress_bar:
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for index, gt_index in train_loader:
            logits, _ = model(index)
            loss = loss_fn(logits, gt_index)
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)  # Get the index of the max log-probability
            correct_predictions += (predicted == gt_index).sum().item()
            total_predictions += gt_index.size(0)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate epoch loss and accuracy
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_predictions

        model_rest = model_inference(model, indices)
        eval_accuracy = (model_rest == gt).sum().item() / len(gt)

        # Update progress bar with loss and accuracy
        progress_bar.set_postfix({"Loss": epoch_loss, "Accuracy": epoch_accuracy * 100, "Eval Accuracy": eval_accuracy})

    print(f"Training completed. Final Loss: {epoch_loss:.4f}, Final Accuracy: {epoch_accuracy * 100:.2f}%")

if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    N = 300_000
    k = N // 10

    indices = torch.arange(N).cuda()
    gt = torch.randint(0, k, indices.shape).cuda()
    model = DifferentiableIndexing(N, k, hidden_size = 64).cuda()

    initial_train(model, gt)

   # model_inference(model, indices)

   # dataset = IndexDataset(indices, gt)
   # train_loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True)
   # 
   # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   # loss = nn.CrossEntropyLoss()

   # n_epochs = 40

   # progress_bar = tqdm(range(n_epochs))

   # losses = []

   # for epoch in progress_bar:
   #     model.train()
   #     for index, gt_index in train_loader:
   #         logits, out = model(index)
   #         l = loss(logits, gt_index)
   #         optimizer.zero_grad()
   #         l.backward()
   #         optimizer.step()
   #         progress_bar.set_postfix({"Loss": l.item()})
   #         losses.append(l.item())

   # plt.plot(list(range(len(losses))), losses)
   # plt.savefig('plot.png')

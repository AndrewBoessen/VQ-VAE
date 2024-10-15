import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataloader, randon_split
import torch.optim as optim

class VectorQuantizeEMA(nn.Module):
    '''
    Exponential Moving Average (EMA) vector quantization for VQ-VAE model
    '''

    def __init__(self, n_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizeEMA, self).__init__()

        self._embedding_dim = embedding_dim # Dimension of an embedding vector, D
        self._n_embeddings = n_embeddings # Number of categories in distribution, K
        
        # Parameters
        self._embedding = nn.Embedding(self._n_embeddings, self._embedding_dim) # Embedding table for categorical distribution
        self._embedding.weight.data.normal_() # Randomly initialize embeddings
        self.register_buffer('_ema_cluster_size', torch.zeros(n_embeddings)) # Clusters for EMA
        self._ema_w = nn.Parameters(torch.Tensor(n_embeddings, self._embedding_dim)) # EMA weights
        self._ema_w.data.normal()

        # Loss / Training Parameters
        self._commitment_cost = commitment_cost
        self._decay = decay
        self.epsilon = epsilon

    def formward(self, z_e):
        # reshape from BCHW -> BHWC
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        shape = z_e.shape

        # Flatten input embeddings
        flat_z_e = z_e.view(-1, self._embedding_dim)

        # Claculate Distances
        # ||z_e||^2 + ||e||^2 - 2 * z_q
        distances = (torch.sum(flat_z_e**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2,)
                     - 2 * torch.matmul(flat_z_e, self._embedding.weight.t())
                    )
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._n_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1) # Convert to shape of embeddings

        # Quantize and Unflatten
        z_q = torch.matmul(encodings, self._embedding.weight).view(shape)

        # Update weights with EMA
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon) / (n + self._n_embeddings * self._epsilon) * n
            )

            dq = torch.matmul(encodings.t(), flat_z_e)
            self._ema_w = nn.Parameters(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(z_q.detach(), z_e) # distance from encoder output and quantized embeddings
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Loss
        z_q = z_e + (z_q - z_e).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Convert shape back to BCHW
        return loss, z_q.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

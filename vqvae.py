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

    def forward(self, z_e):
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
        encodings = torch.zeros(encoding_indices.shape[0], self._n_embeddings, device=z_e.device)
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

class Residual(nn.Module):
    '''
    Residual Convolutional Layer
    '''
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            # C: 3 -> residual hidden
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            # C: resiual hidden -> out_hidden
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x) # residual output


class ResidualStack(nn.Module):
    '''
    Residual Convolution Stack
    '''
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        # Apply all residual layers in stack
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Encoder(nn.Module):
    '''
    Convolutional Encoder
    '''
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._conv2 = nn.Conv2d(in_channels=num_hiddens//2, out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._conv3 = nn.Conv2d(in_channels=num_hiddens, out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = F.relu(x)
        x = self._conv2(x)
        x = F.relu(x)
        x = self._conv3(x)
        return self._residual_stack(x)

class Decoder(nn.Module):
    '''
    Convolutional Decoder
    '''
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self._conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens,
                                kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4, stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=3,
                                                kernel_size=4, stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        return self._conv_trans_2(x)

class VQVAE(nn.Module):
    '''
    VQ-VAE Network
    Contains: Encoder, Quantizer, Decoder
    '''
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay):
        super(VQVAE, self).__init__()

        self._encoder = Encoder(3, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim,
                                      kernel_size=1, stride=1)

        self._vq = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x) # encode image to latent
        z = self._pre_vq_conv(z)
        loss, z_q, perplexity, _ = self._vq(z) # quantize encoding to dicrete space
        x_recon = self._decoder(z_q) # reconstruction of input from decoder
        return loss, x_recon, perplexity

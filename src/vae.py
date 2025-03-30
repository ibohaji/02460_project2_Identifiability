import torch 
import torch.nn as nn 
import torch.distributions as td 
from configs import TrainingConfig




M = TrainingConfig.latent_dim 


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
            The encoder network that takes a tensor of dimension
            `(batch_size, feature_dim1, feature_dim2)` as input
            and outputs a tensor of dimension `(batch_size, 2M)`,
            where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of input data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
            A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`.

        Returns:
        A Gaussian distribution with computed mean and standard deviation.
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)

        # Example:
        # z = torch.randn(4, 10)  # Assume z is a tensor of shape [batch_size=4, 10]
        # a, b = torch.chunk(z, 2, dim=-1)
        # a and b will have shape [4, 5], as the tensor is split into two parts along the last dimension.

class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Gaussian decoder distribution based on a given decoder network.

        Parameters:
        decoder_net: [torch.nn.Module]
            The decoder network that takes a tensor of dimension `(batch_size, M)`
            as input, where M is the dimension of the latent space, and outputs a
            tensor of dimension `(batch_size, feature_dim1, feature_dim2)`.
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True)
        # In case you want to learn the standard deviation of the Gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Gaussian distribution over the data space.

        Parameters:
        z: [torch.Tensor]
            A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.

        Returns:
        A Gaussian distribution with computed mean and a fixed standard deviation.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3) #note the variance of decoder is fixed
        # This defines a 784-dimensional independent normal distribution, where each dimension is independent.


class VAE(nn.Module):
    def __init__(self, prior, decoders, encoder):
        """
        Variational Autoencoder (VAE) with multiple decoders.

        Parameters:
        prior: [torch.nn.Module]
            The prior distribution over the latent space.
        decoders: [list of torch.nn.Module]
            A list containing multiple decoders.
        encoder: [torch.nn.Module]
            The encoder network that maps input data to a latent distribution.
        """
        super(VAE, self).__init__()
        self.prior = prior
        self.decoders = nn.ModuleList(decoders)  # Use ModuleList to allow PyTorch to properly track parameters
        self.encoder = encoder

    def elbo(self, x, decoder_idx):
        """
        Compute the Evidence Lower Bound (ELBO) for a given input and selected decoder.

        Parameters:
        x: [torch.Tensor]
            The input data tensor.
        decoder_idx: [int]
            The index of the decoder to be used.

        Returns:
        The computed ELBO value.
        """
        q = self.encoder(x)  # Encode input into a latent distribution
        z = q.rsample()  # Sample from the latent distribution using the reparameterization trick
        decoder = self.decoders[decoder_idx]  # Select the corresponding decoder

        elbo = torch.mean(
            decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )  # Compute ELBO using the likelihood, posterior, and prior

        return elbo

    def sample(self, decoder_idx, n_samples=1):
        """
        Generate samples from the specified decoder.

        Parameters:
        decoder_idx: [int]
            The index of the decoder to be used.
        n_samples: [int, default=1]
            The number of samples to generate.

        Returns:
        A batch of generated samples.
        """
        z = self.prior().sample(torch.Size([n_samples]))  # Sample from the prior distribution
        decoder = self.decoders[decoder_idx]  # Select the corresponding decoder
        return decoder(z).sample()  # Generate samples from the decoder

    def forward(self, x, decoder_idx):
        """
        Compute the negative ELBO for optimization.

        Parameters:
        x: [torch.Tensor]
            The input data tensor.
        decoder_idx: [int]
            The index of the decoder to be used.

        Returns:
        The negative ELBO value.
        """
        return -self.elbo(x, decoder_idx)



def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net


def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net


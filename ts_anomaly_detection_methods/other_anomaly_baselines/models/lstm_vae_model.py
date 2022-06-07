import torch
from torch import nn
from torch.nn import functional as F


class LSTM_Encoder(nn.Module):
    '''
    Encodes the input by passing through the encoder network and returns the latent representations.
    '''

    def __init__(self, device, in_channel, hidden_size=16, hidden_dim=3):
        super(LSTM_Encoder, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTM(input_size=in_channel, hidden_size=hidden_size, batch_first=True, bidirectional=True)

        self.fc_mu = nn.Linear(2*hidden_size, hidden_dim)
        self.fc_var = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_dim),
            nn.Softplus()
            )

    def forward(self, inputs):
        '''
        Args: 
            inputs: [batch_size, max_length, in_channel]
        Returns:
            z_mu: [batch_size, max_length, hidden_dim]
            z_log_var: [batch_size, max_length, hidden_dim]
        '''
        batch_size, _, _ = inputs.size()
        h_0 = torch.zeros((2, batch_size, self.hidden_size), requires_grad=True).to(self.device)
        c_0 = torch.zeros((2, batch_size, self.hidden_size), requires_grad=True).to(self.device)

        # hidden_res: [batch_size, max_length, 2*hidden_size]
        hidden_res, (h_n, c_n) = self.encoder(inputs, (h_0, c_0))
        z_mu = self.fc_mu(hidden_res) # [batch_size, max_length, hidden_dim]
        z_log_var = self.fc_var(hidden_res) + 1e-4 # [batch_size, max_length, hidden_dim]

        return z_mu, z_log_var


class LSTM_Decoder(nn.Module):
    '''
    Maps the given latent representations through the decoder network onto the inputs space.
    '''

    def __init__(self, device, in_channel, hidden_size=16, hidden_dim=3):
        super(LSTM_Decoder, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_dim

        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)

        self.fc_mu = nn.Linear(2*hidden_size, in_channel)
        self.fc_var = nn.Sequential(
            nn.Linear(2*hidden_size, in_channel),
            nn.Softplus()
            )

    def forward(self, z):
        '''
        Args: 
            z: [batch_size, max_length, hidden_dim]
        Returns:
            x_mu: [batch_size, max_length, in_channel]
            x_log_var: [batch_size, max_length, in_channel]
        '''
        batch_size, _, _ = z.size()
        h_0 = torch.zeros((2, batch_size, self.hidden_size), requires_grad=True).to(self.device)
        c_0 = torch.zeros((2, batch_size, self.hidden_size), requires_grad=True).to(self.device)

        # hidden_res: [batch_size, max_length, 2*hidden_size]
        hidden_res, (h_n, c_n) = self.decoder(z, (h_0, c_0))
        x_mu = self.fc_mu(hidden_res) # [batch_size, max_length, in_channel]
        x_log_var = self.fc_var(hidden_res) + 1e-4 # [batch_size, max_length, in_channel]

        return x_mu, x_log_var


class LSTM_VAE_Model(nn.Module):

    def __init__(self, device, in_channel, hidden_size=16, hidden_dim=3):
        super(LSTM_VAE_Model, self).__init__()

        self.device = device
        self.in_channel = in_channel
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_dim

        self.Encoder = LSTM_Encoder(self.device, self.in_channel, self.hidden_size, self.hidden_dim)
        self.Decoder = LSTM_Decoder(self.device, self.in_channel, self.hidden_size, self.hidden_dim)


    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [batch_size, max_length, hidden_dim]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [batch_size, max_length, hidden_dim]
        :return: (Tensor) [batch_size, max_length, hidden_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, inputs):
        '''
        Args:
            inputs: [batch_size, max_length, in_channel]
        Returns:
            outputs: [batch_size, max_length, in_channel]
            z_mu, z_log_var: [batch_size, max_length, hidden_dim]
            x_mu, x_log_var: [batch_size, max_length, in_channel]
        '''
        z_mu, z_log_var = self.Encoder(inputs) 
        z = self.reparameterize(z_mu, z_log_var) # [batch_size, max_length, hidden_dim]

        x_mu, x_log_var = self.Decoder(z)
        outputs = self.reparameterize(x_mu, x_log_var) # [batch_size, max_length, in_channel]

        return  outputs, z_mu, z_log_var, x_mu, x_log_var

    def loss_function(self, inputs, outputs, z_mu, z_log_var, x_mu, x_log_var, z_kld_weight, x_kld_weight):
        """
        Computes the VAE loss function.
        KL(N(/mu, /sigma), N(0, 1)) = /log /frac{1}{/sigma} + /frac{/sigma^2 + /mu^2}{2} - /frac{1}{2}
        Args:
            inputs, outputs: [batch_size, max_length, in_channel]
            z_mu, z_log_var: [batch_size, max_length, hidden_dim]
            x_mu, x_log_var: [batch_size, max_length, in_channel]
            z_kld_weight, x_kld_weight: float Value
        """
        recons_loss = F.mse_loss(outputs, inputs)

        _, _, hidden_dim = z_mu.size()
        z_mu = z_mu.reshape(-1, hidden_dim)
        z_log_var = z_log_var.reshape(-1, hidden_dim)
        z_kld_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mu ** 2 - z_log_var.exp(), dim = 1), dim = 0)

        _, _, in_channel = x_mu.size()
        x_mu = x_mu.reshape(-1, in_channel)
        x_log_var = x_log_var.reshape(-1, in_channel)
        x_kld_loss = torch.mean(-0.5 * torch.sum(1 + x_log_var - x_mu ** 2 - x_log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + z_kld_weight * z_kld_loss + x_kld_weight * x_kld_loss

        return loss
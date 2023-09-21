import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np

import architecture_library as nn_lib

NOISE_DIM = 96


class Generator(nn.Module):
    #bsc format
    def __init__(self, n_out_timesteps, n_out_channels, n_time_label_channels,
                 n_in_channels = NOISE_DIM, name="generator"):
        super(Generator, self).__init__()
        self.n_in_channels = n_in_channels
        self.hidden_size = n_in_channels
        self.n_out_channels = n_out_channels
        self.n_time_label_channels = n_time_label_channels
        self.n_out_timesteps = n_out_timesteps
        self.name = name

        self.dropout_rate = .1

        self.time_channel_embedding = nn.Sequential(
            nn.Linear(self.n_time_label_channels, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=False),
        )

        self.time_embedding2 = nn.Sequential(
            llib.PositionalEncoding(self.hidden_size),
        )

        self.attention = nn.MultiheadAttention(self.hidden_size, num_heads=8)
        self.post_attention = nn.Sequential(
            nn.LeakyReLU(inplace=False),
        )

        self.cnns = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            llib.CausalConv1d(self.hidden_size, self.hidden_size, kernel_size=8),
            nn.BatchNorm1d(self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            llib.Transpose(1, 2),
            llib.CausalConv1d(self.hidden_size, self.hidden_size, kernel_size=8),
            nn.LeakyReLU(inplace=True),
        )

        self.output_layers = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            llib.CausalConv1d(self.hidden_size, self.n_out_timesteps, kernel_size=8),
            nn.LayerNorm(self.hidden_size),
            nn.LeakyReLU(inplace=True),
            llib.Transpose(1, 2),
            llib.CausalConv1d(self.hidden_size, self.n_out_channels, kernel_size=4),
            llib.Transpose(1, 2),
        )

        self.output_activation = nn.Sigmoid()
        self.to(DEVICE)


    def forward(self, noise, time_labels):
        #prepare time
        time_labels_out = self.time_channel_embedding(time_labels)
        time_labels_out = time_labels_out.unsqueeze(1).repeat(1, self.hidden_size, 1)
        time_labels_out = self.time_embedding2(time_labels_out)

        #prepare noise
        noise_out = noise.unsqueeze(1).repeat(1, self.hidden_size, 1)

        #combine
        out = noise_out + time_labels_out

        #attention
        attention = self.attention(out, out, out)[0]
        attention = self.post_attention(attention)

        #cnns
        cnn_out = self.cnns(out)

        #combine
        out = attention + cnn_out
        out = self.output_layers(out)
        out = self.output_activation(out)
        return out


class Critic(nn.Module):
    #bsc format
    def __init__(self, ntimesteps, nchannels, n_time_label_channels, hidden_size=96, name="critic"):
        super(Critic, self).__init__()

        self.ntimesteps = ntimesteps
        self.nchannels = nchannels
        self.hidden_size = hidden_size
        self.n_time_label_channels = n_time_label_channels
        self.dropout_rate = .1
        self.name = name

        self.time_labels_channel_embedding = nn.Sequential(
            nn.Linear(self.n_time_label_channels, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=False),
        )

        self.time_labels_embedding2 = nn.Sequential(
            llib.PositionalEncoding(self.hidden_size),
        )

        #to hidden_size * hidden_size
        self.input_embedding = nn.Sequential(
            nn.Linear(self.nchannels, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            llib.CausalConv1d(in_channels=self.ntimesteps, out_channels=self.hidden_size, kernel_size=8),
            nn.LeakyReLU(inplace=False),
        )

        self.post_time_x_combined = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
        )

        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=8, batch_first=True)
        self.post_attention = nn.Sequential(
            nn.LeakyReLU(inplace=False),
        )

        self.cnns = nn.Sequential(
            llib.CausalConv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=8),
            nn.LayerNorm(self.hidden_size), #todo: delete
            nn.LeakyReLU(inplace=False),
        )

        self.final_layers = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            llib.CausalConv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=8),
            nn.LayerNorm(self.hidden_size),
            nn.LeakyReLU(inplace=True),
            llib.CausalConv1d(in_channels=self.hidden_size, out_channels=1, kernel_size=8),
            nn.Linear(self.hidden_size, 1),
            nn.Flatten(),
        )
        self.to(DEVICE)

    def forward(self, x, time_labels):
        time_labels_out = time_labels.clone()
        out = x.clone()
        if self.training:
            time_labels_out = time_labels_out + torch.randn_like(time_labels_out, device=DEVICE) * 0.01
            out = out + torch.randn_like(x, device=DEVICE) * 0.025

        #embed time
        time_labels_out = self.time_labels_channel_embedding(time_labels_out)
        time_labels_out = time_labels_out.unsqueeze(1).repeat(1, self.hidden_size, 1)
        time_labels_out = self.time_labels_embedding2(time_labels_out)

        #embed input
        out = self.input_embedding(out)

        #combine
        out += time_labels_out
        out = self.post_time_x_combined(out)

        #attention
        attention = self.attention(out, out, out)[0]
        attention = self.post_attention(attention)

        #cnns
        cnns_out = self.cnns(out)

        #combine
        out = attention + cnns_out
        out = self.final_layers(out)
        return out



class WGAN:
    def __init__(self, generator, critic, name:str, critic_train_ratio=5, gp_lambda=10,
                 lr=0.0002, beta1=0.5, beta2=0.95, auto_tune_ratio=False, data_range=(0, 1)
                 ):
        self.device = DEVICE
        self.generator = generator.to(self.device)
        self.critic = critic.to(self.device)
        self.name = name
        self.initial_critic_train_ratio = critic_train_ratio
        self.critic_train_ratio = critic_train_ratio
        self.auto_tune_ratio = auto_tune_ratio
        self.gp_lambda = gp_lambda
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizer_C = optim.Adam(self.critic.parameters(), lr=lr, betas=(beta1, beta2))
        self.train_losses = []
        self.data_range = data_range


    def train(self):
        self.generator.train()
        self.critic.train()

    def eval(self):
        self.generator.eval()
        self.critic.eval()

    def gradient_penalty(self, real_data, fake_data, time_labels):
        batch_size = real_data.size(0)

        epsilon = torch.rand(batch_size, 1, 1, device=self.device)
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated = interpolated.requires_grad_(True)

        # epsilon = torch.rand(batch_size, 1, device=self.device)
        interpolated_labels = time_labels.clone()
        interpolated_labels = interpolated_labels.requires_grad_(True)

        interpolated_loss = self.critic(interpolated, interpolated_labels)

        # the interpolated_labels are given to the critic, but not used in the loss
        gradients = torch.autograd.grad(outputs=interpolated_loss, inputs=interpolated,
                         grad_outputs=torch.ones_like(interpolated_loss, requires_grad=False, device=self.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.reshape(batch_size, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda
        return gp

    def train_step(self, x, time_labels_in):
        #bsc format
        _, real_data, _ = x
        time_labels = time_labels_in.squeeze(1)
        batch_size, n_real_timesteps, n_real_channels = real_data.shape

        # Update Critic
        c_loss = None
        for _ in range(self.critic_train_ratio):
            noisy_real_data = real_data + torch.randn_like(real_data, device=self.device) * 0.05
            self.optimizer_C.zero_grad(set_to_none=True)

            # Generate fake data
            noise = torch.randn(batch_size, self.generator.n_in_channels, device=self.device)
            fake_data = self.generator(noise, time_labels)

            real_pred = self.critic(noisy_real_data, time_labels)
            fake_pred = self.critic(fake_data.detach(), time_labels)
            gp = self.gradient_penalty(real_data, fake_data.detach(), time_labels)

            c_loss = fake_pred.mean() - real_pred.mean() + gp
            c_loss.backward()
            self.optimizer_C.step()

        # Update Generator
        self.optimizer_G.zero_grad(set_to_none=True)
        noise = torch.randn(batch_size, self.generator.n_in_channels, device=self.device)
        fake_data = self.generator(noise, time_labels)
        fake_pred = self.critic(fake_data, time_labels)
        g_loss = -fake_pred.mean()
        g_loss.backward()
        self.optimizer_G.step()

        return c_loss.item(), g_loss.item()

    def adjust_critic_train_ratio(self):
        closs, gloss = self.train_losses[-1]
        if gloss > 10:
            self.critic_train_ratio -= 1
        if closs < -10:
            self.critic_train_ratio += 1

        self.critic_train_ratio = max(1, self.critic_train_ratio)
        self.critic_train_ratio = min(10, self.critic_train_ratio)
        _logger.info(f"Critic train ratio: {self.critic_train_ratio}")


    def update(self, data, epochs, steps_per_epoch, save_every=0, start_epoch=0):
        _logger.info(f"Training {self.name}")
        if start_epoch > 0:
            self.load(start_epoch)
        if start_epoch >= epochs:
            _logger.warning(f"Start epoch {start_epoch} is greater than or equal to epochs {epochs}, no training will be done")
            return

        start_time = time.time()
        critic_loss_summary = torch.zeros(1, device=self.device, dtype=torch.float32)
        gen_loss_summary = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.train()
        for epoch in range(start_epoch, epochs):
            critic_loss_summary.zero_()
            gen_loss_summary.zero_()
            for i, (x,y) in zip(range(0, steps_per_epoch), data):
                closs, gloss = self.train_step(x, y)
                critic_loss_summary += closs
                gen_loss_summary += gloss
            critic_loss_summary /= (i+1)
            gen_loss_summary /= (i+1)
            self.train_losses.append((critic_loss_summary.item(), gen_loss_summary.item()))
            _logger.info(f"Epoch {epoch+1}. Average critic loss: {critic_loss_summary.item():.8f}."
                         f" Average generator loss: {gen_loss_summary.item():.8f}")
            if not torch.isfinite(critic_loss_summary) or not torch.isfinite(gen_loss_summary):
                _logger.error(f"Loss is not finite. Stopping training on epoch {epoch}.")
                break
            if self.auto_tune_ratio:
                self.adjust_critic_train_ratio()
            if save_every > 0 and (epoch+1) % save_every == 0:
                self.save(epoch_num=epoch+1)


        self.save(epoch_num=epoch+1)
        minutes, seconds = divmod(time.time() - start_time, 60)
        _logger.info(f"Training {self.name} took {int(minutes)} minutes and {seconds:.2f} seconds.")
        return

    def generate_data(self, time_labels, as_numpy=False, batch_size=128):
        #not for training calls
        self.generator.eval()
        fake_datas = []
        with torch.no_grad():
            for i in range(0, time_labels.shape[0], batch_size):
                end = min(i+batch_size, time_labels.shape[0])
                current_batch_size = end - i
                noise = torch.randn(current_batch_size, self.generator.n_in_channels, device=self.device)
                fake_data = self.generator(noise, time_labels[i:end])
                fake_datas.append(fake_data)
        fake_data = torch.cat(fake_datas, dim=0)
        if as_numpy:
            fake_data = fake_data.detach().cpu().numpy()
        return fake_data
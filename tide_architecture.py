import torch
from torch import nn
from torch.nn import functional as F
import math

import architecture_library as nn_lib

class ResidualBlock(nn.Module):
    def __init__(self, layer_norm:bool, input_dim:int, hidden_dim:int, output_dim:int, dropout:float):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.dropout = dropout

        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None
        self.inner_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )
        self.residual_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        inner = self.inner_model(x)
        residual = self.residual_layer(x)
        x = residual + inner
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x

class ResidualStack(nn.Module):
    def __init__(self, layer_norm:bool, input_dim:int, hidden_dim:int, output_dim:int, dropout:float, n_layers:int):
        super().__init__()
        self.output_dim = output_dim
        self.dropout = dropout
        self.n_layers = n_layers
        if n_layers >=2:
            self.layers = nn.ModuleList(
                [ResidualBlock(layer_norm, input_dim, hidden_dim, hidden_dim, dropout)]
                + [ResidualBlock(layer_norm, hidden_dim, hidden_dim, hidden_dim, dropout) for _ in range(n_layers - 2)]
                + [ResidualBlock(layer_norm, hidden_dim, hidden_dim, output_dim, dropout)]
            )
        else:
            assert n_layers == 1
            self.layers = nn.ModuleList(
                [ResidualBlock(layer_norm, input_dim, hidden_dim, output_dim, dropout)]
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Tide(nn_lib.NnBase):
    def __init__(
        self,
        lookback_shape:tuple,
        static_shape:tuple,
        covariates_shape:tuple,
        target_shape:tuple,
        covariate_embedding_dim:int,
        encoder_layers:int,
        decoder_layers:int,
        encoder_dim:int,
        decoder_dim:int,
        temporal_decoder_dim:int,
        dropout:float,
        layer_norm:bool,
        device:torch.device,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.device = device

        self.covariate_updated_temporal_size = target_shape[0] + lookback_shape[0]
        concatted_dim = math.prod(lookback_shape) + math.prod(static_shape) + covariate_embedding_dim * self.covariate_updated_temporal_size
        self.encoder =  ResidualStack(
            layer_norm = layer_norm,
            input_dim=concatted_dim,
            hidden_dim=encoder_dim,
            output_dim=encoder_dim,
            dropout=dropout,
            n_layers=encoder_layers,
        )
        self.decoder = ResidualStack(
            layer_norm=layer_norm,
            input_dim=encoder_dim,
            hidden_dim=decoder_dim,
            output_dim=decoder_dim * target_shape[0],
            dropout=dropout,
            n_layers=decoder_layers,
        )

        # changes time dimension only so transposes needed
        self.global_residual_time = nn.Linear(lookback_shape[0], target_shape[0])
        self.global_residual_channel = nn.Linear(lookback_shape[1], target_shape[1])

        #time steps are independent of each other
        self.covariate_projection = ResidualBlock(
            layer_norm=layer_norm,
            input_dim=covariates_shape[-1],
            hidden_dim=covariate_embedding_dim,
            output_dim=covariate_embedding_dim,
            dropout=dropout,
        )

        #time steps are independent of each other
        self.temporal_decoder = ResidualBlock(
            layer_norm=layer_norm,
            input_dim=decoder_dim + covariate_embedding_dim,
            hidden_dim=temporal_decoder_dim,
            output_dim=target_shape[-1],
            dropout=dropout,
        )

        self.lookback_shape = lookback_shape
        self.static_shape = static_shape
        self.target_shape = target_shape
        self.to(device)


    def forward(self, x):
        static, history, covariates = x
        if covariates.ndim == 4: #only the most recent forecast
            covariates = covariates[:, -1]
        batch_size = static.shape[0]

        # temporal dimension compliance
        covariates_temporal_shape = self.covariate_updated_temporal_size
        if covariates.shape[1] < covariates_temporal_shape:
            #pad left
            pad = (0, 0, covariates_temporal_shape - covariates.shape[1], 0)
            covariates = F.pad(covariates, pad, "constant", 0)
        assert covariates.shape[1] == covariates_temporal_shape

        covariate_projection = self.covariate_projection(covariates)

        enc_input = torch.cat(
            (static.reshape(batch_size, -1),
             history.reshape(batch_size, -1),
             covariate_projection.reshape(batch_size, -1)),
            dim=-1
        )
        enc_out = self.encoder(enc_input)
        dec_out = self.decoder(enc_out)
        dec_out = dec_out.reshape(batch_size, self.target_shape[0], -1)


        temporal_decoder_input = torch.cat(
            (
                dec_out,
                covariate_projection[:, -self.target_shape[0]:]
            ), dim=-1
        )
        temporal_decoder_output = self.temporal_decoder(temporal_decoder_input)

        lookback_global_residual_time = self.global_residual_time(history.transpose(-1, -2)).transpose(-1, -2)
        lookback_global_residual_channel = self.global_residual_channel(lookback_global_residual_time)

        return temporal_decoder_output + lookback_global_residual_channel

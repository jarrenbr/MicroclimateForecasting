import torch
from torch import nn
import math

import architecture_library as nn_lib

class ForeseerDeterministic(meter_nn.MeterNnBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_layer_size = 128
        num_mha_heads = 4
        assert self.hidden_layer_size % 2 == 0, "Our bi-directional RNNs requires an even hidden layer size"
        assert self.hidden_layer_size % 8 == 0, "Our Conv1D requires a hidden layer size divisible by 8"
        assert self.hidden_layer_size % num_mha_heads == 0, "Our MultiheadAttention requires a hidden layer size divisible by the number of heads"
        self.dropout_rate = 0.1
        leaky_alpha=0.15

        self.history_embed = nn.Sequential(
            nn.Linear(self.history_input_shape[-1], self.hidden_layer_size),
            nn.LayerNorm(self.hidden_layer_size),
            nn.LeakyReLU(inplace=True, negative_slope=leaky_alpha),
            nn.Dropout(self.dropout_rate),
            nnlib.CausalConv1d(self.history_input_shape[-2], self.hidden_layer_size, kernel_size=6),
            nn.LeakyReLU(inplace=False, negative_slope=leaky_alpha),
        )

        self.history_peek = nn.MultiheadAttention(self.hidden_layer_size, num_mha_heads)
        self.history_peek2 = nn.Sequential(
            nn.LeakyReLU(inplace=False, negative_slope=leaky_alpha),
        )

        self.history_gru = nn.GRU(self.hidden_layer_size, self.hidden_layer_size//2,
             num_layers=2, batch_first=True, bidirectional=True, dropout=self.dropout_rate)
        self.history_after_gru = nn.Sequential(
            nn.BatchNorm1d(self.hidden_layer_size),
            nn.LeakyReLU(inplace=False, negative_slope=leaky_alpha),
        )

        self.forecasts_embed = nn.Sequential(
            nn.Linear(self.forecasts_input_shape[1], self.hidden_layer_size),
            nn.LayerNorm(self.hidden_layer_size),
            nn.LeakyReLU(inplace=True, negative_slope=leaky_alpha),
            nn.Dropout(self.dropout_rate),
            nnlib.CausalConv1d(self.forecasts_input_shape[0], self.hidden_layer_size, kernel_size=6),
            nn.LeakyReLU(inplace=False, negative_slope=leaky_alpha),
        )

        self.forecasts_gru = nn.GRU(self.hidden_layer_size, self.hidden_layer_size//2,num_layers=2,
                        batch_first=True, bidirectional=True, dropout=self.dropout_rate)

        self.forecasts_after_gru = nn.Sequential(
            nn.BatchNorm1d(self.hidden_layer_size),
            nn.LeakyReLU(inplace=False, negative_slope=leaky_alpha),
        )

        self.forecasts_peek = nn.MultiheadAttention(self.hidden_layer_size, num_mha_heads)
        self.forecasts_peek2 = nn.Sequential(
            nn.LeakyReLU(inplace=False, negative_slope=leaky_alpha),
        )

        self.static_in = nn.Sequential(
            nn.Linear(math.prod(self.static_input_shape), self.hidden_layer_size),
            nn.BatchNorm1d(self.hidden_layer_size),
            nn.LeakyReLU(inplace=True, negative_slope=leaky_alpha),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            nn.BatchNorm1d(self.hidden_layer_size),
            nn.LeakyReLU(inplace=True, negative_slope=leaky_alpha),
            nn.Dropout(self.dropout_rate),
        )

        self.static_bridge_to_forecast = nn.Sequential(
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size**2),
            nn.LeakyReLU(inplace=True, negative_slope=leaky_alpha),
            nnlib.Reshape((-1, self.hidden_layer_size, self.hidden_layer_size))
        )

        self.static_bridge_to_history = nn.Sequential(
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size**2),
            nn.LeakyReLU(inplace=True, negative_slope=leaky_alpha),
            nnlib.Reshape((-1, self.hidden_layer_size, self.hidden_layer_size)),
        )

        self.static_and_history_bridge_to_forecasts = nn.Sequential(
            nn.LayerNorm(self.hidden_layer_size),
            nnlib.CausalConv1d(self.hidden_layer_size, self.hidden_layer_size, kernel_size=10),
            nn.LeakyReLU(inplace=False, negative_slope=leaky_alpha),
        )

        self.static_and_forecasts_bridge_to_history = nn.Sequential(
            nn.LayerNorm(self.hidden_layer_size),
            nnlib.CausalConv1d(self.hidden_layer_size, self.hidden_layer_size, kernel_size=10),
            nn.LeakyReLU(inplace=False, negative_slope=leaky_alpha),
        )

        self.after_combine_all = nn.Sequential(
            nn.LayerNorm(self.hidden_layer_size),
            nnlib.CausalConv1d(self.hidden_layer_size, self.hidden_layer_size, kernel_size=4),
            nn.LeakyReLU(inplace=False, negative_slope=leaky_alpha),
        )

        self.after_all_peek = nn.Sequential(
            nn.LayerNorm(self.hidden_layer_size),
            nnlib.CausalConv1d(self.hidden_layer_size, self.hidden_layer_size, kernel_size=8),
            nn.LeakyReLU(inplace=False, negative_slope=leaky_alpha),
        )

        self.output_layers = nn.Sequential(
            nn.LayerNorm(self.hidden_layer_size),
            nnlib.Transpose(1, 2),
            nnlib.CausalConv1d(self.hidden_layer_size, self.output_shape[0], kernel_size=8),
            nn.LayerNorm(self.hidden_layer_size),
            nn.LeakyReLU(inplace=True, negative_slope=leaky_alpha),
            nnlib.Transpose(1, 2),
            nnlib.CausalConv1d(self.hidden_layer_size, self.output_shape[1], kernel_size=4),
            nnlib.Transpose(1, 2),
        )

        self.output_activation = nn.Sigmoid()

        return

    def forward(self, x):
        static, history, forecasts = x
        # forecasts has shape (batch_size, num_forecasts, forecast_length, num_features)
        # history has shape (batch_size, history_length, num_features)
        # static has shape (batch_size, num_features)

        # We've investigated using multiple forecasts. In this experiment, we only use the most recent.
        forecasts_out = forecasts[..., -1]


        #history initial embed
        history_embed = self.history_embed(history)

        history_peek, _ = self.history_peek(history_embed, history_embed, history_embed)
        history_peek = self.history_peek2(history_peek)

        history_out, _ = self.history_gru(history_embed)
        history_out = self.history_after_gru(history_out)

        #static initial embed
        static_out = self.static_in(static)

        #combine history and static
        static_for_history = self.static_bridge_to_history(static_out)
        history_and_static = history_out + static_for_history
        history_and_static_for_forecasts = self.static_and_history_bridge_to_forecasts(history_and_static)

        #forecasts initial embed
        forecasts_embed = self.forecasts_embed(forecasts_out)

        forecasts_peek, _ = self.forecasts_peek(forecasts_embed, forecasts_embed, forecasts_embed)
        forecasts_peek = self.forecasts_peek2(forecasts_peek)

        forecasts_out, _ = self.forecasts_gru(forecasts_embed)
        forecasts_out = self.forecasts_after_gru(forecasts_out)

        #combine forecasts and static
        static_for_forecasts = self.static_bridge_to_forecast(static_out)
        forecasts_and_static = forecasts_out + static_for_forecasts
        forecasts_and_static_for_history = self.static_and_forecasts_bridge_to_history(forecasts_and_static)

        #combine all
        all_combined = history_and_static_for_forecasts + forecasts_and_static_for_history
        all_combined = self.after_combine_all(all_combined)
        all_peek = history_peek + forecasts_peek
        all_peek = self.after_all_peek(all_peek)

        combine_and_peek = all_combined + all_peek
        output = self.output_layers(combine_and_peek)
        output = self.output_activation(output)
        return output

class ForeseerProbabilistic(meter_nn.MeterNnBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_layer_size = 128
        assert self.hidden_layer_size % 2 == 0, "Our bi-directional RNNs requires an even hidden layer size"
        assert self.hidden_layer_size % 8 == 0, "Our Conv1D requires a hidden layer size divisible by 8"
        self.dropout_rate = 0.1

        self.history_embed = nn.Sequential(
            nn.Linear(self.history_input_shape[-1], self.hidden_layer_size),
            nn.LayerNorm(self.hidden_layer_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nnlib.CausalConv1d(self.history_input_shape[-2], self.hidden_layer_size, kernel_size=6),
            nn.LeakyReLU(inplace=False),
        )

        self.history_peek = nn.MultiheadAttention(self.hidden_layer_size, 8)
        self.history_peek2 = nn.Sequential(
            nn.LeakyReLU(inplace=False),
        )

        self.history_gru = nn.GRU(self.hidden_layer_size, self.hidden_layer_size//2,
             num_layers=3, batch_first=True, bidirectional=True, dropout=self.dropout_rate)
        self.history_after_gru = nn.Sequential(
            nn.BatchNorm1d(self.hidden_layer_size),
            nn.LeakyReLU(inplace=False),
        )

        self.forecasts_embed = ForecastTransform(
            self.forecasts_input_shape, self.hidden_layer_size, self.dropout_rate
        )

        self.forecasts_peek = nn.MultiheadAttention(self.hidden_layer_size, 8)
        self.forecasts_peek2 = nn.Sequential(
            nn.LeakyReLU(inplace=False),
        )

        self.static_in = nn.Sequential(
            nnlib.GaussianNoise(0.1),
            nn.Linear(meter_nn.size_dims(self.static_input_shape), self.hidden_layer_size),
            nn.BatchNorm1d(self.hidden_layer_size),
            nn.LeakyReLU(inplace=True),
            nnlib.Repeat((1,self.hidden_layer_size,1), expand_dim=-2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            nn.LeakyReLU(inplace=False),
        )

        self.static_and_history_bridge_to_forecasts = nn.Sequential(
            nn.LayerNorm(self.hidden_layer_size),
            nnlib.CausalConv1d(self.hidden_layer_size, self.hidden_layer_size, kernel_size=12),
            nn.LeakyReLU(inplace=False),
        )

        self.static_and_forecasts_bridge_to_history = nn.Sequential(
            nn.LayerNorm(self.hidden_layer_size),
            nnlib.CausalConv1d(self.hidden_layer_size, self.hidden_layer_size, kernel_size=12),
            nn.LeakyReLU(inplace=False),
        )

        self.after_combine_embeds = nn.Sequential(
            nn.LayerNorm(self.hidden_layer_size),
            nnlib.CausalConv1d(self.hidden_layer_size, self.hidden_layer_size, kernel_size=4),
            nn.LeakyReLU(inplace=False),
        )

        self.after_combine_peeks = nn.Sequential(
            nn.LayerNorm(self.hidden_layer_size),
            nnlib.CausalConv1d(self.hidden_layer_size, self.hidden_layer_size, kernel_size=8),
            nn.LeakyReLU(inplace=False),
        )

        self.output_layers = nn.Sequential(
            nn.LayerNorm(self.hidden_layer_size),
            nnlib.Transpose(1, 2),
            nnlib.CausalConv1d(self.hidden_layer_size, self.output_shape[0], kernel_size=8),
            nn.LayerNorm(self.hidden_layer_size),
            nn.LeakyReLU(inplace=True),
            nnlib.Transpose(1, 2),
            nnlib.CausalConv1d(self.hidden_layer_size, self.output_shape[1], kernel_size=4),
            nnlib.Transpose(1, 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.output_shape[1], self.output_shape[1]),
            nn.Sigmoid(),
        )
        return

    def forward(self, x):
        static, history, forecasts = x
        # forecasts has shape (batch_size, num_forecasts, forecast_length, num_features)
        # history has shape (batch_size, history_length, num_features)
        # static has shape (batch_size, num_features)

        #history initial embed
        history_embed = self.history_embed(history)

        history_peek, _ = self.history_peek(history_embed, history_embed, history_embed)
        history_peek = self.history_peek2(history_peek)

        history_out, _ = self.history_gru(history_embed)
        history_out = self.history_after_gru(history_out)

        #static initial embed
        static_out = self.static_in(static)

        #combine history and static
        history_and_static = history_out + static_out
        history_and_static_for_forecasts = self.static_and_history_bridge_to_forecasts(history_and_static)

        #forecasts initial embed
        forecasts_embed = self.forecasts_embed(forecasts)

        forecasts_peek, _ = self.forecasts_peek(forecasts_embed, forecasts_embed, forecasts_embed)
        forecasts_peek = self.forecasts_peek2(forecasts_peek)

        #combine forecasts and static
        forecasts_and_static = forecasts_embed + static_out
        forecasts_and_static_for_history = self.static_and_forecasts_bridge_to_history(forecasts_and_static)

        #combine all
        all_combined = history_and_static_for_forecasts + forecasts_and_static_for_history
        all_combined = self.after_combine_embeds(all_combined)
        all_peek = history_peek + forecasts_peek
        all_peek = self.after_combine_peeks(all_peek)

        combine_and_peek = all_combined + all_peek
        output = self.output_layers(combine_and_peek)
        return output
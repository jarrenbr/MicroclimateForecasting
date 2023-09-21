import numpy as np
import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CausalConv1d, self).__init__()
        self.pad = nn.ConstantPad1d((kernel_size - 1, 0), 0)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.pad(x)
        return self.conv1(x)
		
class GaussianNoise(nn.Module):
    def __init__(self, std=0.05):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn(x.size(), device=x.device) * self.std
            return x + noise
        return x
		
        
class PositionalEncoding(nn.Module):
    #bsc input
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x_bsc):
        #x is in format batch_size, sequence_length, channels
        x_bsc = x_bsc * math.sqrt(self.d_model)
        x_bsc = x_bsc + self.pe[:x_bsc.size(0), :]
        return x_bsc

class NnBase(nn.Module):
    loss_key = "loss"
    def __init__(self,
                 loss_fn=None,
                 name="anonymous",
                 device=DEVICE,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.loss_fn = loss_fn
        self.name = name

    @property
    def losses(self):
        trainkey = "Training"
        valkey = "Validation"
        df = pd.DataFrame({trainkey: self.train_losses, valkey: self.val_losses})
        df['Epoch'] = df.index + 1
        df = df.melt(id_vars='Epoch', value_vars=[trainkey, valkey], var_name='Metric', value_name='Value')
        return df

    def optimizer_setup(self, set_func=torch.optim.AdamW, **kwargs):
        self.optimizer = set_func(self.parameters(), foreach=True, **kwargs)

    def scheduler_setup(self, set_func=torch.optim.lr_scheduler.ReduceLROnPlateau, **kwargs):
        self.scheduler = set_func(self.optimizer, **kwargs)

    def update(self, data, epochs, steps_per_epoch,
               val_data=None, val_steps_per_epoch='same',
               patience=0, save_every=0, scheduler_patience=0,
               verbose=True,
               ):
        if val_steps_per_epoch == 'same':
            val_steps_per_epoch = steps_per_epoch
        if verbose:
            _logger.info(f"Training {self.name}")
        else:
            _logger.log(ulib.MIN_LOG_FLAG, f"Training {self.name}")
        start_time = time.time()
        patience_count = -1
        best_val_loss = torch.inf

        self.scheduler_setup(patience=scheduler_patience, verbose=verbose)

        for epoch in range(epochs):
            self.train()
            train_loss_summary = torch.zeros(1, device=self.device, dtype=torch.float32)
            for i, (x,y) in zip(range(0, steps_per_epoch), data):
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self(x)
                loss = self.loss_fn(outputs, y)
                train_loss_summary += loss.detach()
                loss.backward()
                self.optimizer.step()
            train_loss_summary /= (i+1)
            self.train_losses.append(train_loss_summary.item())
            if verbose:
                _logger.info(f"Epoch {epoch+1} average loss: {train_loss_summary.item():.8f}.")
            if save_every > 0 and (epoch+1) % save_every == 0:
                self.save(filepath=get_model_path(self.name, epoch+1),)

            if val_data is not None:
                self.eval()
                with torch.no_grad():
                    val_loss_summary = torch.zeros(1, device=self.device, dtype=torch.float32)
                    for i, (x,y) in zip(range(0, val_steps_per_epoch), val_data):
                        preds = self(x)
                        loss = self.loss_fn(preds, y)
                        val_loss_summary += loss.detach()
                val_loss_summary /= (i+1)
                self.val_losses.append(val_loss_summary.item())
                #patience update
                if val_loss_summary >= best_val_loss:
                    patience_count += 1
                else:
                    best_val_loss = val_loss_summary
                    patience_count = -1
                if verbose:
                    _logger.info(f"Validation average loss: {val_loss_summary.item():.8f}.")
                if patience_count >= patience:
                    if verbose:
                        _logger.info(f"Validation loss increased from the best ({best_val_loss.item():.8f}). "
                              f"Stopping training.")
                    break
                self.scheduler.step(val_loss_summary.detach())

        if verbose:
            minutes, seconds = divmod(time.time() - start_time, 60)
            _logger.info(f"Training {self.name} took {int(minutes)} minutes and {seconds:.2f} seconds.")
        return

    def test(self, data, nsamples, reduce_mean=True):
        metrics = {
            NnBase.loss_key : np.empty(nsamples, dtype=np.float32),
            "MAE" : np.empty(nsamples, dtype=np.float32),
            "RMSE" : np.empty(nsamples, dtype=np.float32),
            "MASE" : np.empty(nsamples, dtype=np.float32),
        }
        self.eval()
        with torch.no_grad():
            for i, (x, y) in zip(range(0, nsamples), data):
                preds = self(x)
                loss = self.loss_fn(preds, y)
                metrics[NnBase.loss_key][i] = loss.item()
                metrics["MAE"][i] = torch.abs(preds - y).mean().item()
                metrics["RMSE"][i] = torch.sqrt(torch.square(preds - y).mean()).item()
                metrics["MASE"][i] = torch.abs(preds - y).mean() / torch.abs(y[1:] - y[:-1]).mean()
        if reduce_mean:
            for key, values in metrics.items():
                metrics[key] = values[:i].mean()
        return metrics

    def predict(self, dataloader, num_predictions):
        self.eval()
        all_x = []
        all_y = []
        predictions = []

        with torch.no_grad():
            for i, (x, y) in zip(range(num_predictions), dataloader):
                preds = self(x)
                all_x.append(x)
                all_y.append(y)
                predictions.append(preds)

        return all_x, all_y, predictions

    def save(self, filepath:Path):
        torch.save(self.state_dict(), filepath)
        _logger.log(ulib.MIN_LOG_FLAG, f"Saved model {self.name} to {filepath}")

    def load(self, path:Path):
        self.load_state_dict(torch.load(path))
        return self
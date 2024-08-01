import torch
from torch.optim import Adam
from pytorch_lightning import LightningModule
import pandas as pd


class GANModule(LightningModule):
    def __init__(self,
        gen,
        disc,
        latent_dim: int,
        lr: float,
        num_disc_steps: int
    ):
        super().__init__()
        self.gen = gen
        self.disc = disc
        self.latent_dim = latent_dim
        self.automatic_optimization = False
        self.lr = lr
        self.num_disc_steps = num_disc_steps

    def disc_loss(self, real_logits, fake_logits):
        real_is_real = torch.log(torch.sigmoid(real_logits) + 1e-10)
        fake_is_fake = torch.log(1 - torch.sigmoid(fake_logits) + 1e-10)
        return -(real_is_real + fake_is_fake).mean() / 2

    def gen_loss(self, fake_logits):
        fake_is_real = torch.log(torch.sigmoid(fake_logits) + 1e-10)
        return -fake_is_real.mean()

    def training_step(self, batch, batch_idx):
        gen_opt, disc_opt = self.optimizers()
        target, cond = batch
        batch_size = target.shape[0]
        seq_len = target.shape[1]
        z = torch.randn(batch_size, seq_len, self.latent_dim, device=self.device)
        for _ in range(self.num_disc_steps):
            real_logits, _ = self.disc(target, cond)
            with torch.no_grad():
                fake = self.gen(z, cond)
            fake_logits, _ = self.disc(fake, cond)
            d_loss = self.disc_loss(real_logits, fake_logits)

            disc_opt.zero_grad()
            self.manual_backward(d_loss)
            disc_opt.step()

        fake = self.gen(z, cond)
        fake_logits, _ = self.disc(fake, cond)
        g_loss = self.gen_loss(fake_logits)

        gen_opt.zero_grad()
        self.manual_backward(g_loss)
        gen_opt.step()

        self.log_dict({'train_gen_loss': g_loss, 'train_disc_loss': d_loss}, prog_bar=True)

    def configure_optimizers(self):
        gen_opt = Adam(self.gen.parameters(), lr=self.lr)
        disc_opt = Adam(self.disc.parameters(), lr=self.lr)
        return gen_opt, disc_opt

    def sample(self, data: pd.DataFrame, n_samples: int) -> list[pd.DataFrame]:
        """
        Sample time series data with slicing dataset on window_size elements and step size 1.

        Args:
            data (pd.DataFrame): A dataframe with columns for conditioning.
            n_samples (int): A number of samples to sample.
        Returns:
            list[pd.DataFrame]: A list with samples dataframes.
        """
        output_data = []
        for i in range(data.shape[0] - self.window_size + 1):
            if i == 0:
                output_data.extend(super().sample(data[i:i + self.window_size], n_samples))
            else:
                output_data.extend([[elem.values[-1]] for elem in super().sample(data[i:i + self.window_size], n_samples)])
        dct = {i:[] for i in range(n_samples)}
        for i in range(len(output_data)):
            if i < n_samples:
                dct[i].extend(output_data[i].values.tolist())
            else:
                dct[i % n_samples].extend(output_data[i])
        return [pd.DataFrame(data=val, columns=data.columns) for val in dct.values()]

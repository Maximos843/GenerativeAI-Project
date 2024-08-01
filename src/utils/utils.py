from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np


class SlidingWindowDataset(Dataset):
    def __init__(self,
            df: pd.DataFrame,
            target_columns: list[str],
            conditional_columns: list[str],
            window_size: int,
            step_size: int
        ):
        self.window_size = window_size
        self.target_columns = target_columns
        self.conditional_columns = conditional_columns
        self.df = df
        self.step_size = step_size

    def __len__(self) -> int:
        return (len(self.df) - self.window_size + 1) // self.step_size

    def __getitem__(self, idx: int) -> tuple[torch.FloatTensor]:
        target = self.df[self.target_columns].iloc[range(idx*self.step_size, idx*self.step_size + self.window_size)]
        cond = self.df[self.conditional_columns].iloc[range(idx*self.step_size, idx*self.step_size + self.window_size)]
        return torch.FloatTensor(target.values), torch.FloatTensor(cond.values)


def positional_encoding(index: pd.Index, freqs: list[str]) -> pd.DataFrame:
    encoding = []
    for freq in freqs:
        values = getattr(index, freq)
        num_values = max(values) + 1
        steps = [x * 2.0 * np.pi / num_values for x in values]
        encoding.append(pd.DataFrame({f'{freq}_cos': np.cos(steps), f'{freq}_sin': np.sin(steps)}, index=index))
    encoding = pd.concat(encoding, axis=1)
    return encoding

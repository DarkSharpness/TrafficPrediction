from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

class TrafficDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.sensor_ids = data['iu_ac'].unique()
        self.samples = self.create_samples()
        self.device = device

# 数据的处理待定
    def create_samples(self):
        samples = []
        for sensor_id in self.sensor_ids:
            sensor_data = self.data[self.data['iu_ac'] == sensor_id]
            values = sensor_data['q'].values
            for i in range(0, len(values) - self.seq_len - self.pred_len + 1, self.seq_len + self.pred_len):
                seq_x = values[i:i+self.seq_len]
                seq_y = values[i+self.seq_len:i+self.seq_len+self.pred_len]
                samples.append((seq_x, seq_y))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_x, seq_y = self.samples[idx]
        return torch.tensor(seq_x, dtype=torch.float32, device=self.device).reshape(-1, 1), torch.tensor(seq_y, dtype=torch.float32, device=self.device).reshape(-1, 1)


class TrafficDatasetTrain(Dataset):
    def __init__(self, idx, data, seq_len, pred_len, device) -> None:
        # if len(time) != len(data):
        #     raise ValueError("time and data must have the same length")
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.bag_size = seq_len + pred_len
        self.idx = idx
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        idx = self.idx[idx]
        return torch.tensor(self.data[idx: idx+self.seq_len], dtype=torch.float32, device=self.device).reshape(-1, 1), torch.tensor(self.data[idx+self.seq_len: idx+self.seq_len+self.pred_len],dtype=torch.float32, device=self.device).reshape(-1, 1)

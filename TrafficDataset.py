from torch.utils.data import Dataset
import torch


__all__ = [
    'TrafficDatasetTrain',
    'TrafficDatasetPredict',
    'TrafficDatasetFinetune',
    'TrafficDatasetPredictSpec'
]


class TrafficDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
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
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.bag_size = seq_len + pred_len
        self.idx = idx
        self.device = device

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        idx = self.idx[idx]
        x = torch.tensor(self.data[idx: idx+self.seq_len],
                         dtype=torch.float32, device=self.device).reshape(-1, 1)
        y = torch.tensor(self.data[idx+self.seq_len: idx+self.seq_len+self.pred_len],
                         dtype=torch.float32, device=self.device).reshape(-1, 1)
        return x, y


class TrafficDatasetPredict(Dataset):
    def __init__(self, filename, seq_len=24, device=torch.device('cpu')):
        super().__init__()
        lines = open(filename, 'r').readlines()
        self.data = []
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            numbers = line.strip().split(',')
            if len(numbers) != seq_len:
                raise ValueError(
                    "seq_len must be equal to the length of the data")
            numbers = [float(number) for number in numbers]
            numbers = torch.tensor(
                numbers, dtype=torch.float32, device=device).reshape(-1, 1)
            self.data.append(numbers)
        self.device = device
        if len(self.data) != 439298:
            raise ValueError("The length of the data must be 439298")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], torch.tensor([], device=self.device)


class TrafficDatasetPredictSpec(Dataset):
    def __init__(self, L: int, R: int, predict_dataset: TrafficDatasetPredict):
        super().__init__()
        self.L = L
        self.R = R
        self.predict_dataset = predict_dataset

    def __len__(self):
        return self.R - self.L + 1

    def __getitem__(self, idx):
        return self.predict_dataset[idx + self.L]

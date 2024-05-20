import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from torch import Tensor

__all__ = ['PatchTST']


from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x
    
class TrafficDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.sensor_ids = data['iu_ac'].unique()
        self.samples = self.create_samples()
        
#数据的处理待定
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
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)

file1 = pd.read_csv('iiid.csv')

file3 = pd.read_csv('for_prediction.csv')

file2 = pd.read_csv('modified_file.csv')

output_file = open('PST.csv', 'w')
output_file.write('id,estimate_q\n')

seq_len = 24  # 使用过去24小时的数据
pred_len = 1  # 预测接下来1小时的数据

k = 1

class Configs:
    seq_len = 24
    pred_len = 1
    individual = False
    enc_in = 1  # 单通道，即每个探头一个时间序列
    drop = 0.2  # Dropout率
    revin = True  # 启用RevIN
    channel = 1  # 输入通道数量
    e_layers = 2
    n_heads = 8
    d_model = 64
    d_ff = 256
    dropout = 0.15
    fc_dropout = 0.1
    head_dropout = 0.1
    patch_len = 16
    stride = 8
    padding_patch = False
    decomposition = True
    kernel_size = 25
    affine = True
    subtract_last = True

for index, row in file1.iterrows():
    current_id = row['id']

    train_data = file2[file2['iu_ac'] == current_id]

    dataset = TrafficDataset(train_data, seq_len, pred_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    configs = Configs()
    model = Model(configs)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()
    for epoch in range(200):  # 训练50个epoch
        epoch_loss = 0
        for seq_x, seq_y in dataloader:
            seq_x, seq_y = seq_x.to(device), seq_y.to(device)
            optimizer.zero_grad()
            output = model(seq_x.unsqueeze(2))  # 添加通道维度
            loss = criterion(output, seq_y.unsqueeze(2))  # 添加通道维度
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader)}')
        

    predict_data = file3[file3['iu_ac'] == current_id]

    predict_dataset = TrafficDataset(predict_data, seq_len, 0)
    predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False)

    model.eval()
    predictions = []
    with torch.no_grad():
    
        for seq_x, seq_y in predict_loader:
            seq_x = seq_x.to(device)
            output = model(seq_x.unsqueeze(2))
            predictions.append(output.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    for  prediction in predictions:
        estimate_q = int(round(prediction.item()))
        output_file.write(f"{k},{estimate_q}\n")
        k += 1
    
    del(model)
    print(index)

output_file.close()


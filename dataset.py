import torch
import torch.utils.data as Data


class Dataset(Data.Dataset):
    def __init__(self, data, label, device, mode):
        self.device = device
        self.datas = data
        self.label = label
        self.mode = mode

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        data = torch.tensor(self.datas[item]).to(self.device)
        label = self.label[item]
        return data, torch.tensor(label).to(self.device)

    def shape(self):
        return self.datas[0].shape

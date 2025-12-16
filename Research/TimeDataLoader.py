from torch.utils.data import Dataset, DataLoader

class TimeWindowDataset(Dataset):
    def __init__(self, features, labels, window_size):
        # features 形状: [总样本数, window_size, features] → 如 [1164, 10, 20]
        # labels 形状: [总样本数, output_dim] → 如 [1164, 2]
        self.features = features
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.features)  # 总样本数（已按窗口切割）

    def __getitem__(self, idx):
        # 输入：时间窗口（保留时间维度）
        x = self.features[idx]  # 形状: [window_size, features] → 如 [10, 20]
        # 标签：当前窗口最后一个时间步的标签（与输入窗口对齐）
        y = self.labels[idx]    # 形状: [output_dim] → 如 [2]
        return x, y

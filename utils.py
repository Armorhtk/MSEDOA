import os
import json
import random
import numpy as np
from tqdm.auto import tqdm
import argparse
import torch
from scipy.signal import find_peaks
from torch.utils.data import Dataset,DataLoader

def seed_everything(seed):
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

# 生成导向矢量 A, 矢量大小为 N x len(deg)
def steering_vector(N, deg):
    dd = 0.5  # Element spacing (in units of wavelength)
    l = torch.arange(0, N).view(-1, 1)  # Antenna element indices [0, 1, ..., N-1]
    theta = deg * torch.pi / 180  # Convert degrees to radians
    return torch.exp(- 1j * 2 * torch.pi * dd * l * torch.sin(theta))  # Complex exponential for each phase shift

# 生成单个复数信号 X(t)
def generate_complex_signal(N=10, snr_db=10, deg=torch.tensor([30]), snapshot=1):

    # A --> ( N, len(deg) ); if len(deg) = 1 is single-DOA estimation, elif len(deg) > 1 multi-DOA estimation
    a_theta = steering_vector(N, deg)
    # S(t)
    phase = torch.exp(2j * torch.pi * torch.randn(a_theta.size()[1], snapshot))
    # X(t) = A * S(t)
    signal = torch.matmul(a_theta.to(phase.dtype), phase)
    # N(t)
    signal_power = torch.mean(torch.abs(signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_real = torch.sqrt(noise_power / 2) * torch.randn_like(signal.real)
    noise_imag = torch.sqrt(noise_power / 2) * torch.randn_like(signal.imag)
    noise = torch.complex(noise_real, noise_imag)
    # X(t) = A * S(t) + N(t)
    signal = signal + noise 
    return signal

# 生成有网格的角度标签
def generate_ongrid_label(degrees, min_angle=-30, max_angle=30):
    labels = torch.zeros(max_angle - min_angle + 1)
    indices = degrees - min_angle
    labels[indices.long()] = 1
    return labels

# 生成模拟数据集的核心函数
def generate_data(N,
                  num_samples=1,
                  max_targets=3,
                  snapshot=128,
                  min_angle=-30,
                  max_angle=30,
                  snr_levels=[-30,30],
                  targets_strategy='random', 
                  folder_path='data',
                  min_angleMargin=10,
                  ):
    # 定义角度范围 [-0,0]
    angles = torch.arange(min_angle, max_angle + 1, 1)
    
    # 创建文件夹, 分别保存信号和标签
    signal_folder = os.path.join(folder_path, 'signal')
    label_folder = os.path.join(folder_path, 'label')
    os.makedirs(signal_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    # 第一层循环：设置不同信噪比
    for snr_db in tqdm(range(snr_levels[0], snr_levels[1], 5), desc='SNR levels', unit='snr', dynamic_ncols=True): 
        all_signals, all_labels = [], []
        # 第二层循环：每种信噪比生成指定数目的接收信号
        for _ in range(num_samples):
            # 随机生成目标数目，范围在 [1, max_targets] 之间
            if targets_strategy == 'random':
                num_targets = torch.randint(1, max_targets + 1, (1,)).item()
            # 生成固定目标数目，数目等同于max_targets, 最少为1个目标
            elif targets_strategy == 'fixed':
                num_targets = max(1, max_targets)
            else:
                raise ValueError("Invalid strategy. Choose either 'random' or 'fixed'.")
            # 随机选择目标角度
            deg_indices = torch.randperm(len(angles))[:num_targets]
            # 生成目标角度
            degs = angles[deg_indices]
            # 获取真实标签
            label = generate_ongrid_label(degs,min_angle=min_angle,max_angle=max_angle)
            # 生成接收信号
            noisy_signal = generate_complex_signal(N=N, snr_db=snr_db, deg=degs, snapshot=snapshot)
            # 保存接收信号和标签
            all_signals.append(noisy_signal)
            all_labels.append(label)
        # 不同信噪比的接收信号和标签保存在同一个文件夹下，做好命名
        torch.save(all_signals, os.path.join(signal_folder, f'signals_snr_{snr_db}dB.pt'))
        torch.save(all_labels, os.path.join(label_folder, f'labels_snr_{snr_db}dB.pt'))
    return None 

# 定义有网格DOA数据集类
class OnGridDOADataset(Dataset):
    def __init__(self, file_paths, label_paths):
        """
        Initializes a dataset containing signals and their corresponding labels.

        Args:
            file_paths (list): Paths to files containing signals.
            label_paths (list): Paths to files containing labels.
        """
        self.signals = [torch.stack(torch.load(file), dim=0) for file in file_paths]
        self.labels = [torch.stack(torch.load(label), dim=0) for label in label_paths]
        self.signals = torch.cat(self.signals, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]
    
# 创建数据加载器
def create_dataloader(data_path, batch_size=32, shuffle=True):
    """
    Create a DataLoader for batching and shuffling the dataset.

    Args:
        data_path (str): Path to the directory containing the data files.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: Configured DataLoader for the dataset.
    """
    signal_dir_path = os.path.join(data_path, "signal")
    label_dir_path = os.path.join(data_path, "label")
    signal_files = [os.path.join(signal_dir_path, f) for f in os.listdir(signal_dir_path) if 'signals' in f]
    label_files = [os.path.join(label_dir_path, f) for f in os.listdir(label_dir_path) if 'labels' in f]
    dataset = OnGridDOADataset(sorted(signal_files), sorted(label_files))

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def Calculate_DOA_RMSE(doa_grid, labels, preds):
    acc = 0
    mse = 0
    for label,spec in zip(labels, preds):
      true_targets = np.sort(doa_grid[label == 1])
      nums_targets = len(true_targets)
      probs = (spec - np.min(preds) )/ (np.max(spec) - np.min(spec))
      peaks, properties = find_peaks(probs, height=0.0)
      top_peaks = peaks[np.argsort(properties['peak_heights'])[-nums_targets:]]
      top_degs = np.sort(doa_grid[top_peaks])
      acc += np.sum(np.abs(top_degs - true_targets) <= 1) / nums_targets
      mse += np.mean((true_targets - top_degs) ** 2)
    acc = acc / len(labels)
    rmse = np.sqrt(mse / len(labels))
    return rmse,acc

# 保存参数
def save_args_as_json(args, json_path):
    # Convert argparse Namespace to dictionary
    args_dict = vars(args)
    with open(json_path, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

# 加载参数
def load_args_from_json(json_path):
    with open(json_path, 'r') as json_file:
        args_dict = json.load(json_file)
    return argparse.Namespace(**args_dict)
 
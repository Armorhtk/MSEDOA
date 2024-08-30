import torch
import numpy as np
from tqdm.auto import tqdm
from utils import steering_vector

# from scipy.signal import find_peaks
# def doa_estimate(ang_list, spec, target_nums):
#     peaks, _ = find_peaks(spec, height=None)
#     # 按照峰值从小到大排序对应的索引
#     sorted_indices = peaks[np.argsort(spec[peaks])]
#     pred_deg = ang_list[sorted_indices]
#     # 保证预测角度的长度与真实角度长度一致
#     pred_deg = pred_deg[-target_nums:]
#     # 对pre_deg进行排序
#     pred_deg = np.sort(pred_deg)
#     return pred_deg

def cbf_doa(X, Theta):
    """
    Compute CBF-DOA estimation spectrum
    Args:
        X: Array received signal (kelm x snapshot)
        numSignal: Number of signal sources (not used in this function, included for consistency)
        dd: Element spacing wavelength ratio
        Theta: Angles to traverse
    Returns:
        P_CBF: Normalized CBF-DOA estimation spectrum
    """
    kelm = X.shape[0]
    CBF = torch.zeros(Theta.shape)
    for i,deg in enumerate(Theta):
        weight = steering_vector(kelm,deg)
        CBF[i] = torch.abs(weight.conj().T @ X @ X.conj().T @ weight)

    logP0 = torch.log10(CBF / torch.max(CBF))
    P_CBF = (logP0 - torch.min(logP0)) / (torch.max(logP0) - torch.min(logP0))  # Normalization
    
    return P_CBF.numpy()

def capon_doa(X, Theta):
    """
    Capon-DOA estimation spectrum.
    X: array received signal (tensor)
    numSignal: number of signal sources
    dd: element spacing wavelength ratio
    Theta: angles to scan
    """
    L = X.size(1)
    kelm = X.size(0)
    # R = X @ X.conj().t() / L
    R = torch.matmul(X, X.conj().T) / L
    R_inv = torch.inverse(R)
    P_Capon = torch.zeros(len(Theta))
    for i,deg in enumerate(Theta):
        a_vector = steering_vector(kelm,deg)
        P_Capon[i] = torch.abs(1 / (a_vector.conj().t() @ R_inv @ a_vector))
    logP0 = torch.log10(P_Capon / torch.max(P_Capon))
    P_Capon = (logP0 - torch.min(logP0)) / (torch.max(logP0) - torch.min(logP0))  # Normalize
    return P_Capon.numpy()

def music_doa(X, Theta, numSignal):
    """
    MUSIC-DOA estimation spectrum.
    X: array received signal (tensor)
    numSignal: number of signal sources
    dd: element spacing wavelength ratio
    Theta: angles to scan
    """
    L = X.size(1)
    kelm = X.size(0)
    R = torch.matmul(X, X.conj().T) / L
    eigenvalues, eigenvectors = torch.linalg.eigh(R)
    Un = eigenvectors[:, :kelm - numSignal]
    P_MUSIC = torch.zeros(len(Theta))
    for i,deg in enumerate(Theta):
        a_theta = steering_vector(kelm,deg)
        P_MUSIC[i] = 1 / torch.abs((a_theta.conj().T @ Un @ Un.conj().T @ a_theta)[0])
        
    logP0 = torch.log10(P_MUSIC / torch.max(P_MUSIC))
    P_MUSIC = (logP0 - torch.min(logP0)) / (torch.max(logP0) - torch.min(logP0)) 
    return P_MUSIC.numpy()

def esprit_doa(X, numSignal):
    """
    P_ESPRIT: Two angles estimated by ESPRIT-DOA
    X: Array received signal
    numSignal: Number of signal sources
    dd: Ratio of element spacing to wavelength
    Theta: Angle traversal
    """
    L = X.shape[1]
    kelm = X.shape[0]
    subkelm = X.shape[0] // 4
    ESP_X1 = X[:kelm - subkelm, :]
    ESP_X2 = X[1:kelm - subkelm + 1:, :]
    ESP_XX = torch.cat((ESP_X1, ESP_X2), dim=0)
    R = ESP_XX @ ESP_XX.conj().T / L
    _,Evetor = torch.linalg.eigh(R)
    Us = Evetor[:, 2*(kelm-subkelm)-numSignal:2*(kelm-subkelm)]
    Us1 = Us[:kelm - subkelm, :]
    Us2 = Us[kelm - subkelm:2*(kelm - subkelm), :]

    # LS-ESPRIT
    ESP_ANS1 = torch.pinverse(Us1.conj().T @ Us1) @ Us1.conj().T @ Us2
    # 获取对角线元素并解算来向角
    Pusai1 = -torch.angle(torch.linalg.eigvals(ESP_ANS1))
    # 将相位角转化为角度（弧度）
    Pusai1 = torch.arcsin(Pusai1 * 1 / (2 * torch.pi * 0.5))
    # 将弧度转化为度数
    P_LS_ESPRIT = torch.rad2deg(Pusai1)

    # TLS-ESPRIT
    ESP_US12 = torch.cat((Us1, Us2), dim=1)
    _, ESP_V2 = torch.linalg.eigh(ESP_US12.conj().T @ ESP_US12)
    ESP_EN2 = ESP_V2[:, :numSignal]
    ESP_ANS2 = -ESP_EN2[:numSignal, :] @ torch.pinverse(ESP_EN2[numSignal:2*numSignal,:])
    Pusai2 = -torch.angle(torch.linalg.eigvals(ESP_ANS2))
    Pusai2 = torch.arcsin(Pusai2 * 1 / (2 * torch.pi * 0.5))
    P_TLS_ESPRIT = torch.rad2deg(Pusai2)
    return P_LS_ESPRIT.numpy(), P_TLS_ESPRIT.numpy()

## IAA 迭代自适应算法
def iaa_doa(X, Theta):
    N = X.size(0)
    A = steering_vector(N, Theta)
    AH = torch.transpose(A.conj(),0,1)
    N,K = A.shape
    threshold = 0e-3
    iter_num = 30

    Pk = np.zeros(K, dtype=np.complex128)
    
    X = X.squeeze().numpy()
    A = A.numpy()
    AH = AH.resolve_conj().numpy()

    Pk = (np.sum(AH @ X, axis=1) / N) ** 2
    Po = np.diag(Pk)
    P = A @ Po @ AH

    # Main iteration
    for _ in range(iter_num):
        P += np.eye(N) * threshold
        ak_P = AH @ np.linalg.pinv(P)
        T = ak_P @ X
        B = ak_P @ A
        b = B.diagonal()
        sk = np.sum(T,axis=1)/np.abs(b)
        Pk = np.abs(sk) ** 2
        Poo = np.diag(Pk)
        P = A @ Poo @ A.conj().T
    spec = Pk
    logP0 = np.log10(spec / np.max(spec))
    P_IAA = (logP0 - np.min(logP0)) / (np.max(logP0) - np.min(logP0))
    return torch.tensor(P_IAA)

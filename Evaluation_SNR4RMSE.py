import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import torch
import torch.nn.functional as F
from utils import seed_everything
from utils import load_args_from_json
from utils import generate_complex_signal
from network.DNN import DOADNN
from network.ResNet import DOAResNet
from network.SEResNet import DOASEResNet
from network.MSEDOANet import MSEDOANet
from methods import cbf_doa,music_doa,esprit_doa,iaa_doa
from tqdm.auto import tqdm

def DL_DOA(X,model):
    model.eval()
    probs = model(X.unsqueeze(0))[0]
    # probs = F.sigmoid(probs)
    P_DL = (probs - torch.min(probs) )/ (torch.max(probs) - torch.min(probs))
    return P_DL.detach().numpy()

def search_peaks(doa_grid,doa_spectrum,nums_targets):
    peaks, properties = find_peaks(doa_spectrum, height=0.0)
    top_peaks = peaks[np.argsort(properties['peak_heights'])[-nums_targets:]]
    top_degs = np.sort(doa_grid[top_peaks])
    return top_degs

def main(args):
  seed_everything(2024)
  # Get doa grid list, array elements number, doa grid size
  doa_grid = torch.arange(args.min_angle,args.max_angle + 1,1)
  arrayelements_Number = args.number_elements
  doagrid_classesnums = len(doa_grid)

  # Chioces model and Instantiate model
  if args.model_name == 'DNN':
    model =  DOADNN(arrayelements_Number, doagrid_classesnums, args.snapshot)
  elif args.model_name == 'ResNet':
    model = DOAResNet(arrayelements_Number, doagrid_classesnums)
  elif args.model_name == 'SEResNet':
    model = DOASEResNet(arrayelements_Number, doagrid_classesnums)
  elif args.model_name == 'MSEDOANet':
    model = MSEDOANet(arrayelements_Number, doagrid_classesnums)
  else:
    raise ValueError('Invalid model name')
  model.load_state_dict(torch.load(os.path.join(args.checkpoints_model,'best_model.pth')))
  # model.eval()

  rmse_metrics = {
    "MSEDOA":[],
    "CBF":[],
    "MUSIC":[],
    "ESPRIT":[],
    # "LS_ESPRIT":[],
    # "TLS_ESPRIT":[],
    # "IAA":[],
  }
  
  snr_levels = np.arange(args.min_snr,args.max_snr,5)

  # create two theta list

  # 从角度范围内随机采样MonteCarlo_nums个不重复的角度
  angle_range = torch.linspace(-60, 53, steps=args.MonteCarlo_nums * 3)
  theta1 = angle_range[torch.randperm(angle_range.size(0))[:args.MonteCarlo_nums]]
  theta2 = theta1 + args.set_angle_spacing
  true_theta_pair = torch.stack((theta1,theta2),dim=1)
  
  # # 选择和实验一相同的角度
  # theta1 = torch.arange(args.min_angle, args.max_angle - int(args.set_angle_spacing), 1)
  # theta2 = theta1 + args.set_angle_spacing
  # true_theta_pair = torch.stack((theta1,theta2),dim=1)

  # 使用蒙特卡洛测试不同SNR下的DOA估计准确性
  for snr_db in tqdm(snr_levels, desc='SNR levels', unit='snr', dynamic_ncols=True):
    
    method_errors = {key:[] for key in rmse_metrics}

    for theta_pair in true_theta_pair:
      nums_targets = len(theta_pair)
      # Run simulation
      X = generate_complex_signal(N=args.number_elements,
                                  snr_db=snr_db,
                                  deg=theta_pair,
                                  snapshot=args.snapshot)
      # Run each method
      for method_name in method_errors.keys():
          if method_name == "MSEDOA":
              spec = DL_DOA(X,model)
              pred = search_peaks(doa_grid, spec, nums_targets)
              method_errors[method_name].append(pred)
          elif method_name == "CBF":
              spec = cbf_doa(X, doa_grid)
              pred = search_peaks(doa_grid, spec, nums_targets)
              method_errors[method_name].append(pred)
          elif method_name == "MUSIC":
              spec = music_doa(X, doa_grid, nums_targets)
              pred = search_peaks(doa_grid,spec,nums_targets)
              method_errors[method_name].append(pred)
          elif method_name == "ESPRIT":
              pred,_ = esprit_doa(X, nums_targets)
              pred = np.sort(pred)
              method_errors[method_name].append(pred)
          elif method_name == "LS_ESPRIT":
              pred,_ = esprit_doa(X, nums_targets)
              pred = np.sort(pred)
              method_errors[method_name].append(pred)
          elif method_name == "TLS_ESPRIT":
              _,pred = esprit_doa(X, nums_targets)
              pred = np.sort(pred)
              method_errors[method_name].append(pred)
          elif method_name == "IAA":
              spec = iaa_doa(X,doa_grid)
              pred = search_peaks(doa_grid,spec,nums_targets)
              method_errors[method_name].append(pred)

    for method_name, pred_theta in method_errors.items():
      pred_theta = np.array(pred_theta)
      true_theta1 = true_theta_pair[:,0]
      true_theta2 = true_theta_pair[:,1]
      pred_theta1 = pred_theta[:,0]
      pred_theta2 = pred_theta[:,1]
      theta1_error = (true_theta1 - pred_theta1).numpy()
      theta2_error = (true_theta2 - pred_theta2).numpy()
      rmse = np.sqrt(np.mean(theta1_error**2 + theta2_error**2))
      rmse_metrics[method_name].append(rmse)
  print(rmse_metrics)
  plt.figure(figsize=(8, 6), dpi=600)

  line_styles = {
      "MSEDOA": {"color": "red", "linestyle": "-", "linewidth": 2, "marker": "o"},
      "CBF": {"color": "green", "linestyle": "-.", "linewidth": 2, "marker": "s"},
      "MUSIC": {"color": "blue", "linestyle": "--", "linewidth": 2, "marker": "d"},
      # "LS_ESPRIT": {"color": "purple", "linestyle": ":", "linewidth": 2, "marker": "v"},
      # "TLS_ESPRIT": {"color": "orange", "linestyle": "-", "linewidth": 2, "marker": "^"},
      "ESPRIT": {"color": "orange", "linestyle": "--", "linewidth": 2, "marker": "^"},
      # "IAA": {"color": "brown", "linestyle": "--", "linewidth": 2, "marker": "p"},
  }
  for method_name, rmse in rmse_metrics.items():
    plt.plot(snr_levels, rmse, label=method_name, **line_styles[method_name])
  plt.xlabel('SNR [dB]', fontsize=14)
  plt.ylabel('RMSE [log10]', fontsize=14)
  plt.yscale('log')
  plt.title('SNR vs RMSE', fontsize=16, fontweight='bold')
  plt.grid(True, linestyle='--', linewidth=0.5)
  plt.legend(loc='best', fontsize=12)
  plt.xticks(np.arange(-20,35,5),fontsize=12)
  plt.yticks([10**-1,10**0, 10**1, 10**2])
  plt.tight_layout()
  plt.savefig(os.path.join(args.output_dir,'SNR4RMSE.png'))
  plt.show()


if __name__ == '__main__':
  # Parse arguments
    parser = argparse.ArgumentParser(description="Run Monte Carlo simulations for DOA estimation accuracy")
    parser.add_argument('--MonteCarlo_nums',type=int,default = 100) 
    parser.add_argument('--min_snr', type=int, default=-20, help="Number of snapshots to simulate")
    parser.add_argument('--max_snr', type=int, default=35, help="Number of snapshots to simulate")
    parser.add_argument('--set_angle_spacing', type=float, default=6.1, help="Angle spacing to simulate")
    parser.add_argument('--checkpoints_model', type=str, default="checkpoints/Model20240825181151", help="Number of targets to simulate")
    parser.add_argument('--output_dir', type=str, default="image", help="Output directory")
    args =parser.parse_known_args()[0]

    # add simulation args to learning args
    sumlation_args = load_args_from_json(os.path.join(args.checkpoints_model, 'learning_args.json'))
    for key, value in vars(sumlation_args).items():
        setattr(args, key, value)

    # args = parser.parse_args()
    main(args)
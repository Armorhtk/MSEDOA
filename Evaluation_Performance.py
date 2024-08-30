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

def DL_DOA(X,model):
    model.eval()
    probs = model(X.unsqueeze(0))[0]
    probs = F.sigmoid(probs)
    P_DL = (probs - torch.min(probs))/ (torch.max(probs) - torch.min(probs))
    return P_DL.detach().numpy()

def search_peaks(doa_grid,doa_spectrum,nums_targets):
    peaks, properties = find_peaks(doa_spectrum, height=0.0)
    top_peaks = peaks[np.argsort(properties['peak_heights'])[-nums_targets:]]
    top_degs = np.sort(doa_grid[top_peaks])
    return top_degs

def main(args):
  seed_everything(args.seed)
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
  model.eval()
  
  # create two theta list
  theta1 = torch.arange(args.min_angle, args.max_angle - int(args.set_angle_spacing), 1)
  theta2 = theta1 + args.set_angle_spacing
  true_theta_pair = torch.stack((theta1,theta2),dim=1)

  method_pred_theta = {
    "MSEDOA":[],
    "CBF":[],
    "MUSIC":[],
    # "LS_ESPRIT":[],
    # "TLS_ESPRIT":[],
    "ESPRIT":[],
    "IAA":[],
  }

  for theta_pair in true_theta_pair:
    nums_targets = len(theta_pair)
    # Run simulation
    X = generate_complex_signal(N=args.number_elements,
                                snr_db=args.set_snr,
                                deg=theta_pair,
                                snapshot=args.snapshot)
    
    # Run each method
    for method_name, pred_theta in method_pred_theta.items():
        if method_name == "MSEDOA":
            spec = DL_DOA(X,model)
            pred = search_peaks(doa_grid, spec, nums_targets)
            method_pred_theta[method_name].append(pred)
        elif method_name == "CBF":
            spec = cbf_doa(X, doa_grid)
            pred = search_peaks(doa_grid, spec, nums_targets)
            method_pred_theta[method_name].append(pred)
        elif method_name == "MUSIC":
            spec = music_doa(X, doa_grid, nums_targets)
            pred = search_peaks(doa_grid,spec,nums_targets)
            method_pred_theta[method_name].append(pred)
        elif method_name == "ESPRIT":
            pred,_ = esprit_doa(X, nums_targets)
            pred = np.sort(pred)
            method_pred_theta[method_name].append(pred)
        elif method_name == "LS_ESPRIT":
            pred,_ = esprit_doa(X, nums_targets)
            pred = np.sort(pred)
            method_pred_theta[method_name].append(pred)
        elif method_name == "TLS_ESPRIT":
            _,pred = esprit_doa(X, nums_targets)
            pred = np.sort(pred)
            method_pred_theta[method_name].append(pred)
        elif method_name == "IAA":
            spec = iaa_doa(X,doa_grid)
            pred = search_peaks(doa_grid,spec,nums_targets)
            method_pred_theta[method_name].append(pred)
  
  for method_name, pred_theta in method_pred_theta.items():
    pred_theta = np.array(pred_theta)
    true_theta1 = true_theta_pair[:,0]
    true_theta2 = true_theta_pair[:,1]
    pred_theta1 = pred_theta[:,0]
    pred_theta2 = pred_theta[:,1]
    theta1_error = true_theta1 - pred_theta1
    theta2_error = true_theta2 - pred_theta2

    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(range(len(true_theta1)), true_theta1, 'r-', label=r'$\Theta_1$', markersize=6)
    plt.plot(range(len(true_theta2)), true_theta2, 'b-', label=r'$\Theta_2$', markersize=6)
    plt.plot(range(len(pred_theta1)), pred_theta1, 'r^', label=r'$\hat{\Theta}_1$', markersize=6, markerfacecolor='none')
    plt.plot(range(len(pred_theta2)), pred_theta2, 'b^', label=r'$\hat{\Theta}_2$', markersize=6, markerfacecolor='none')
    plt.xlabel('Samples Index', fontsize=12)
    plt.ylabel('DOA [degrees]', fontsize=12)
    if method_name == "MSEDOA":
       plt.title('{}(our)'.format(method_name), fontsize=14, fontweight='bold')
    else:
      plt.title('{}'.format(method_name), fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(loc='best', fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(range(-120,125,20),fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, '{}.png'.format(method_name)))
    plt.show()

    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(range(len(theta1_error)), theta1_error, 'ro', label=r'$\Delta\Theta_1 = \Theta_1 - \hat{\Theta}_1$', markersize=6, markerfacecolor='none')
    plt.plot(range(len(theta2_error)), theta2_error, 'bs', label=r'$\Delta\Theta_2 = \Theta_2 - \hat{\Theta}_2$', markersize=6, markerfacecolor='none')
    plt.xlabel('Samples Index', fontsize=12)
    plt.ylabel('DOA Error [degrees]', fontsize=12)
    plt.title('{} errors'.format(method_name), fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(loc='best', fontsize=10)
    plt.xticks(fontsize=12)
    if method_name == "MSEDOA":
      plt.yticks(np.arange(-3,3.5,0.5))
      plt.ylim(ymin=-3,ymax=3)
    elif method_name == "CBF":
      plt.yticks(np.arange(-30,31,5))
      plt.ylim(ymin=-30,ymax=30)
    elif method_name == "MUSIC":
      plt.yticks(np.arange(-60,61,10))
      plt.ylim(ymin=-60,ymax=60)
    elif method_name == "ESPRIT":
      plt.yticks(np.arange(-8,9,1))
      plt.ylim(ymin=-8,ymax=8)
    elif method_name == "IAA":
      plt.yticks(np.arange(-120,121,20))
      plt.ylim(ymin=-120,ymax=120)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, '{}_errors.png'.format(method_name)))
    plt.show()

if __name__ == '__main__':
  # Parse arguments
    parser = argparse.ArgumentParser(description="Run Monte Carlo simulations for DOA estimation accuracy")
    parser.add_argument('--set_snr', type=int, default=12, help="Number of snapshots to simulate")
    parser.add_argument('--set_angle_spacing', type=float, default=4.7, help="Angle spacing to simulate")
    parser.add_argument('--checkpoints_model', type=str, default="checkpoints/Model20240825181151", help="Number of targets to simulate")
    parser.add_argument('--output_dir', type=str, default="image", help="Output directory")
    args =parser.parse_known_args()[0]

    # add simulation args to learning args
    sumlation_args = load_args_from_json(os.path.join(args.checkpoints_model, 'learning_args.json'))
    for key, value in vars(sumlation_args).items():
        setattr(args, key, value)

    # args = parser.parse_args()
    main(args)
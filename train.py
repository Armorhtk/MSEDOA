import os
import json
import time
import numpy as np
from scipy.signal import argrelextrema

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import create_dataloader
from utils import seed_everything
from utils import save_args_as_json,load_args_from_json
from utils import Calculate_DOA_RMSE 
from network.DNN import DOADNN
from network.ResNet import DOAResNet
from network.SEResNet import DOASEResNet
from network.MSEDOANet import MSEDOANet

import argparse

def main(args):
  # create save model path
  timestamp = time.strftime("%Y%m%d%H%M%S")
  checkpoint_dir = os.path.join(args.save_model_path, f'Model{timestamp}')
  os.makedirs(checkpoint_dir, exist_ok=True)
  best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
  save_args_as_json(args, os.path.join(checkpoint_dir, 'learning_args.json'))

  # set device (if GPU is available, use GPU else use CPU)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Get doa grid list, array elements number, doa grid size
  doa_grid = torch.arange(args.min_angle,args.max_angle + 1,1)
  arrayelements_Number = args.number_elements
  doagrid_classesnums = len(doa_grid)

  # set dataset path
  train_data_path = os.path.join(args.dataset_name,'train')
  val_data_path = os.path.join(args.dataset_name,'val')
  # create train dataloader and validation dataloader
  train_loader = create_dataloader(train_data_path, args.train_batch_size, shuffle=True)
  val_loader = create_dataloader(val_data_path, args.val_batch_size, shuffle=False)

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
  model.to(device)

  # create optimizer, scheduler 
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
  # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_epochnums, gamma=args.scheduler_gamma)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6)

  # create loss function
  criterion = nn.BCEWithLogitsLoss()

  print('Start training...')
  best_result = {"Best_valloss": float('inf'), "Best_valmse": float('inf'), "Best_valacc": 0.0}
  early_stop_count = 0
  for epoch in range(args.epochs):
    """ Training Epoch"""
    model.train()
    for signals, labels in train_loader:
      signals, labels = signals.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(signals)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
    scheduler.step()

    """ Validation Epoch"""
    model.eval()
    val_loss = 0
    val_mse = 0
    val_acc = 0
    with torch.no_grad():
      for signals, labels in val_loader:
        signals, labels = signals.to(device), labels.to(device)
        outputs = model(signals)
        val_loss += criterion(outputs, labels)
        outputs = F.sigmoid(outputs)
        mse,acc = Calculate_DOA_RMSE(doa_grid, labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
        val_mse += mse
        val_acc += acc
      val_loss /= len(val_loader)
      val_mse /= len(val_loader)
      val_acc /= len(val_loader)

    """ Save best model and Early stopping """
    if val_mse < best_result['Best_valmse']:
      best_result['Best_valmse'] = val_mse

    if val_loss < best_result['Best_valloss']:
      best_result['Best_valloss'] = val_loss.item()

    if val_acc > best_result['Best_valacc']:
      best_result['Best_valacc'] = val_acc
      torch.save(model.state_dict(), best_model_path)
      print(f'Epoch [{epoch}/{args.epochs}], Train Loss: {loss.item()} Val Loss: {val_loss.item()} Acc:{val_acc} MSE:{val_mse} model saved!') 
      early_stop_count = 0
    else:
      early_stop_count += 1
      print(f'Epoch [{epoch}/{args.epochs}], Train Loss: {loss.item()} Val Loss: {val_loss.item()} Acc:{val_acc} MSE:{val_mse}')  



    if early_stop_count >= 15:
      break

    print(f"BEST RESULT: {best_result}")

if __name__ == '__main__':
  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_name', type=str, default='data_ongrid_100k')
  parser.add_argument('--train_batch_size', type=int, default=256)
  parser.add_argument('--val_batch_size', type=int, default=1024)
  parser.add_argument('--lr', type=float, default=0.0001)
  parser.add_argument('--epochs', type=int, default=100)
  parser.add_argument('--model_name', type=str, default='MSEDOANet')
  parser.add_argument('--scheduler_epochnums', type=int, default=5)
  parser.add_argument('--scheduler_gamma', type=float, default=0.85)
  parser.add_argument('--save_model_path', type=str, default='checkpoints')
  parser.add_argument('--seed', type=int, default=42)
  args = parser.parse_args()

  # add simulation args to learning args
  sumlation_args = load_args_from_json(os.path.join(args.dataset_name, 'sumlation_args.json'))
  for key, value in vars(sumlation_args).items():
      setattr(args, key, value)
  print("================================")
  for key, value in vars(args).items():
      print(f"{key}: {value}")
  print("================================")
  save_args_as_json(args, os.path.join(args.dataset_name, 'learning_args.json'))

  # run training
  seed_everything(args.seed)
  main(args)
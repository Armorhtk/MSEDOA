import os
import argparse
from utils import seed_everything
from utils import generate_data
from utils import save_args_as_json

def parse_args():
    parser = argparse.ArgumentParser(description='Generate simulation data for radar target detection')
    parser.add_argument('--folder_path', type=str, default='./data_ongrid_100k', help='Folder path to save simulation data')
    parser.add_argument('--number_elements', type=int, default=12, help='Number of elements in the array')
    parser.add_argument('--train_num_samples', type=int, default=180000, help='Number of training samples')
    parser.add_argument('--val_num_samples', type=int, default=20000, help='Number of validation samples')
    parser.add_argument('--max_targets', type=int, default=3, help='Maximum number of targets in the scene')
    parser.add_argument('--snapshot', type=int, default=10, help='Number of snapshots')
    parser.add_argument('--strategy', type=str, default='random', help='Strategy to generate targets')
    parser.add_argument('--min_angle', type=int, default=-60, help='Minimum angle of the targets')
    parser.add_argument('--max_angle', type=int, default=60, help='Maximum angle of the targets')
    parser.add_argument('--min_angleMargin', type=int, default=10, help='Minimum angle margin between targets')
    parser.add_argument('--SNR_levels', type=list, default=[0,35], help='SNR levels')
    parser.add_argument('--SumlationSEED', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    return args

def SumlationData_Load(args):
    # 指定文件夹名称，自动创建文件夹
    folder_path = os.path.join(os.getcwd(),args.folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹已创建，路径{folder_path} ")
    else:
        print(f"文件夹已存在，路径{folder_path} ")

    """ 生成数据 """
    # 如果数据不存在，则生成数据
    if not os.path.exists(os.path.join(folder_path, 'train/signal/signals_snr_0dB.pt')):
      
      # Generate train data
      print("数据不存在，开始生成数据")
      train_path = os.path.join(folder_path, 'train')
      generate_data(N=args.number_elements,
                    num_samples=args.train_num_samples,
                    max_targets=args.max_targets,
                    snapshot=args.snapshot,
                    targets_strategy=args.strategy,
                    min_angle=args.min_angle,
                    max_angle=args.max_angle,
                    min_angleMargin=args.min_angleMargin,
                    snr_levels=args.SNR_levels,
                    folder_path=train_path
                    )

      # Generate val data
      valid_path = os.path.join(folder_path, 'val')
      generate_data(N=args.number_elements,
                    num_samples=args.val_num_samples,
                    max_targets=args.max_targets,
                    snapshot=args.snapshot,
                    targets_strategy=args.strategy,
                    min_angle=args.min_angle,
                    max_angle=args.max_angle,
                    snr_levels=args.SNR_levels,
                    folder_path=valid_path
                    )
    else:
      print("模拟数据已经存在")


if __name__ == '__main__':
  args = parse_args()
  seed_everything(args.SumlationSEED)
  SumlationData_Load(args)
  save_args_as_json(args, os.path.join(args.folder_path, 'sumlation_args.json'))
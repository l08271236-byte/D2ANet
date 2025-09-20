import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=300, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')  # 5e-5
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=416, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')  # 100
parser.add_argument('--load', type=str, default='/home/mapeiyao/D2ANet-main/D2ANet/smt_base.pth', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default='/home/D2ANet-main/D2ANet/datasets/COD_datasets/TrainDataset/Imgs/', help='the training rgb images root')
parser.add_argument('--gt_root', type=str, default='/home/D2ANet-main/D2ANet/datasets/COD_datasets/TrainDataset/GT/', help='the training gt images root')
parser.add_argument('--edge_root', type=str, default='/home/D2ANet-main/D2ANet/datasets/COD_datasets/TrainDataset/Edge/', help='the training gt images root')
parser.add_argument('--test_rgb_root', type=str, default='/home/D2ANet-main/D2ANet/datasets/COD_datasets/TestDataset/CAMO/Imgs/', help='the test gt images root')
parser.add_argument('--test_gt_root', type=str, default='/home/D2ANet-main/D2ANet//datasets/COD_datasets/TestDataset/CAMO/GT/', help='the test gt images root')
parser.add_argument('--save_path', type=str, default='./cpts/', help='the path to save models and logs')
opt = parser.parse_args()

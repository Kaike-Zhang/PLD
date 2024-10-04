import argparse
import torch
import os

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser(description='RS Models')

parser.add_argument('--seed', type=int, default=2024, help='seed')

parser.add_argument('--model', type=str, default='MF', help='model')

# dataset
parser.add_argument('--dataset', type=str, default='MIND', help='dataset')
parser.add_argument('--min_interaction', type=int, default=10, help='Min interactions')
parser.add_argument('--noise', type=float, default=0.0, help='noise ratio')
parser.add_argument('--add_p', type=float, default=1.0, help='noise ratio')

# model
parser.add_argument('--out_dim', type=int, default=64, help='Output size of Adapter')

# experiment
parser.add_argument('--use_gpu', type=str2bool, default=True, help='training device')
parser.add_argument('--device', type=str, default='gpu', help='training device')
parser.add_argument('--device_id', type=str, default='4', help='device id for gpu')
parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
parser.add_argument('--test_batch_size', type=int, default=2048, help='Batch size')
parser.add_argument('--patience', type=int, default=100, help='patience for early stop')
parser.add_argument('--val_interval', type=int, default=20, help='Validation interval')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight for L2 loss on basic models.')
parser.add_argument('--min_epochs', type=int, default=40, help='min epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='max epoch')


parser.add_argument('--rec_top_k', type=list, default=[20, 50], help='K in evaluation')


parser.add_argument('--method', type=str, default='DeCA', help='presentage of injected user')

parser.add_argument('--temp', type=float, default=0.1, help='noise ratio')
parser.add_argument('--item_num', type=int, default=5, help='max epoch')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

if torch.cuda.is_available() and args.use_gpu:
    print('using gpu:{} to train the model'.format(args.device_id))
    args.device_id = list(range(torch.cuda.device_count()))
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")
    print('using cpu to train the model')
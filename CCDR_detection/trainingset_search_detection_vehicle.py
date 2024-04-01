import os
import feat_stas.CCDR_detection
import argparse
import numpy as np
#import torch
import random
#from scipy.special import softmax

# run in terminal
# python trainingset_search_detection_vehicle.py --target 'region100' --select_method 'SnP' --c_num 50 --result_dir 'main_results_mmd/sample_data_detection_vehicle_region100/' --n_num 8000 --output_data 'SnP_region100_vehicle_8000_random_c_num50_mmd.json'

# vit_l14_336px
# python trainingset_search_detection_vehicle.py --target 'region100' --select_method 'SnP' --c_num 50 --result_dir 'main_results_mmd_vit_l14_336px/sample_data_detection_vehicle_region100/' --n_num 8000 --output_data 'SnP_region100_vehicle_8000_random_c_num50_mmd_vit_l14_336px.json'


parser = argparse.ArgumentParser(description='outputs')
parser.add_argument('--logs_dir', type=str, metavar='PATH', default='sample_data/log.txt')
parser.add_argument('--result_dir', type=str, metavar='PATH', default='main_results/sample_data_detection/')
parser.add_argument('--use_camera', action='store_true', help='use use_camera')
parser.add_argument('--random_sample', action='store_true', help='random sample')
parser.add_argument('--c_num', default=50, type=int, help='number of cluster')
parser.add_argument('--select_method', type=str, default='CCDR', choices=['greedy', 'random', 'CCDR'], help='how to sample')
parser.add_argument('--n_num', default=8000, type=int, help='number of ids')
parser.add_argument('--no_sample', action='store_true', help='do not perform sample')
parser.add_argument('--cuda', action='store_true', help='whether cuda is enabled')
parser.add_argument('--target', type=str, default='region100', choices=['region100'], help='select which target')
parser.add_argument('--FD_model', type=str, default='inception', choices=['inception', 'posenet'],
                     help='model to calculate FD distance')
parser.add_argument('--output_data', type=str, metavar='PATH', default='CCDR_region100_vehicle_8000_random_c_num50.json')
parser.add_argument('--seed', default=0, type=int, help='random seed')

opt = parser.parse_args()
logs_dir=opt.logs_dir
result_dir=opt.result_dir

np.random.seed(opt.seed)
#torch.manual_seed(opt.seed)
random.seed(opt.seed) 

data_dict = {
        'ade': '/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/2024_challenge/data/source_pool/ade_train/',
        'bdd': '/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/2024_challenge/data/source_pool/bdd_train/',
        'cityscape': '/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/2024_challenge/data/source_pool/cityscapes_train/',
        'coco': '/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/2024_challenge/data/source_pool/coco_train/',
        'detrac': '/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/2024_challenge/data/source_pool/detrac_train/',
        'kitti': '/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/2024_challenge/data/source_pool/kitti_train/',
        'voc': '/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/2024_challenge/data/source_pool/voc_train/',
        'region100': '/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/2024_challenge/data/region_100/train/'
        }

annotation_dict = {
        'ade': '/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/2024_challenge/data/source_pool/ade_annotation.json',
        'bdd': '/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/2024_challenge/data/source_pool/bdd_annotation.json',
        'cityscape': '/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/2024_challenge/data/source_pool/cityscapes_annotation.json',
        'coco': '/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/2024_challenge/data/source_pool/coco_annotation.json',
        'detrac': '/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/2024_challenge/data/source_pool/detrac_annotation.json',
        'exdark': '/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/2024_challenge/data/source_pool/exdark_annotation.json',
        'kitti': '/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/2024_challenge/data/source_pool/kitti_annotation.json',
        'voc': '/media/jason/cc0aeb62-0bc7-4f3e-99a0-3bba3dd9f8fc/2024_challenge/data/source_pool/voc_annotation.json'
        }


databse_id= ['ade', 'bdd', 'cityscape', 'coco', 'detrac', 'kitti', 'voc']

if opt.target == 'region100':
    target = data_dict['region100'] 

result_dir=opt.result_dir
c_num = opt.c_num

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

if os.path.isdir(opt.output_data):
    assert ("output dir has already exist")

# sampled_data = feat_stas.SnP_detection_mmd_with_pruning.training_set_search(target, data_dict, annotation_dict, databse_id, opt, result_dir, c_num, version = "vehicle")
sampled_data = feat_stas.CCDR_detection.training_set_search(target, data_dict, annotation_dict, databse_id, opt, result_dir, c_num, version = "vehicle")



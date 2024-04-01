#!/usr/bin/env python3
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# import torchvision
# from k_means_constrained import KMeansConstrained
#
# import numpy as np
# import torch
# from scipy import linalg
# # from scipy.misc import imread
# from matplotlib.pyplot import imread, imsave
# from torch.nn.functional import adaptive_avg_pool2d, adaptive_max_pool2d
# from scipy import misc
# import random
# import re
from scipy.special import softmax
# from collections import defaultdict
# from shutil import copyfile
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from mmdet.datasets.coco_car import CocoCarDataset
import json
# import clip
# import collections
# from glob import glob
# import os.path as osp
# import h5py
# import scipy.io
# import threading
from PIL import Image
import copy
import pickle
import numpy as np
# from skimage.transform import resize
from sklearn.cluster import KMeans
# import time
# import xml.dom.minidom as XD

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from feat_stas.dataloader import get_detection_data_vehicle
from feat_stas.models.inception import InceptionV3
# from feat_stas.feat_extraction import get_activations, calculate_frechet_distance, calculate_activation_statistics
from feat_stas.feat_extraction import get_activations
from feat_stas.strategies import CoreSet
# from feat_stas.distance import mmd
# from feat_stas.distance_without_jax import mmd
from feat_stas.distance_pytorch import mmd
from clip import embedding
from clip.generate_clip_embeddings import compute_embeddings_for_dir

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=2,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='3', type=str,
                    help='GPU to use (leave blank for CPU only)')


def make_square(image, max_dim=512):
    max_dim = max(np.shape(image)[0], np.shape(image)[1])
    h, w = image.shape[:2]
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image


def training_set_search(tpaths, data_dict, annotation_dict, dataset_id, opt, result_dir, c_num, version):
    """clustering the ids from different datasets and sampleing"""

    if version == 'vehicle':
        img_paths, annotations, dataset_ids, meta_dataset = get_detection_data_vehicle(dataset_id, data_dict,
                                                                                       annotation_dict)

    print(len(img_paths), len(annotations), len(dataset_ids), len(meta_dataset))

    if opt.FD_model == 'inception':
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])

    cuda = True
    model.cuda()
    batch_size = 50

    print('=========== extracting feature of target training set ===========')
    embedding_model = embedding.ClipEmbeddingModel()
    files = []
    if version == 'vehicle':
        region_ids = []
        if opt.target == 'region100':
            images = []
            for root, dirs, files_dir in os.walk(data_dict['region100']):
                for file in files_dir:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        regionID = int(file.split("_")[0])
                        region_ids.append (regionID)
                        files.append(os.path.join(root, file))

    if not os.path.exists(result_dir + '/target_feature.npy'):
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)
        target_feature = compute_embeddings_for_dir(files, embedding_model=embedding_model, batch_size=8, max_count=-1)
        # target_feature = get_activations(opt, files, model, batch_size, dims, cuda, verbose=False)
        np.save(result_dir + '/target_feature.npy', target_feature)
    else:
        target_feature = np.load(result_dir + '/target_feature.npy')

    # extracter feature for data pool
    print('=========== extracting feature of data pool ===========')
    if not os.path.exists(result_dir + '/feature_infer.npy'):
        feature_infer = compute_embeddings_for_dir(img_paths, embedding_model=embedding_model, batch_size=8, max_count=-1)
        np.save(result_dir + '/feature_infer.npy', feature_infer)
    else:
        feature_infer = np.load(result_dir + '/feature_infer.npy')

    if not os.path.exists(result_dir + '/feature_infer_inception.npy'):
        feature_infer_inception = get_activations(opt, img_paths, model, batch_size, dims, cuda, verbose=False)
        np.save(result_dir + '/feature_infer_inception.npy', feature_infer_inception)
    else:
        feature_infer_inception = np.load(result_dir  + "/feature_infer_inception.npy")



    # clustering ids based on ids' mean feature
    if not os.path.exists(result_dir + '/label_cluster_' + str(c_num) + '_img.npy'):
        print('=========== clustering ===========')
        estimator = KMeans(n_clusters=c_num)
        estimator.fit(feature_infer)
        label_pred = estimator.labels_

        np.save(result_dir + '/label_cluster_' + str(c_num) + '_img.npy', label_pred)
    else:
        label_pred = np.load(result_dir + '/label_cluster_' + str(c_num) + '_img.npy')

    c_num = len(set(label_pred))

    print('number of clusters:', len(set(label_pred)))

    print('\r=========== calculating the mmd and v_gap between T and C_k ===========')
    if not os.path.exists(result_dir + '/cluster_mmd_by_' + str(c_num) + '_img.npy'):
        cluster_feature = []
        cluster_mmd = []
        for k in tqdm(range(c_num)):
            initial_feature_infer = feature_infer[label_pred == k]
            cluster_feature.append(initial_feature_infer)
            current_mmd = mmd(target_feature, initial_feature_infer)
            cluster_mmd.append(current_mmd)

        np.save(result_dir + '/cluster_mmd_by_' + str(c_num) + '_img.npy',
                np.array(cluster_mmd))
    else:
        cluster_mmd = np.load(result_dir + '/cluster_mmd_by_' + str(c_num) + '_img.npy')

    cluster_mmda = np.array(cluster_mmd)
    score_mmd = softmax(-cluster_mmda)
    sample_rate = score_mmd

    c_num_len = []
    id_score = []
    for kk in range(c_num):
        c_num_len_k = np.sum(label_pred == kk)
        c_num_len.append(c_num_len_k)

    for jj in range(len(label_pred)):
        id_score.append(sample_rate[label_pred[jj]] / c_num_len[label_pred[jj]])

    print(np.shape(id_score))
    # select a number index based on their FD score to the target domain
    print("select_method:", opt.select_method)
    if opt.select_method == 'random':
        selected_data_ind = np.sort(np.random.choice(range(len(id_score)), opt.n_num, replace=False))

    if opt.select_method == 'greedy':
        selected_data_ind = np.argsort(id_score)[-opt.n_num:]

    if opt.select_method == 'CCDR':
        lowest_fd = float('inf')
        lowest_img_list = []
        if not os.path.exists(result_dir + '/domain_seletive_' + str(c_num) + '_img.npy'):
            # cluster_rank = np.argsort(cluster_mmda)
            # current_list = []
            # cluster_feature_aggressive = []
            # for k in tqdm(cluster_rank):
            #     img_list = np.where(label_pred == k)[0]
            #     initial_feature_infer = feature_infer[label_pred == k]
            #     cluster_feature_aggressive.extend(initial_feature_infer)
            #     cluster_feature_aggressive_fixed = cluster_feature_aggressive
            #     target_feature_fixed = target_feature
            #     if len(cluster_feature_aggressive) > len(target_feature):
            #         cluster_idx = np.random.choice(range(len(cluster_feature_aggressive)), len(target_feature),
            #                                        replace=False)
            #         cluster_feature_aggressive_fixed = np.array([np.array(cluster_feature_aggressive[ii]) for ii in cluster_idx])
            #     if len(cluster_feature_aggressive) < len(target_feature):
            #         cluster_idx = np.random.choice(range(len(target_feature)), len(cluster_feature_aggressive),
            #                                        replace=False)
            #         target_feature_fixed = target_feature[cluster_idx]
            #
            #     current_mmd = mmd(target_feature, cluster_feature_aggressive_fixed)
            #     current_list.extend(list(img_list))
            #     print("cluster_feature_aggressive_fixed mmd:", current_mmd)
            #
            #     # current_faetures = feature_infer[lowest_img_list]
            #     # current_mmd = mmd(target_feature, current_faetures)
            #     # print("current_faetures mmd:", current_mmd)
            #     #
            #     # current_mmd = mmd(target_feature, cluster_feature_aggressive_fixed)
            #     if lowest_fd > current_mmd:
            #         lowest_fd = current_mmd
            #         lowest_img_list = copy.deepcopy(current_list)

            cluster_rank = np.argsort(cluster_mmda)
            current_list = []
            cluster_feature_aggressive = np.empty((0, feature_infer.shape[1]))

            for k in tqdm(cluster_rank):
                img_list = np.where(label_pred == k)[0]
                initial_feature_infer = feature_infer[label_pred == k]
                cluster_feature_aggressive = np.vstack((cluster_feature_aggressive, initial_feature_infer))
                cluster_feature_aggressive_fixed = cluster_feature_aggressive
                target_feature_fixed = target_feature

                if len(cluster_feature_aggressive) > len(target_feature):
                    cluster_idx = np.random.choice(np.arange(len(cluster_feature_aggressive)), len(target_feature),
                                                   replace=False)
                    cluster_feature_aggressive_fixed = cluster_feature_aggressive[cluster_idx]

                current_mmd = mmd(target_feature, cluster_feature_aggressive_fixed)
                current_list.extend(list(img_list))
                print("cluster_feature_aggressive_fixed mmd:", current_mmd)

                if lowest_fd > current_mmd:
                    lowest_fd = current_mmd
                    lowest_img_list = copy.deepcopy(current_list)
            np.save(result_dir + '/domain_seletive_' + str(c_num) + '_img.npy', lowest_img_list)
        else:
            lowest_img_list = np.load(result_dir + '/domain_seletive_' + str(c_num) + '_img.npy')

        selected_data_ind = lowest_img_list

    print("number of images after step 1:", len(selected_data_ind))

    feature_step1 = feature_infer[selected_data_ind]
    print("the shape of feature_step1:", np.shape(feature_step1))

    current_mmd = mmd(target_feature, feature_step1)
    print('finished with a dataset of searching(step 1) has MMD', current_mmd)


    # step 2 pruning

    feature_infer = np.load(result_dir + "feature_infer_inception.npy")
    from sklearn.preprocessing import normalize
    feature_infer = normalize(feature_infer, axis=1, norm='l2')

    set_ds = set()
    choose_set = []
    all_coco_indices = []
    for ind in selected_data_ind:
        choose_set.append(meta_dataset[ind][0])

    choose_set = [a.split("/")[-2] + "/" + a.split("/")[-1] for a in choose_set]
    choose_set = set(choose_set)
    choose_feats = []
    choose_feat_img_path = []

    all_coco_for_back_fill = []
    with open(result_dir + "scoring.pkl", "rb") as f:
        scores = pickle.load(f)
    scores_list = []

    for i in range(len(img_paths)):
        name = img_paths[i].split("/")[-2] + "/" + img_paths[i].split("/")[-1]
        if name in choose_set:
            all_coco_indices.append(i)
            set_ds.add(img_paths[i])
            #print(img_paths[i])
            choose_feats.append(feature_infer[i])
            choose_feat_img_path.append(img_paths[i])
            parts = img_paths[i].split("/")
            score_name = [parts[-1]]
            p = -2
            while "_train" not in parts[p]:
                score_name.append(parts[p])
                p -= 1
            score_name.append(parts[p])
            score_name = score_name[::-1]
            score_name = "/".join(score_name)
            scores_list.append(scores[score_name])
        if "coco" in img_paths[i]:
            all_coco_for_back_fill.append(img_paths[i])
    all_coco_indices = set(all_coco_indices)
    choose_feats = np.array(choose_feats)

    print("total before graph: {}".format(len(set_ds)))
    coco_source_feats = []
    all_coco_indices = list(all_coco_indices)
    all_coco_indices.sort()
    for idx in all_coco_indices:
        coco_source_feats.append(feature_infer[idx])
    coco_source_feats = np.array(coco_source_feats)
    coco_res = np.matmul(coco_source_feats, coco_source_feats.T)
    for i in range(len(coco_res)):
        coco_res[i,i] = 0


    size_before_removing = len(set_ds)
    threshold = 0.8183
    to_keep = set()
    removed = set()
    #o = open("sim_pairs.txt", "w")
    while len(set_ds) - len(removed) > 8000:
        sim_pair = (coco_res>threshold).nonzero()
        to_keep = set()
        removed = set()
        for i in tqdm(range(len(sim_pair[0]))):
            a = sim_pair[0][i]
            b = sim_pair[1][i]
            if scores_list[a] > scores_list[b] and img_paths[all_coco_indices[b]] not in to_keep:
                if img_paths[all_coco_indices[b]] in set_ds:
                    set_ds.remove(img_paths[all_coco_indices[b]])
                    to_keep.add(img_paths[all_coco_indices[a]])
            elif img_paths[all_coco_indices[a]] not in to_keep:
                if img_paths[all_coco_indices[a]] in set_ds:
                    set_ds.remove(img_paths[all_coco_indices[a]])
                    to_keep.add(img_paths[all_coco_indices[b]])

        print("orig size:{} current threshold: {}, number left: {}".format(size_before_removing, threshold, len(set_ds) - len(removed)))
        if len(set_ds) - len(removed) > 8000:
            threshold -= 0.0001

    for a in removed:
        set_ds.remove(a)
    print(str(size_before_removing) + " keeps" + str(len(set_ds)))


    i = 0
    while len(set_ds) < 8000:
        if all_coco_for_back_fill[i] not in set_ds:
            set_ds.add(all_coco_for_back_fill[i])
        i += 1

    final_selected_img_ind = []
    for img in set_ds:
        final_selected_img_ind.append(img_paths.index(img))
    json_generate(final_selected_img_ind, meta_dataset, opt)

    # return selected_data_ind
    return final_selected_img_ind


INFO = {
    "description": "Cityscapes_Instance Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": "2020",
    "contributor": "Kevin_Jia",
    "date_created": "2020-1-23 19:19:19.123456"
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'car',
        'supercategory': 'car',
    }, ]

from pycococreatortools import pycococreatortools


def json_generate(selected_data_ind, meta_dataset, opt):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    annotation_id = 0
    for image_id, idx in enumerate(selected_data_ind):
        image_path, anno, dataset_id = meta_dataset[idx]
        image = Image.open(image_path)
        image_info = pycococreatortools.create_image_info(
            image_id, image_path, image.size)
        coco_output["images"].append(image_info)

        for annotation_info in anno:
            annotation_info['image_id'] = image_id
            annotation_info['id'] = annotation_id
            annotation_id = annotation_id + 1
            coco_output["annotations"].append(annotation_info)

    with open(opt.output_data, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)



def json_generate_new(selected_data_ind, meta_dataset, path):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    annotation_id = 0
    for image_id, idx in enumerate(selected_data_ind):
        image_path, anno, dataset_id = meta_dataset[idx]
        image = Image.open(image_path)
        image_info = pycococreatortools.create_image_info(
            image_id, image_path, image.size)
        coco_output["images"].append(image_info)

        for annotation_info in anno:
            annotation_info['image_id'] = image_id
            annotation_info['id'] = annotation_id
            annotation_id = annotation_id + 1
            coco_output["annotations"].append(annotation_info)

    with open(path, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)



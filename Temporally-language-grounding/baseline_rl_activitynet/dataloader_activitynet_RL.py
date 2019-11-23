" Dataloader of ActivityNet dataset for Reinforcenments Learning based methods"

import torch
import torch.utils.data
import os
import pickle
import numpy as np
import math
from utils import *
import random
import glob
import h5py
import json

'''
v_---9CpRcKoU
duration: 14.07
51x500
{'annotations': [{'label': 'Drinking beer',
                  'segment': [0.01, 12.644405017160688]}],
 'duration': 14.07,
 'resolution': '320x240',
 'subset': 'training',
 'url': 'https://www.youtube.com/watch?v=---9CpRcKoU'}
'''

class Activitynet_Train_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path = '/home/share/hanxianjing/activityNet/'):
        self.data_path = data_path
        self.visual_feats_dimen = 500
        self.sentence_size = 20
        self.word_embedding_size = 300
        self.c3d_features = os.path.join(self.data_path, "sub_activitynet_v1-3.c3d.hdf5")
        self.clip_sentence_pairs_iou_all = pickle.load(open(os.path.join(self.data_path, "activitynet_rl_train_feature_all_glove_embedding_final.pkl"), 'rb'))
        vocab = pickle.load(open(os.path.join(self.data_path, "activitynet_rl_train_vocab.pkl"), 'rb'))
        self.vocab, self.embeddings = vocab['words'], vocab['embeddings']

        self.num_samples_iou = len(self.clip_sentence_pairs_iou_all)
        print((self.num_samples_iou, "iou clip-sentence pairs are readed"))  # 49442

    def word2idx(self, word):
        if word in self.vocab:
            return self.vocab.index(word)
        else:
            return 2  # 2为<unk>的下标

    def read_video_level_feats(self, movie_name):
        # read unit level feats by just passing the start and end number
        c3d_features = h5py.File(self.c3d_features,'r')
        original_feats = c3d_features[movie_name]['c3d_features'][:]
        num_units = original_feats.shape[0]
        ten_unit = int(num_units / 10)
        four_unit = int(num_units / 4)
        oneinfour_unit = four_unit
        threeinfour_unit = num_units - four_unit

        global_feature = np.mean(original_feats, axis=0)

        # 选取[N/4,3N/4]作为初始边界并提取特征local_feature(取边界内所有unit特征平均)
        initial_feature = original_feats[(oneinfour_unit - 1):(threeinfour_unit)]
        initial_feature = np.mean(initial_feature, axis=0)

        initial_offset_start = oneinfour_unit - 1
        initial_offset_end = threeinfour_unit - 1

        initial_offset_start_norm = initial_offset_start / float(num_units - 1)
        initial_offset_end_norm = initial_offset_end / float(num_units - 1)

        return global_feature, original_feats, initial_feature, ten_unit, initial_offset_start, initial_offset_end, initial_offset_start_norm, initial_offset_end_norm, num_units

    def __getitem__(self, index):
        # print(index)
        feats_dim = 500
        offset = np.zeros(2, dtype=np.float32)
        offset_norm = np.zeros(2, dtype=np.float32)
        initial_offset = np.zeros(2, dtype=np.float32)
        initial_offset_norm = np.zeros(2, dtype=np.float32)

        samples = self.clip_sentence_pairs_iou_all[index]

        # end = samples['frames_num'] + 1
        movie_name = samples['video']

        global_feature, original_feats, initial_feature, ten_unit, initial_offset_start, initial_offset_end, initial_offset_start_norm, initial_offset_end_norm, num_units \
            = self.read_video_level_feats(movie_name)

        # print(np.shape(global_feature), np.shape(original_feats), np.shape(initial_feature))

        sentence = samples['sent_skip_thought_vec'][0][0]
        offset_start = samples['offset_start']
        offset_end = samples['offset_end']
        offset_start_norm = samples['offset_start_norm']
        offset_end_norm = samples['offset_end_norm']

        # offest
        offset[0] = offset_start
        offset[1] = offset_end

        offset_norm[0] = offset_start_norm
        offset_norm[1] = offset_end_norm

        initial_offset[0] = initial_offset_start
        initial_offset[1] = initial_offset_end

        initial_offset_norm[0] = initial_offset_start_norm
        initial_offset_norm[1] = initial_offset_end_norm

        original_feats_padding = np.zeros((3700, feats_dim), dtype=np.float32)
        original_feats_padding[:original_feats.shape[0]] = original_feats
        return global_feature, original_feats_padding, initial_feature, sentence, offset_norm, initial_offset, initial_offset_norm, ten_unit, num_units

    def __len__(self):
        return self.num_samples_iou


class Activitynet_Test_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path = '/home/share/hanxianjing/activityNet/'):
        self.data_path = data_path
        self.load_clip_dict = {}
        self.visual_feature_dim = 500
        self.sentence_size = 20
        self.word_embedding_size = 300
        self.c3d_features = os.path.join(self.data_path, "sub_activitynet_v1-3.c3d.hdf5")
        # self.sliding_clip_path = os.path.join(self.data_path, "all_fc6_unit16_overlap0.5")
        self.clip_sentence_pairs = pickle.load(open(os.path.join(self.data_path, "activitynet_rl_test_feature_all_glove_embedding_final.pkl"), 'rb'))
        vocab = pickle.load(open(os.path.join(self.data_path, "activitynet_rl_train_vocab.pkl"), 'rb'))
        self.vocab, self.embeddings = vocab['words'], vocab['embeddings']

        with open(os.path.join(self.data_path,'activity_net.v1-3.min.json'), 'r') as f:
            self.duration = json.load(f)['database']

        movie_names_set = set()
        for ii in self.clip_sentence_pairs:
            for iii in self.clip_sentence_pairs[ii]:
                clip_name = iii
                movie_name = ii
                if not movie_name in movie_names_set:
                    movie_names_set.add(movie_name)
        self.movie_names = list(movie_names_set)

        print(str(len(self.clip_sentence_pairs)) + " test videos are readed")

    def word2idx(self, word):
        if word in self.vocab:
            return self.vocab.index(word)
        else:
            return 2  # 2为<unk>的下标

    def read_video_level_feats(self, movie_name):
        # read unit level feats by just passing the start and end number
        c3d_features = h5py.File(self.c3d_features,'r')
        original_feats = c3d_features[movie_name]['c3d_features'][:]
        num_units = original_feats.shape[0]
        ten_unit = int(num_units / 10)
        four_unit = int(num_units / 4)
        oneinfour_unit = four_unit
        threeinfour_unit = num_units - four_unit

        global_feature = np.mean(original_feats, axis=0)

        # 选取[N/4,3N/4]作为初始边界并提取特征local_feature(取边界内所有unit特征平均)
        initial_feature = original_feats[(oneinfour_unit - 1):(threeinfour_unit)]
        initial_feature = np.mean(initial_feature, axis=0)

        initial_offset_start = oneinfour_unit - 1
        initial_offset_end = threeinfour_unit - 1

        initial_offset_start_norm = initial_offset_start / float(num_units - 1)
        initial_offset_end_norm = initial_offset_end / float(num_units - 1)

        return global_feature, original_feats, initial_feature, ten_unit, initial_offset_start, initial_offset_end, initial_offset_start_norm, initial_offset_end_norm, num_units

    def load_movie_slidingclip(self, movie_name):
        # load unit level feats and sentence vector
        initial_offset = np.zeros(2, dtype=np.float32)
        initial_offset_norm = np.zeros(2, dtype=np.float32)
        movie_clip_sentences = []

        global_feature, original_feats, initial_feature, ten_unit, initial_offset_start, initial_offset_end, initial_offset_start_norm, initial_offset_end_norm, num_units \
            = self.read_video_level_feats(movie_name)
        ten_unit = np.array(ten_unit)
        num_units = np.array(num_units)
        duration = self.duration[movie_name[2:]]['duration']
        # movie_clip_sentences:(FeatureFileName(clip),sent_skip_thought_vec(sentence))
        for dict_2nd in self.clip_sentence_pairs[movie_name]:
            for dict_3rd in self.clip_sentence_pairs[movie_name][dict_2nd]:
                token_embeddings = np.concatenate((self.embeddings[0][np.newaxis, :], dict_3rd['glove_embeddings']),
                                                  axis=0)
                sentence_vec_ = dict_3rd['sent_skip_thought_vec'][0][0]
                offset_norm
                movie_clip_sentences.append((dict_2nd, sentence_vec_))

        initial_offset[0] = initial_offset_start
        initial_offset[1] = initial_offset_end

        initial_offset_norm[0] = initial_offset_start_norm
        initial_offset_norm[1] = initial_offset_end_norm

        return movie_clip_sentences, global_feature, original_feats, initial_feature, initial_offset, initial_offset_norm, ten_unit, num_units

if __name__ == '__main__':
    # tr = Activitynet_Train_dataset()
    # trainloader = torch.utils.data.DataLoader(dataset=tr,
    #                                           batch_size=32,
    #                                           shuffle=True,
    #                                           num_workers=4,
    #                                           pin_memory=True)
    # for i in trainloader:
    #     pass
    # print('train pass')
    te = Activitynet_Test_dataset()
    for movie_name in te.movie_names:
        print(movie_name)
        temp = te.load_movie_slidingclip(movie_name)
    print('test pass')
" Dataloader of ActivityNet DenseCaption dataset for Supervised Learning based methods"

import torch
import torch.utils.data
import os
import pickle
import numpy as np
import math
from utils import *
import random
import h5py
import json

class Activitynet_Train_dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.feats_dimen = 500
        self.context_num = 1
        self.context_size = 128
        self.sent_vec_dim = 4800
        self.data_path = '/home/share/hanxianjing/activityNet'
        self.proposals = os.path.join(self.data_path, "activitynet_v1-3_proposals.hdf5")
        self.c3d_features = os.path.join(self.data_path, "sub_activitynet_v1-3.c3d.hdf5")
        self.clip_sentence_pairs_iou_all = pickle.load(open(os.path.join(self.data_path, "activitynet_rl_train_feature_all_glove_embedding_final.pkl"), 'rb'))

        self.num_samples_iou = len(self.clip_sentence_pairs_iou_all)
        print((self.num_samples_iou, "iou clip-sentence pairs are readed"))  # 49442

    def correct_win(self, start, end, max_row):
        start = max(0, start)
        end = min(max_row, end)
        if start > end:
            start, end = end, start
        if start == end:
            if start == 0:
                end += 1
            if start == max_row:
                start -= 1
            else:
                end += 1
        return start, end

    def read_unit_level_feats(self, movie_name, start_norm, end_norm):
        # read unit level feats by just passing the start and end number

        c3d_features = h5py.File(self.c3d_features,'r')
        original_feats = c3d_features[movie_name]['c3d_features'][:]
        total_row = original_feats.shape[0]
        start_row = int(total_row*start_norm)
        end_row = int(total_row*end_norm)
        start_row, end_row = self.correct_win(start_row, end_row, total_row)
        original_feats_1 = np.mean(original_feats[start_row: end_row], axis=0)

        left_start_row = int(start_row - total_row / 5)
        left_end_row = start_row
        left_start_row, left_end_row = self.correct_win(left_start_row, left_end_row, total_row)
        if left_start_row >= left_end_row:
            print('error1')
        left_context_feat = np.mean(original_feats[left_start_row: left_end_row], axis=0)

        right_end_row = int(end_row + total_row / 5)
        right_start_row = end_row
        right_start_row, right_end_row = self.correct_win(right_start_row, right_end_row, total_row)
        if right_start_row >= right_end_row:
            print('error2')
        right_context_feat = np.mean(original_feats[right_start_row: right_end_row], axis=0)

        return left_context_feat, original_feats_1, right_context_feat

    def __getitem__(self, index):

        offset = np.zeros(2, dtype=np.float32)

        # get this clip's: sentence  vector, swin, p_offest, l_offset, sentence, Vps
        sample = self.clip_sentence_pairs_iou_all[index]
        proposal = h5py.File(self.proposals, 'r')[sample['video']]
        start, end = proposal['segment-init'][0], proposal['segment-end'][0]
        start_norm, end_norm = start / sample['duration'], end / sample['duration']
        # read visual feats
        left_context_feat, featmap, right_context_feat = self.read_unit_level_feats(sample['video'], start_norm, end_norm)
        image = np.hstack((left_context_feat, featmap, right_context_feat))
        # sentence batch
        sentence = sample['sent_skip_thought_vec'][0][0, :self.sent_vec_dim]

        # offest
        p_offset = sample['offset_start']
        l_offset = sample['offset_end']
        offset[0] = p_offset
        offset[1] = l_offset

        return image, sentence, offset

    def __len__(self):
        return self.num_samples_iou


class Activitynet_Test_dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data_path = '/home/share/hanxianjing/activityNet'
        self.feats_dimen = 500
        self.semantic_size = 4800
        self.index_in_epoch = 0
        self.sent_vec_dim = 4800
        self.epochs_completed = 0
        self.proposals = os.path.join(self.data_path, "activitynet_v1-3_proposals.hdf5")
        self.c3d_features = os.path.join(self.data_path, "sub_activitynet_v1-3.c3d.hdf5")
        self.clip_sentence_pairs = pickle.load(open(os.path.join(self.data_path, "activitynet_rl_test_feature_all_glove_embedding_final.pkl"),"rb"), encoding='iso-8859-1')
        with open(os.path.join(self.data_path,'activity_net.v1-3.min.json'), 'r') as f:
            self.duration = json.load(f)['database']

        print(str(len(self.clip_sentence_pairs)) + " test videos are readed")  # 1334

        movie_names_set = set()
        for ii in self.clip_sentence_pairs:
            for iii in self.clip_sentence_pairs[ii]:
                movie_name = ii
                if not movie_name in movie_names_set:
                    movie_names_set.add(movie_name)
        self.movie_names = list(movie_names_set)

    def correct_win(self, start, end, max_row):
        start = max(0, start)
        end = min(max_row, end)
        if start > end:
            start, end = end, start
        if start == end:
            if start == 0:
                end += 1
            if start == max_row:
                start -= 1
            else:
                end += 1
        return start, end

    def read_unit_level_feats(self, movie_name, start_norm, end_norm):
        # read unit level feats by just passing the start and end number

        c3d_features = h5py.File(self.c3d_features, 'r')
        original_feats = c3d_features[movie_name]['c3d_features'][:]
        total_row = original_feats.shape[0]
        start_row = int(total_row * start_norm)
        end_row = int(total_row * end_norm)
        start_row, end_row = self.correct_win(start_row, end_row, total_row)
        original_feats_1 = np.mean(original_feats[start_row: end_row], axis=0)

        left_start_row = int(start_row - total_row / 5)
        left_end_row = start_row
        left_start_row, left_end_row = self.correct_win(left_start_row, left_end_row, total_row)
        if left_start_row >= left_end_row:
            print('error1')
        left_context_feat = np.mean(original_feats[left_start_row: left_end_row], axis=0)

        right_end_row = int(end_row + total_row / 5)
        right_start_row = end_row
        right_start_row, right_end_row = self.correct_win(right_start_row, right_end_row, total_row)
        if right_start_row >= right_end_row:
            print('error2')
        right_context_feat = np.mean(original_feats[right_start_row: right_end_row], axis=0)

        return left_context_feat, original_feats_1, right_context_feat


    def load_movie_slidingclip(self, movie_name):
        # load unit level feats and sentence vector
        movie_clip_sentences = []
        movie_clip_featmap = []

        for dict_2nd in self.clip_sentence_pairs[movie_name]:
            for dict_3rd in self.clip_sentence_pairs[movie_name][dict_2nd]:
                sentence_vec_ = dict_3rd['sent_skip_thought_vec'][0][0, :self.sent_vec_dim]
                movie_clip_sentences.append((dict_2nd, sentence_vec_))
                duration = self.duration[movie_name[2:]]['duration']
                proposal = h5py.File(self.proposals, 'r')[movie_name]
                start, end = proposal['segment-init'][0], proposal['segment-end'][0]
                start_norm, end_norm = start / duration, end / duration
                # read visual feats
                left_context_feat, featmap, right_context_feat = self.read_unit_level_feats(movie_name, start_norm, end_norm)
                image = np.hstack((left_context_feat, featmap, right_context_feat))
                movie_clip_featmap.append((movie_name+'_'+str(start)+'_'+str(end), image))

        return movie_clip_featmap, movie_clip_sentences

if __name__ == "__main__":
    a = Activitynet_Test_dataset()
    for i in range(len(a.movie_names)):
        a.load_movie_slidingclip(a.movie_names[i])
        print(i)
    # a = Activitynet_Train_dataset()
    # print(a.__getitem__(0))
    # trainloader = torch.utils.data.DataLoader(dataset=a,
    #                                           batch_size=32,
    #                                           shuffle=True,
    #                                           num_workers=4)
    # for idx, i in enumerate(trainloader):
    #     print(idx)
    # print('complete')

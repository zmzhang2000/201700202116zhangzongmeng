" Dataloader of charades-STA dataset for Reinforcenments Learning based methods"

import torch
import torch.utils.data
import os
import pickle
import numpy as np
import math
from utils import *
import random
import glob


class Charades_Train_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.unit_size = 16
        self.feats_dimen = 4096
        self.context_num = 1
        self.context_size = 128
        self.visual_feature_dim = 4096
        self.sentence_size = 10
        self.word_embedding_size = 300

        self.sliding_clip_path = os.path.join(self.data_path, "all_fc6_unit16_overlap0.5")
        '''
        all_fc6_unit16_overlap0.5/每个视频的滑动窗口特征(numpy数组，每个滑动窗口作为一个unit提取特征)
        MovieName_StartFrame.0_EndFrame.0.npy"
        窗口大小为unit_size=16
        滑动步长为8
        帧编号从1开始
        '''

        self.clip_sentence_pairs_iou_all = pickle.load(
            open(os.path.join(self.data_path, "charades_rl_train_feature_all_glove_embedding.pkl"), 'rb'), encoding='iso-8859-1')
        self.clip_sentence_pairs_iou_all = self.clip_sentence_pairs_iou_all[:int(len(self.clip_sentence_pairs_iou_all)/2)]
        ''' 
        charades_rl_train_feature_all: [item0, item1...]
        
        item: 
        {'frames_num': 496,
         'num_units': 31,
         'offset_end': 101,
         'offset_end_norm': 0.20362903225806453,
         'offset_start': 0,
         'offset_start_norm': 0.0,
         'proposal_or_sliding_window': 'AO8RW_1_129',
         'sent_skip_thought_vec': [array([[-0.00339404,  0.00549818, -0.00043744, ...,  0.08359446,
                 0.00238644,  0.01462354]], dtype=float32)], 
         'tokens': ['person', 'is', 'putting', 'book', 'on', 'shelf'] (length <= 10, 去掉a an the . , #)
         'glove_embeddings': numpy.array(shape:[10,300])(单个embedding长300，句子长10，不够用0向量补齐)
         'sentence': 'a person is putting a book on a shelf.',
         'video': 'AO8RW'}
         
         item['sent_skip_thought_vec']: [numpy.array(shape:[1,4800])]
        '''

        vocab = pickle.load(open(os.path.join(self.data_path, "charades_rl_train_vocab.pkl"), 'rb'))
        self.vocab, self.embeddings = vocab['words'], vocab['embeddings']
        '''
        vocab: 训练集所有词组成的元组
        embeddings: 第i行对应vocab[i]的embedding  vocab_size x 300
        '''

        self.num_samples_iou = len(self.clip_sentence_pairs_iou_all)
        print((self.num_samples_iou, "iou clip-sentence pairs are readed"))  # 49442

    def word2idx(self, word):
        if word in self.vocab:
            return self.vocab.index(word)
        else:
            return 2    # 2为<unk>的下标

    def read_video_level_feats(self, movie_name, end):
        # read unit level feats by just passing the start and end number
        # 将视频切割为unit提取特征
        unit_size = 16
        feats_dimen = 4096
        start = 1
        num_units = int((end - start) / unit_size)
        # print(start, end, num_units)
        curr_start = 1

        ten_unit = int(num_units / 10)
        four_unit = int(num_units / 4)
        oneinfour_unit = four_unit
        threeinfour_unit = num_units - four_unit

        start_end_list = []
        while (curr_start + unit_size <= end):
            start_end_list.append((curr_start, curr_start + unit_size))
            curr_start += unit_size

        # 数据集中最长3944帧 < 300x16
        original_feats = np.zeros([300, feats_dimen], dtype=np.float32)
        original_feats_1 = np.zeros([int(num_units), feats_dimen], dtype=np.float32)
        for k, (curr_s, curr_e) in enumerate(start_end_list):
            one_feat = np.load(os.path.join(self.sliding_clip_path, movie_name + "_" + str(curr_s) + ".0_" + str(curr_e) + ".0.npy"))
            original_feats[k] = one_feat
            original_feats_1[k] = one_feat

        # print(np.shape(original_feats))
        global_feature = np.mean(original_feats_1, axis=0)

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
        offset = np.zeros(2, dtype=np.float32)
        offset_norm = np.zeros(2, dtype=np.float32)
        initial_offset = np.zeros(2, dtype=np.float32)
        initial_offset_norm = np.zeros(2, dtype=np.float32)

        samples = self.clip_sentence_pairs_iou_all[index]

        proposal_or_sliding_window = samples['proposal_or_sliding_window']
        end = samples['frames_num'] + 1

        movie_name = proposal_or_sliding_window.split("_")[0]

        global_feature, original_feats, initial_feature, ten_unit, initial_offset_start, initial_offset_end, initial_offset_start_norm, initial_offset_end_norm, num_units \
            = self.read_video_level_feats(movie_name, end)

        # print(np.shape(global_feature), np.shape(original_feats), np.shape(initial_feature))

        token_embeddings = np.concatenate((self.embeddings[0][np.newaxis, :], samples['glove_embeddings']), axis=0)
        target = np.zeros((self.sentence_size+1, len(self.vocab)))
        for row,col in enumerate([self.word2idx(word) for word in samples['tokens']] + [1]):
            target[row, col] = 1

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

        return global_feature, original_feats, initial_feature, token_embeddings, target, offset_norm, initial_offset, initial_offset_norm, ten_unit, num_units

    def __len__(self):
        return self.num_samples_iou


class Charades_Test_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        # il_path: image_label_file path
        self.load_clip_dict = {}
        self.context_num = 1
        self.context_size = 128
        self.visual_feature_dim = 4096
        self.feats_dimen = 4096
        self.unit_size = 16
        self.semantic_size = 4800
        self.sliding_clip_path = os.path.join(self.data_path, "all_fc6_unit16_overlap0.5")
        self.index_in_epoch = 0
        self.spacy_vec_dim = 300
        self.sentence_size = 10
        self.word_embedding_size = 300
        self.epochs_completed = 0

        self.clip_sentence_pairs = pickle.load(
            open(os.path.join(self.data_path, "ref_info/charades_sta_test_semantic_sentence_VP_sub_obj_glove_embedding.pkl"), 'rb'),
            encoding='iso-8859-1')
        print(str(len(self.clip_sentence_pairs)) + " test videos are readed")  # 1334
        '''
        charades_sta_test_semantic_sentence_VP_sub_obj.pkl: {MovieName:{FeatureFileName:[feature]}}
        feature:
        {'VP_skip_thought_vec': [],
         'VP_spacy_vec_one_by_one_word': [],
         'dobj_or_VP': [],
         'obj': [],
         'obj_spacy_vec': [],
         'sent_skip_thought_vec': [numpy.array(shape:[1,4800],dtype=float32)],
         'sent_spacy_vec': [numpy.array(shape:[300,],dtype=float32)],
         'sentence': 'person close the door.',
         'tokens': ['person', 'close', 'door'] (length <= 10, 去掉a an the . , #)
         'glove_embeddings': numpy.array(shape:[10,300])(单个embedding长300，句子长10，不够用0向量补齐)
         'subj': [],
         'subj_spacy_vec': []}
         
         每个MovieName可能有多个FeatureFileName，每个FeatureFileName可能有多个feature
         每个movie可能有多个clip，每个clip又可能有多个description
        '''

        vocab = pickle.load(open(os.path.join(self.data_path, "charades_rl_train_vocab.pkl"), 'rb'))
        self.vocab, self.embeddings = vocab['words'], vocab['embeddings']
        '''
        vocab: 训练集所有词组成的元组
        embeddings: 第i行对应vocab[i]的embedding
        '''

        movie_names_set = set()
        for ii in self.clip_sentence_pairs:
            for iii in self.clip_sentence_pairs[ii]:
                clip_name = iii
                movie_name = ii
                if not movie_name in movie_names_set:
                    movie_names_set.add(movie_name)
        self.movie_names = list(movie_names_set)

        self.movie_length_dict = {}
        with open(os.path.join(self.data_path, "ref_info/charades_movie_length_info.txt"))  as f:
            for l in f:
                self.movie_length_dict[l.rstrip().split(" ")[0]] = float(l.rstrip().split(" ")[1])
        '''charades_movie_length_info.txt:
        46GP8 24.92 597
        N11GT 18.44 459
        0IH69 30.33 757
        ......
        '''

    def word2idx(self, word):
        if word in self.vocab:
            return self.vocab.index(word)
        else:
            return 2    # 2为<unk>的下标

    def read_video_level_feats(self, movie_name, end):
        # read unit level feats by just passing the start and end number
        unit_size = 16
        feats_dimen = 4096
        start = 1
        num_units = int((end - start) / unit_size)
        # print(start, end, num_units)
        curr_start = 1

        ten_unit = int(num_units / 10)
        four_unit = int(num_units / 4)
        oneinfour_unit = four_unit
        threeinfour_unit = num_units - four_unit

        start_end_list = []
        while (curr_start + unit_size <= end):
            start_end_list.append((curr_start, curr_start + unit_size))
            curr_start += unit_size

        # original_feats = np.zeros([num_units, feats_dimen], dtype=np.float32)
        original_feats = np.zeros([num_units, feats_dimen], dtype=np.float32)
        for k, (curr_s, curr_e) in enumerate(start_end_list):
            one_feat = np.load(os.path.join(self.sliding_clip_path, movie_name + "_" + str(curr_s) + ".0_" + str(curr_e) + ".0.npy"))
            original_feats[k] = one_feat
        # print(np.shape(original_feats))
        global_feature = np.mean(original_feats, axis=0)

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
        # 以已提取视频特征的最大帧编号作为视频结束点
        checkpoint_paths = glob.glob(os.path.join(self.sliding_clip_path,movie_name + "_*"))
        checkpoint_file_name_ints = [int(float(x.split('/')[-1].split('.npy')[0].split('_')[-1]))
                                     for x in checkpoint_paths]
        end = max(checkpoint_file_name_ints)
        global_feature, original_feats, initial_feature, ten_unit, initial_offset_start, initial_offset_end, initial_offset_start_norm, initial_offset_end_norm, num_units \
            = self.read_video_level_feats(movie_name, end)
        ten_unit = np.array(ten_unit)
        num_units = np.array(num_units)

        # movie_clip_sentences:(FeatureFileName(clip),sent_skip_thought_vec(sentence))
        for dict_2nd in self.clip_sentence_pairs[movie_name]:
            for dict_3rd in self.clip_sentence_pairs[movie_name][dict_2nd]:
                token_embeddings = np.concatenate((self.embeddings[0][np.newaxis, :], dict_3rd['glove_embeddings']), axis=0)
                target = np.zeros((self.sentence_size + 1, len(self.vocab)))
                for row, col in enumerate([self.word2idx(word) for word in dict_3rd['tokens']] + [1]):
                    target[row, col] = 1
                movie_clip_sentences.append((dict_2nd, token_embeddings, target))

        initial_offset[0] = initial_offset_start
        initial_offset[1] = initial_offset_end

        initial_offset_norm[0] = initial_offset_start_norm
        initial_offset_norm[1] = initial_offset_end_norm

        return movie_clip_sentences, global_feature, original_feats, initial_feature, initial_offset, initial_offset_norm, ten_unit, num_units

    def multi_process_load_clip(self, chunk:int):
        try:
            for movie_name in self.movie_names:
                yield self.load_clip_dict[movie_name]
        except Exception as e:
            from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
            with ProcessPoolExecutor() as pool:
                for i in range(0, len(self.movie_names), chunk):
                    for i, movie_name in zip([
                        pool.submit(self.load_movie_slidingclip, movie_name)
                        for movie_name in self.movie_names[i:i+chunk]
                    ], self.movie_names[i:i+chunk]):
                        data = i.result()
                        self.load_clip_dict[movie_name] = data
                        yield data
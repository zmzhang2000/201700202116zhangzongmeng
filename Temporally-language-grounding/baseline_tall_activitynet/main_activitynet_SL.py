"""
Train and test file for Supervised Learning based methods (TALL) for ActivityNet DenseCaption dataset \
TALL: Temporal Activity Localization via Language Query(http://openaccess.thecvf.com/content_ICCV_2017/papers/Gao_TALL_Temporal_Activity_ICCV_2017_paper.pdf) \
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import numpy as np
import os
import argparse
from utils import *
import random
from torch.autograd import Variable
from dataloader_activitynet_SL import Activitynet_Train_dataset, Activitynet_Test_dataset
from model_TALL import TALL

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
parser = argparse.ArgumentParser(description='Video Grounding of PyTorch')
parser.add_argument('--dataset', type=str, default='Charades', help='dataset type')
parser.add_argument('--batch_size', default=56, type=int, help='batch size')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
opt = parser.parse_args()

path = os.path.join(opt.dataset + '_TALL')
if not os.path.exists(path):
    os.makedirs(path)

train_dataset = Activitynet_Train_dataset()
test_dataset = Activitynet_Test_dataset()

num_train_batches = int(len(train_dataset)/opt.batch_size)
print ("num_train_batches:", num_train_batches)

trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=opt.batch_size,
                                           shuffle=True,
                                           num_workers=4)

# Model
net = TALL().cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

setup_seed(0)
best_R1_IOU5 = 0
best_R5_IOU5 = 0
best_R1_IOU5_epoch = 0
best_R5_IOU5_epoch = 0

# Training
def train(epoch):
    net.train()
    train_loss = 0

    for batch_idx, (images, sentences, offsets) in enumerate(trainloader):
        images, sentences, offsets = images.float().cuda(), sentences.float().cuda(), offsets.float().cuda()

        # network forward
        outputs = net(images, sentences)
        # compute alignment and regression loss
        sim_score_mat = outputs[0]
        p_reg_mat = outputs[1]
        l_reg_mat = outputs[2]
        # loss cls, not considering iou
        input_size = outputs.size(1)
        I = torch.eye(input_size).cuda()
        I_2 = -2 * I
        all1 = torch.ones(input_size, input_size).cuda()

        mask_mat = I_2 + all1  # 56,56

        #               | -1  1   1...   |
        #   mask_mat =  | 1  -1   1...   |
        #               | 1   1  -1 ...  |

        alpha = 1.0 / input_size
        lambda_regression = 1
        batch_para_mat = alpha * all1
        para_mat = I + batch_para_mat

        loss_mat = torch.log(all1 + torch.exp(mask_mat*sim_score_mat))
        loss_mat = loss_mat*para_mat
        loss_align = loss_mat.mean()

        # regression loss
        l_reg_diag = torch.mm(l_reg_mat*I, torch.ones(input_size, 1).cuda())
        p_reg_diag = torch.mm(p_reg_mat*I, torch.ones(input_size, 1).cuda())
        offset_pred = torch.cat([p_reg_diag, l_reg_diag], 1)
        loss_reg = torch.abs(offset_pred - offsets).mean() # L1 loss

        loss= lambda_regression*loss_reg +loss_align

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data.item()
        print('Epoch: %d | Step: %d | Loss: %.3f | loss_align: %.3f | loss_reg: %.3f' % (epoch, batch_idx, train_loss / (batch_idx + 1), loss_align, loss_reg))

def test(epoch):

    global best_R1_IOU5
    global best_R5_IOU5
    global best_R1_IOU5_epoch
    global best_R5_IOU5_epoch

    net.eval()

    IoU_thresh = [0.1, 0.3, 0.5, 0.7]
    all_correct_num_10 = [0.0] * 5
    all_correct_num_5 = [0.0] * 5
    all_correct_num_1 = [0.0] * 5
    all_retrievd = 0.0
    all_number = len(test_dataset.movie_names)
    idx = 0
    for movie_name in test_dataset.movie_names:
        idx += 1
        print("%d/%d" % (idx, all_number))
        print("Test movie: " + movie_name + "....loading movie data")
        movie_clip_featmaps, movie_clip_sentences = test_dataset.load_movie_slidingclip(movie_name)

        sentence_image_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])
        sentence_image_reg_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps), 2])
        for k in range(len(movie_clip_sentences)):

            sent_vec = movie_clip_sentences[k][1]
            sent_vec = np.reshape(sent_vec, [1, sent_vec.shape[0]])  # 1,4800
            sent_vec = torch.from_numpy(sent_vec).float().cuda()

            for t in range(len(movie_clip_featmaps)):
                featmap = movie_clip_featmaps[t][1]
                visual_clip_name = movie_clip_featmaps[t][0]

                start = float(visual_clip_name.split("_")[-2])
                end = float(visual_clip_name.split("_")[-1])

                featmap = np.reshape(featmap, [1, featmap.shape[0]])
                featmap = torch.from_numpy(featmap).float().cuda()

                # network forward
                outputs = net(featmap, sent_vec)
                outputs = outputs.squeeze(1).squeeze(1)

                sentence_image_mat[k, t] = outputs[0]

                reg_end = end + outputs[2]
                reg_start = start + outputs[1]

                sentence_image_reg_mat[k, t, 0] = reg_start
                sentence_image_reg_mat[k, t, 1] = reg_end

        iclips = [b[0] for b in movie_clip_featmaps]
        sclips = [b[0] for b in movie_clip_sentences]

        # calculate Recall@m, IoU=n
        for k in range(len(IoU_thresh)):
            IoU = IoU_thresh[k]
            correct_num_10 = compute_IoU_recall_top_n_forreg(10, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            correct_num_5 = compute_IoU_recall_top_n_forreg(5, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            correct_num_1 = compute_IoU_recall_top_n_forreg(1, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            print(movie_name + " IoU=" + str(IoU) + ", R@10: " + str(correct_num_10 / len(sclips)) + "; IoU=" + str(
                IoU) + ", R@5: " + str(correct_num_5 / len(sclips)) + "; IoU=" + str(IoU) + ", R@1: " + str(
                correct_num_1 / len(sclips)))

            all_correct_num_10[k] += correct_num_10
            all_correct_num_5[k] += correct_num_5
            all_correct_num_1[k] += correct_num_1
        all_retrievd += len(sclips)
    for k in range(len(IoU_thresh)):
        print(" IoU=" + str(IoU_thresh[k]) + ", R@10: " + str(all_correct_num_10[k] / all_retrievd) + "; IoU=" + str(
            IoU_thresh[k]) + ", R@5: " + str(all_correct_num_5[k] / all_retrievd) + "; IoU=" + str(
            IoU_thresh[k]) + ", R@1: " + str(all_correct_num_1[k] / all_retrievd))

        test_result_output.write("Epoch " + str(epoch) + ": IoU=" + str(IoU_thresh[k]) + ", R@10: " + str(
            all_correct_num_10[k] / all_retrievd) + "; IoU=" + str(IoU_thresh[k]) + ", R@5: " + str(
            all_correct_num_5[k] / all_retrievd) + "; IoU=" + str(IoU_thresh[k]) + ", R@1: " + str(
            all_correct_num_1[k] / all_retrievd) + "\n")

    R1_IOU5 = all_correct_num_1[2] / all_retrievd
    R5_IOU5 = all_correct_num_5[2] / all_retrievd

    if R1_IOU5 > best_R1_IOU5:
        print("best_R1_IOU5: %0.3f" % R1_IOU5)
        state = {
            'net': net.state_dict(),
            'best_R1_IOU5': best_R1_IOU5,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'best_R1_IOU5_model.t7'))
        best_R1_IOU5 = R1_IOU5
        best_R1_IOU5_epoch = epoch

    if R5_IOU5 > best_R5_IOU5:
        print("best_R5_IOU5: %0.3f" % R5_IOU5)
        state = {
            'net': net.state_dict(),
            'best_R5_IOU5': best_R5_IOU5,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'best_R5_IOU5_model.t7'))
        best_R5_IOU5 = R5_IOU5
        best_R5_IOU5_epoch = epoch


if __name__ == '__main__':
    start_epoch = 0
    total_epoch = 20
    if not os.path.isdir(path):
        os.mkdir(path)
    test_result_output=open(os.path.join(path, "test_results.txt"), "w")
    for epoch in range(start_epoch, total_epoch):
        train(epoch)
        test(epoch)

print("best_R1_IOU5: %0.3f in epoch: %d " % (best_R1_IOU5, best_R1_IOU5_epoch))

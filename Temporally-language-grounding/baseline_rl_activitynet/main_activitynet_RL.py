" Train and test file for Reinforcement Learning based methods (A2C) for Charades-STA dataset \
: Read,Watch, and Move Reinforcement Learning for Temporally Grounding Natural Language Descriptions in video (https://arxiv.org/abs/1901.06829) "

import torch.nn.functional as F
import os
import argparse
from utils import *
from dataloader_activitynet_RL import Activitynet_Train_dataset, Activitynet_Test_dataset
from model_A2C import A2C
import random
import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pysnooper as psp
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
plt.rcParams['figure.figsize'] = (8.0, 4.0)

parser = argparse.ArgumentParser(description='Video Grounding of PyTorch')
parser.add_argument('--model', type=str, default='A2C', help='model type')
parser.add_argument('--dataset', type=str, default='Charades', help='dataset type')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--num_steps', type=int, default=10, help='number of forward steps in A2C (default: 10)')
parser.add_argument('--gamma', type=float, default=0.4,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--entropy-coef', type=float, default=0.1,
                    help='entropy term coefficient (default: 0.01)')

opt = parser.parse_args()

path = os.path.join(opt.dataset + '_' + opt.model)

if not os.path.exists(path):
    os.makedirs(path)

train_dataset = Activitynet_Train_dataset()
test_dataset = Activitynet_Test_dataset()

num_train_batches = int(len(train_dataset) / opt.batch_size)
print("num_train_batches:", num_train_batches)

trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=opt.batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          pin_memory=True)

# Model
if opt.model == 'A2C':
    net = A2C().cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


setup_seed(0)
best_R1_IOU7 = 0
best_R1_IOU5 = 0
best_R5_IOU7 = 0
best_R5_IOU5 = 0

best_R1_IOU7_epoch = 0
best_R1_IOU5_epoch = 0
best_R5_IOU7_epoch = 0
best_R5_IOU5_epoch = 0


def determine_range_test(action, current_offset, ten_unit, num_units):
    abnormal_done = False
    update_offset = np.zeros(2, dtype=np.float32)
    update_offset_norm = np.zeros(2, dtype=np.float32)

    current_offset_start = int(current_offset[0][0])
    current_offset_end = int(current_offset[0][1])
    interval = current_offset_end - current_offset_start

    ten_unit = int(ten_unit)

    if action == 0:
        current_offset_start = current_offset_start + ten_unit
        current_offset_end = current_offset_end + ten_unit
    elif action == 1:
        current_offset_start = current_offset_start - ten_unit
        current_offset_end = current_offset_end - ten_unit
    elif action == 2:
        current_offset_start = current_offset_start + ten_unit
    elif action == 3:
        current_offset_start = current_offset_start - ten_unit
    elif action == 4:
        current_offset_end = current_offset_end + ten_unit
    elif action == 5:
        current_offset_end = current_offset_end - ten_unit

    if current_offset_start < 0:
        current_offset_start = 0
        if current_offset_end < 0:
            abnormal_done = True

    if current_offset_end > num_units:
        current_offset_end = num_units
        if current_offset_start > num_units:
            abnormal_done = True

    if current_offset_end <= current_offset_start:
        abnormal_done = True

    current_offset_start_norm = current_offset_start / float(num_units - 1)
    current_offset_end_norm = current_offset_end / float(num_units - 1)

    update_offset_norm[0] = current_offset_start_norm
    update_offset_norm[1] = current_offset_end_norm

    update_offset[0] = current_offset_start
    update_offset[1] = current_offset_end

    update_offset = torch.from_numpy(update_offset)
    update_offset = update_offset.unsqueeze(0).cuda()

    update_offset_norm = torch.from_numpy(update_offset_norm)
    update_offset_norm = update_offset_norm.unsqueeze(0).cuda()

    return current_offset_start, current_offset_end, update_offset, update_offset_norm, abnormal_done



def determine_range(action, current_offset, ten_unit, num_units):
    batch_size = len(action)
    num_units = num_units.float()
    current_offset_start_batch = np.zeros(batch_size, dtype=np.int8)
    current_offset_end_batch = np.zeros(batch_size, dtype=np.int8)
    abnormal_done_batch = torch.ones(batch_size).cuda()
    update_offset = torch.zeros(batch_size, 2)
    update_offset_norm = torch.zeros(batch_size, 2)

    action_embedding = torch.tensor([
        [1, 1],
        [-1, -1],
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
        [0, 0],
        [0, 0],
        [0, 0]
    ]).float().cuda()
    ten_unit = ten_unit.float().cuda()
    offsett_action = F.embedding(action, action_embedding).float() * ten_unit.unsqueeze(-1)
    current_offset_batch = current_offset.long().float()
    current_offset_batch = current_offset_batch + offsett_action

    pre_current_offset_start_batch = current_offset_batch[:, 0]
    # print(pre_current_offset_start_batch < 0)
    pre_current_offset_start_batch = torch.where(pre_current_offset_start_batch < 0,
                                                 torch.zeros_like(pre_current_offset_start_batch),
                                                 pre_current_offset_start_batch)

    pre_current_offset_end_batch = current_offset_batch[:, 1]
    pre_current_offset_end_batch = torch.where(pre_current_offset_end_batch > num_units, num_units,
                                               pre_current_offset_end_batch)
    pre_update_offset = torch.cat((
        pre_current_offset_start_batch.unsqueeze(-1),
        pre_current_offset_end_batch.unsqueeze(-1)
    ), 1)
    pre_update_offset_norm = torch.cat((
        (pre_current_offset_start_batch / (num_units - 1).float()).unsqueeze(-1),
        (pre_current_offset_end_batch / (num_units - 1).float()).unsqueeze(-1)
    ), 1)
    initial_current_start = current_offset[:, 0]
    initial_current_end = current_offset[:, 1]
    # print(initial_current_start, initial_current_start.type())
    pre_abnormal_done_batch = torch.where(initial_current_end < 0,
                                          torch.ones_like(abnormal_done_batch),
                                          torch.zeros_like(abnormal_done_batch)) + \
                              torch.where(initial_current_start.float() > num_units,
                                          torch.ones_like(abnormal_done_batch),
                                          torch.zeros_like(abnormal_done_batch)) + \
                              torch.where(initial_current_end <= initial_current_start,
                                          torch.ones_like(abnormal_done_batch),
                                          torch.zeros_like(abnormal_done_batch)) + \
                              torch.where(action == 6,
                                          torch.ones_like(abnormal_done_batch),
                                          torch.zeros_like(abnormal_done_batch)) + \
                              torch.where(pre_current_offset_start_batch > num_units,
                                          torch.ones_like(abnormal_done_batch),
                                          torch.zeros_like(abnormal_done_batch)) + \
                              torch.where(pre_current_offset_end_batch < 0,
                                          torch.ones_like(abnormal_done_batch),
                                          torch.zeros_like(abnormal_done_batch)) + \
                              torch.where(pre_current_offset_end_batch <= pre_current_offset_start_batch,
                                          torch.ones_like(abnormal_done_batch),
                                          torch.zeros_like(abnormal_done_batch))
    pre_abnormal_done_batch = pre_abnormal_done_batch.le(0)
    return pre_current_offset_start_batch.int(), pre_current_offset_end_batch.int(), pre_update_offset.int(), pre_update_offset_norm, pre_abnormal_done_batch



# Training
# @psp.snoop(depth=1)
def train(epoch):
    net.train()
    train_loss = 0
    policy_loss_epoch = []
    value_loss_epoch = []
    total_rewards_epoch = []

    for batch_idx, (
            global_feature, original_feats, initial_feature, sentence, offset_norm, initial_offset, initial_offset_norm,
            ten_unit, num_units) in enumerate(trainloader):

        global_feature, original_feats, initial_feature, sentence, offset_norm, initial_offset, initial_offset_norm, ten_unit, num_units = global_feature.float().cuda(), \
                                                                                                                                           original_feats.float().cuda(), initial_feature.float().cuda(), sentence.float().cuda(), offset_norm.float().cuda(), initial_offset.cuda(), initial_offset_norm.float().cuda(), ten_unit.cuda(), num_units.cuda()

        batch_size = len(global_feature)
        entropies = torch.zeros(opt.num_steps, batch_size)
        values = torch.zeros(opt.num_steps, batch_size)
        log_probs = torch.zeros(opt.num_steps, batch_size)
        rewards = torch.zeros(opt.num_steps, batch_size)
        Previous_IoUs = torch.zeros(opt.num_steps, batch_size)
        Predict_IoUs = torch.zeros(opt.num_steps, batch_size)
        locations = torch.zeros(opt.num_steps, batch_size, 2)
        mask = torch.zeros(opt.num_steps, batch_size)

        # network forward
        for step in range(opt.num_steps):

            if step == 0:
                hidden_state = torch.zeros(batch_size, 1024).cuda()
                current_feature = initial_feature
                current_offset = initial_offset
                current_offset_norm = initial_offset_norm

            hidden_state, logit, value, tIoU, location = net(global_feature, \
                                                             current_feature, sentence, current_offset_norm,
                                                             hidden_state)
            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1)
            entropies[step, :] = entropy

            action = prob.multinomial(num_samples=1).data
            log_prob = log_prob.gather(1, action)
            action = action.squeeze()
            current_offset_start, current_offset_end, current_offset, \
            current_offset_norm, abnormal_done = determine_range(action, \
                                                                 current_offset, ten_unit, num_units)
            # print(current_offset_start)
            if step == 0:
                Previou_IoU = calculate_RL_IoU_batch(initial_offset_norm, offset_norm)
            else:
                Previou_IoU = current_IoU

            Previous_IoUs[step, :] = Previou_IoU
            mask[step, :] = abnormal_done

            current_IoU = calculate_RL_IoU_batch(current_offset_norm, offset_norm)
            current_feature = torch.zeros_like(initial_feature).cuda()
            # source
            for i in range(batch_size):
                abnormal = abnormal_done[i]
                if abnormal == 1:
                    current_feature_med = original_feats[i] \
                        [(current_offset_start[i]):(current_offset_end[i] + 1)]
                    current_feature_med = torch.mean(current_feature_med, dim=0)
                    current_feature[i] = current_feature_med
            reward = calculate_reward_batch_withstop(Previou_IoU, current_IoU, step + 1)
            values[step, :] = value.squeeze(1)
            log_probs[step, :] = log_prob.squeeze(1)
            rewards[step, :] = reward
            locations[step, :] = location
            Predict_IoUs[step, :] = tIoU.squeeze()

        total_rewards_epoch.append(rewards.sum())

        policy_loss = 0
        value_loss = 0
        idx = 0
        for j in range(batch_size):
            mask_one = mask[:, j]
            index = opt.num_steps
            for i in range(opt.num_steps):
                if mask_one[i] == 0:
                    index = i + 1
                    break

            for k in reversed(list(range(index))):
                if k == index - 1:
                    R = opt.gamma * values[k][j] + rewards[k][j]
                else:
                    R = opt.gamma * R + rewards[k][j]

                advantage = R - values[k][j]

                value_loss = value_loss + advantage.pow(2)
                policy_loss = policy_loss - log_probs[k][j] * advantage - \
                              opt.entropy_coef * entropies[k][j]
                idx += 1

        policy_loss /= idx
        value_loss /= idx

        policy_loss_epoch.append(policy_loss.item())
        value_loss_epoch.append(value_loss.item())

        iou_loss = 0
        iou_id = 0
        mask_1 = np.zeros_like(Previous_IoUs)
        for i in range(len(Previous_IoUs)):
            for j in range(len(Previous_IoUs[i])):
                iou_id += 1
                iou_loss += torch.abs(Previous_IoUs[i, j] - Predict_IoUs[i, j])
                mask_1[i, j] = Previous_IoUs[i, j] > 0.4
        iou_loss /= iou_id

        loc_loss = 0
        loc_id = 0
        for i in range(len(mask_1)):
            for j in range(len(mask_1[i])):
                if mask_1[i, j] == 1:
                    loc_loss += (torch.abs(offset_norm[j][0].cpu() \
                                           - locations[i][j][0]) + torch.abs(
                        offset_norm[j][1].cpu() - locations[i][j][1]))
                    loc_id += 1
        loc_loss /= loc_id

        optimizer.zero_grad()
        (policy_loss + value_loss + iou_loss + loc_loss).backward(retain_graph=True)
        optimizer.step()

        print("Train Epoch: %d | Index: %d / %d | policy loss: %f" % (
            epoch, batch_idx + 1, len(train_dataset) / opt.batch_size, policy_loss.item()))
        print("Train Epoch: %d | Index: %d / %d | value_loss: %f" % (
            epoch, batch_idx + 1, len(train_dataset) / opt.batch_size, value_loss.item()))

        # test(epoch)
        print("Train Epoch: %d | Index: %d / %d | iou_loss: %f" % (
            epoch, batch_idx + 1, len(train_dataset) / opt.batch_size, iou_loss.item()))
        if loc_loss > 0:
            print("Train Epoch: %d | Index: %d / %d | location_loss: %f" % (
                epoch, batch_idx + 1, len(train_dataset) / opt.batch_size, loc_loss.item()))

    ave_policy_loss = sum(policy_loss_epoch) / len(policy_loss_epoch)
    ave_policy_loss_all.append(ave_policy_loss)
    print("Average Policy Loss for Train Epoch %d : %f" % (epoch, ave_policy_loss))

    ave_value_loss = sum(value_loss_epoch) / len(value_loss_epoch)
    ave_value_loss_all.append(ave_value_loss)
    print("Average Value Loss for Train Epoch %d : %f" % (epoch, ave_value_loss))

    ave_total_rewards_epoch = sum(total_rewards_epoch) / len(total_rewards_epoch)
    ave_total_rewards_all.append(ave_total_rewards_epoch)
    print("Average Total reward for Train Epoch %d: %f" % (epoch, ave_total_rewards_epoch))

    with open(path + "/iteration_ave_reward.pkl", "wb") as file:
        pickle.dump(ave_total_rewards_all, file)
    # plot the val loss vs epoch and save to disk:
    x = np.arange(1, len(ave_total_rewards_all) + 1)
    plt.figure(1)
    plt.plot(x, ave_total_rewards_all, "r-")
    plt.ylabel("Rewards")
    plt.xlabel("Iteration")
    plt.title("Average Reward iteration")
    plt.xticks(fontsize=8)
    plt.savefig(path + "/iteration_ave_reward.png")
    plt.close(1)

    with open(path + "/iteration_ave_policy_loss.pkl", "wb") as file:
        pickle.dump(ave_policy_loss_all, file)
    # plot the val loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(x, ave_policy_loss_all, "r-")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.title("Average Policy Loss iteration")
    plt.xticks(fontsize=8)
    plt.savefig(path + "/iteration_ave_policy_loss.png")
    plt.close(1)

    with open(path + "/iteration_ave_value_loss.pkl", "wb") as file:
        pickle.dump(ave_value_loss_all, file)
    # plot the val loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(x, ave_value_loss_all, "r-")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.title("Average Value Loss iteration")
    plt.xticks(fontsize=8)
    plt.savefig(path + "/iteration_ave_value_loss.png")
    plt.close(1)


def test(epoch):
    global best_R1_IOU7
    global best_R1_IOU5
    global best_R1_IOU7_epoch
    global best_R1_IOU5_epoch

    net.eval()

    IoU_thresh = [0.1, 0.3, 0.5, 0.7]
    all_correct_num_1 = [0.0] * 5
    all_retrievd = 0.0
    all_number = len(test_dataset.movie_names)
    idx = 0

    for movie_name in test_dataset.movie_names:
        idx += 1
        print("%d/%d" % (idx, all_number))
        # movie_length = test_dataset.movie_length_info[movie_name.split(".")[0]]
        print("Test movie: " + movie_name + "....loading movie data")

        movie_clip_sentences, global_feature, original_feats, \
        initial_feature, initial_offset, initial_offset_norm, \
        ten_unit, num_units = test_dataset.load_movie_slidingclip(movie_name)

        global_feature = torch.from_numpy(global_feature).float().cuda().unsqueeze(0)
        original_feats = torch.from_numpy(original_feats).float().cuda().unsqueeze(0)
        initial_feature = torch.from_numpy(initial_feature).float().cuda().unsqueeze(0)
        initial_offset = torch.from_numpy(initial_offset).float().cuda().unsqueeze(0)
        initial_offset_norm = torch.from_numpy(initial_offset_norm).float().cuda().unsqueeze(0)
        ten_unit = torch.from_numpy(ten_unit).cuda().unsqueeze(0)
        num_units = torch.from_numpy(num_units).float().cuda().unsqueeze(0)

        print("sentences: " + str(len(movie_clip_sentences)))

        sentence_image_reg_mat = np.zeros([len(movie_clip_sentences), 2])

        for k in range(len(movie_clip_sentences)):

            sent_vec = movie_clip_sentences[k][1]
            sent_vec = np.reshape(sent_vec, [1, sent_vec.shape[0]])  # 1,4800
            sent_vec = torch.from_numpy(sent_vec).cuda()

            # network forward
            for step in range(opt.num_steps):
                if step == 0:
                    hidden_state = torch.zeros(1, 1024).cuda()
                    current_feature = initial_feature
                    current_offset = initial_offset
                    current_offset_norm = initial_offset_norm

                # hidden_state, logit, value = net(global_feature, current_feature, sent_vec, current_offset_norm, hidden_state)
                hidden_state, logit, value, tIoU, location = net(global_feature, current_feature, sent_vec,
                                                                 current_offset_norm, hidden_state)

                prob = F.softmax(logit, dim=1)
                action = prob.max(1, keepdim=True)[1].data.cpu().numpy()[0, 0]

                current_offset_start, current_offset_end, \
                current_offset, current_offset_norm, abnormal_done = determine_range_test(
                    action, current_offset, ten_unit, num_units)

                if not abnormal_done:
                    current_feature = original_feats[0][int(current_offset_start):int(current_offset_end + 1)]
                    current_feature = torch.mean(current_feature, dim=0)
                    current_feature = current_feature.unsqueeze(0).cuda()

                if action == 6 or abnormal_done == True:
                    break

            sentence_image_reg_mat[k, 0] = current_offset_start / num_units
            sentence_image_reg_mat[k, 1] = current_offset_end / num_units

        sclips = [b[0] for b in movie_clip_sentences]

        # calculate Recall@m, IoU=n
        for k in range(len(IoU_thresh)):
            IoU = IoU_thresh[k]
            correct_num_1 = compute_IoU_recall_top_n_forreg_rl(1, IoU, sentence_image_reg_mat, sclips)
            print(movie_name + " IoU=" + str(IoU) + ", R@1: " + str(correct_num_1 / len(sclips)))

            all_correct_num_1[k] += correct_num_1
        all_retrievd += len(sclips)

        print("Current R1_IOU7", all_correct_num_1[3] / all_retrievd)
        print("Current R1_IOU5", all_correct_num_1[2] / all_retrievd)

    for k in range(len(IoU_thresh)):
        print(" IoU=" + str(IoU_thresh[k]) + ", R@1: " + str(all_correct_num_1[k] / all_retrievd))

        test_result_output.write("Epoch " + str(epoch) + ": IoU=" + str(IoU_thresh[k]) + ", R@1: " + str(
            all_correct_num_1[k] / all_retrievd) + "\n")

    R1_IOU7 = all_correct_num_1[3] / all_retrievd
    R1_IOU5 = all_correct_num_1[2] / all_retrievd

    R1_IOU7_all.append(R1_IOU7)
    print("R1_IOU7 for Train Epoch %d : %f" % (epoch, R1_IOU7))

    R1_IOU5_all.append(R1_IOU5)
    print("R1_IOU5 for Train Epoch %d : %f" % (epoch, R1_IOU5))

    with open(path + "/R1_IOU7_all.pkl", "wb") as file:
        pickle.dump(R1_IOU7_all, file)
    # plot the val loss vs epoch and save to disk:
    x = np.arange(1, len(R1_IOU7_all) + 1)
    plt.figure(1)
    plt.plot(x, R1_IOU7_all, "r-")
    plt.ylabel("Recall")
    plt.xlabel("epoch")
    plt.title("R1_IOU7_all")
    plt.xticks(fontsize=8)
    plt.savefig(path + "/R1_IOU7_all.png")
    plt.close(1)

    with open(path + "/R1_IOU5_all.pkl", "wb") as file:
        pickle.dump(R1_IOU5_all, file)
    # plot the val loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(x, R1_IOU5_all, "r-")
    plt.ylabel("Recall")
    plt.xlabel("epoch")
    plt.title("R1_IOU5_all")
    plt.xticks(fontsize=8)
    plt.savefig(path + "/R1_IOU5_all.png")
    plt.close(1)

    if R1_IOU7 > best_R1_IOU7:
        print("best_R1_IOU7: %0.3f" % R1_IOU7)
        state = {
            'net': net.state_dict(),
            'best_R1_IOU7': R1_IOU7,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'best_R1_IOU7_model.t7'))
        best_R1_IOU7 = R1_IOU7
        best_R1_IOU7_epoch = epoch

    if R1_IOU5 > best_R1_IOU5:
        print("best_R1_IOU5: %0.3f" % R1_IOU5)
        state = {
            'net': net.state_dict(),
            'best_R1_IOU5': R1_IOU5,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'best_R1_IOU5_model.t7'))
        best_R1_IOU5 = R1_IOU5
        best_R1_IOU5_epoch = epoch


if __name__ == '__main__':
    start_epoch = 0
    total_epoch = 100
    ave_policy_loss_all = []
    ave_value_loss_all = []
    ave_total_rewards_all = []

    R1_IOU7_all = []
    R1_IOU5_all = []

    test_result_output = open(os.path.join(path, "A2C_results.txt"), "w")
    for epoch in range(start_epoch, total_epoch):
        train(epoch)
        test(epoch)

print("best_R1_IOU7: %0.3f in epoch: %d " % (best_R1_IOU7, best_R1_IOU7_epoch))
print("best_R1_IOU5: %0.3f in epoch: %d " % (best_R1_IOU5, best_R1_IOU5_epoch))

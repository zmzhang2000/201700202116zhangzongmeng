# Language-driven Temporal Activity Localization: A Semantic Matching Reinforcement Learning Model
## Introduction
### 分类
* Action Recognition 假设视频片段已人为剪辑且已预定义动作种类
* Temporal Action Detection 同时识别和定位动作
### 已有工作缺陷
* 检索方式停留在词级别
* 滑动窗口时间开销大
* 平均池化提取视频特征不能充分利用时序信息
### 本文模型
* 每一个时间步，RNN的hidden state在句嵌入的监督下选择下一个观测位置并且输出候选检测；观察若干个视频帧之后输出最终的动作时序边界
* 使用匹配方式联系整个句子和视频，而不是单独为每个动作训练分类器
### 本文主要贡献
* 提出基于RNN的语言驱动时序行为检测强化学习模型
* 引入中层语义概念联系视觉和语义信息
* 提高效果和速度

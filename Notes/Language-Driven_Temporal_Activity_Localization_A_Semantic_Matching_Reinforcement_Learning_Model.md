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
## Related Work
### Temporal Action Detection
* 生成temporal video proposals，然后对每个proposal的动作分类
* 弱监督学习
* 强化学习
### Temporal Action Proposal Generation
* 目标：从未修剪视频中抽取语义上比较重要的片段
* 一般将此问题视为分类问题(是否重要)
* 本文使用自然语言描述来定位动作
## Methodology
### Semantic Concept Learning
#### 构建词汇表
* TACoS中的Attribute annotations
* Charades-STA、DiDeMo的sentence中抽取名词、形容词、动词和数字作为semantic concepts
#### 分类
* 基于多标签分类模型
* 基于Faster R-CNN模型
### Semantic Matching Reinforcement Learning
* skip-thoughts编码查询语句
* VGG-16提取视频帧的global context feature，和semantic concept、location embedding拼接嵌入，输入LSTM
* 上述两项拼接
#### State and Action Space
1. candidate detection
2. matching score
3. binary prediction indicator
#### Semantic Matching
交叉熵损失
### Location Regression Loss
为了预测结果更准确，每个时间步都使用反向传播
#### 边界回归损失
#### 帧级别回归损失
### Reward
prediction indicator和observation location**不可导**，不能使用反向传播，所以使用强化学习

## Experiments
## Conclusion and Future Work

---
aliases:
  - "RLHF流程"
  - "Reinforcement Learning from Human Feedback Pipeline"
tags:
  - llm
  - llm/procedure
---

# RLHF Pipeline

## Step 1: 收集指令与示范
准备用户任务和高质量示范回答，训练初始助手模型。

## Step 2: 训练奖励模型
让标注者比较多个回答，训练 reward model 预测人类偏好。

## Step 3: 强化学习优化
用 PPO 等方法让策略模型提高奖励，同时控制偏离初始模型的程度。

## Step 4: 评估与安全检查
检查有用性、无害性、真实性和拒答边界。

## 输出
一个更符合人类偏好的助手模型。

## 相关卡片
[[Instruction Tuning and RLHF]]、[[Preference Optimization]]

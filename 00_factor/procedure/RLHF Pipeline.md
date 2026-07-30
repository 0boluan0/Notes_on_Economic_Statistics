---
aliases:
  - "RLHF流程"
  - "Reinforcement Learning from Human Feedback Pipeline"
tags:
  - llm
  - llm/procedure
type: procedure
---

# RLHF Pipeline

## 输入

- 指令与高质量示范数据、偏好比较数据、基础/SFT 模型。
- 奖励模型训练配置、策略优化预算，以及有用性、真实性和安全评测集。

## Step 1: 收集指令与示范
准备用户任务和高质量示范回答，训练初始助手模型。

## Step 2: 训练奖励模型
让标注者比较多个回答，训练 reward model 预测人类偏好。

## Step 3: 强化学习优化
用 PPO 等方法让策略模型提高奖励，同时控制偏离初始模型的程度。

## Step 4: 评估与安全检查
检查有用性、无害性、真实性和拒答边界。

## 检查点

- 检查标注一致性、奖励模型在分布外样本上的校准和策略的 KL 偏离。
- 识别奖励投机、过度拒答、模式化回答和基础能力回退；评测不能只看 reward。

## 输出
一个更符合人类偏好的助手模型。

## 相关卡片
[[Instruction Tuning and RLHF]]、[[Preference Optimization]]

---
aliases:
  - "DPO流程"
  - "Direct Preference Optimization Procedure"
tags:
  - llm
  - llm/procedure
---

# DPO Procedure

## Step 1: 准备偏好对
每条样本包含 prompt、chosen response 和 rejected response。

## Step 2: 选择参考模型
通常使用 SFT 后的模型作为参考，约束优化不要偏离太远。

## Step 3: 优化偏好目标
直接提高 chosen 相对 rejected 的概率优势。

## Step 4: 评估偏好与退化
检查回答质量、安全性和能力回退。

## 输出
一个不显式训练 reward model 的偏好优化模型。

## 相关卡片
[[Preference Optimization]]、[[Instruction Tuning and RLHF]]

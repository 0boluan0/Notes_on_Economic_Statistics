---
aliases:
  - "DPO流程"
  - "Direct Preference Optimization Procedure"
tags:
  - llm
  - llm/procedure
type: procedure
---

# DPO Procedure

## 输入

- 已完成 SFT 或至少具备基本指令跟随能力的策略模型。
- 偏好数据集：`prompt`、`chosen`、`rejected`，并明确数据来源与安全标签。
- 参考模型、温度/β 等超参数，以及能力回归和安全评测集。

## Step 1: 准备偏好对
每条样本包含 prompt、chosen response 和 rejected response。

## Step 2: 选择参考模型
通常使用 SFT 后的模型作为参考，约束优化不要偏离太远。

## Step 3: 优化偏好目标
直接提高 chosen 相对 rejected 的概率优势。

## Step 4: 评估偏好与退化
检查回答质量、安全性和能力回退。

## 检查点

- 检查 chosen/rejected 是否真的表达稳定偏好，避免长度、格式或标注噪声成为捷径。
- 比较训练前后胜率、事实性、安全性和通用能力；监控 KL 偏离与过拟合。
- 若偏好提升但基础能力下降，降低更新强度或重审数据，而不是只扩大训练步数。

## 输出
一个不显式训练 reward model 的偏好优化模型。

## 相关卡片
[[Preference Optimization]]、[[Instruction Tuning and RLHF]]

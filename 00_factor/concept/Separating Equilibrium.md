---
aliases:
- 分离均衡
- Separating Equilibrium
tags:
- concept
- game theory
---
# 分离均衡

>[!note] 定义
>
> 在贝叶斯博弈（特别是信号博弈）中，不同类型的发送者选择不同的信号，使得接收者能够从信号中准确推断发送者的类型。
>
## 特征

### 信号与类型一一对应
- 类型 θ_1 的发送者选择信号 $m_1$
- 类型 θ_2 的发送者选择信号 $m_2$
- $m_1$ ≠ $m_2$

### 完美信息揭示
- 接收者看到信号后，能确定发送者的类型
- 接收者的信念是完全准确的

### 零混淆
- 类型之间没有混淆
- 信息集是完全分离的

## 存在条件

分离均衡的存在需要满足激励相容约束（Incentive Compatibility Constraint）：

对于每种类型 θ，选择对应信号 m(θ) 的收益必须高于选择其他类型信号的收益：

$ U(\theta, m(\theta)) \geq U(\theta, m(\theta')) \quad \forall \theta \neq \theta' $

即：给定接收者的最优反应，每种类型都没有动机去模仿其他类型。

## 与混合均衡的比较

| 特征 | 分离均衡 | 混合均衡 |
|------|---------|---------|
| 信号选择 | 不同类型选不同信号 | 所有类型选相同信号 |
| 信息揭示 | 完全揭示 | 部分揭示 |
| 信念更新 | 确定信念 | 不变（基于先验） |
| 效率 | 高（信息不对称消除） | 低（信息不对称持续） |

>[!example] 典型例子
>
> ### 劳动市场（Spence 模型）
> - 高能力者选择高教育水平
> - 低能力者选择低教育水平
> - 雇主根据教育水平推断能力
>
> ### 信号博弈
> - 类型 "Strong" 选择 Beer
> - 类型 "Weak" 选择 Quiche
> - 对手可以准确判断类型
>
## 相关概念

- [[Pooling Equilibrium|混合均衡]]
- [[Perfect Bayesian Equilibrium|完美贝叶斯均衡]]
- [[Bayesian Game|贝叶斯博弈]]

## 应用



- 教育选择作为能力信号
- 价格作为质量信号
- 广告强度作为产品质量信号

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```

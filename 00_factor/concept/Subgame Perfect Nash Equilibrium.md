---
aliases:
- 子博弈精炼纳什均衡
- SPNE
- Subgame Perfect Nash Equilibrium
tags:
- concept
- game theory
---
# 子博弈精炼纳什均衡

## 定义

一个博弈的均衡同时也是其所有子博弈的均衡，这就是子博弈精炼纳什均衡（SPNE）。

## 子博弈的定义

从一个扩展性博弈出发：
- 从某个**单点信息集**（singleton information set）开始
- 往下所有可能的博弈过程
- **不切割任何一个信息集**
- 这个部分就是一个子博弈

**重要**：一个扩展性博弈一定有它自己作为自己的子博弈。

## 性质

1. **更强的均衡概念**：SPNE 是比纳什均衡更强的概念
2. **序贯理性**：要求每个子博弈都是纳什均衡
3. **可信性**：剔除不可信的威胁和承诺

## 存在性

### 完美信息有限博弈
- 一定存在子博弈精炼纳什均衡
- 可以通过逆向归纳法求解

### 不完美信息博弈
- 可能不存在子博弈精炼纳什均衡

## 求解方法

### 逆向归纳法
在完美信息博弈中，SPNE 等价于逆向归纳法：
1. 从最后一个决策点开始
2. 找到该决策点的最优策略
3. 逐层倒推
4. 得到的策略组合就是 SPNE

### 直观检验法
1. 找到所有纳什均衡
2. 检查每个纳什均衡在所有子博弈中是否仍然是最优
3. 通过检查的是 SPNE

## 与其他均衡概念的关系

```
子博弈精炼纳什均衡 (SPNE)
    ⊂ 完美贝叶斯均衡 (PBE)
        ⊂ 纳什均衡 (NE)
```

SPNE 是最强的均衡概念之一。

## 意义

1. **剔除空洞威胁**：只有在特定情况下才是最优的威胁才是可信的
2. **时间一致性**：策略在博弈的每个阶段都是最优的
3. **预期一致性**：玩家的预期与实际行动一致

## 典型例子

### 入侵威慑（Entry Deterrence）
- 空洞威胁：在位者威胁"如果你进入我就摧毁你"
- 这种威胁是不可信的（如果进入已经发生，在位者的最优策略不是摧毁）
- SPNE 会剔除这种不可信的均衡

### 千足虫博弈（Centipede Game）
- 唯一的 SPNE 是从一开始就停止
- 虽然这看起来不合理，但确实是序贯理性的

## 相关概念

- [[Nash Equilibrium|纳什均衡]]
- [[Perfect Bayesian Equilibrium|完美贝叶斯均衡]]
- [[Backward Induction|逆向归纳法]]
- [[Subgame|子博弈]]

## 应用

- 序贯寡头竞争
- 谈判理论
- 契约设计
- 策略性贸易政策

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

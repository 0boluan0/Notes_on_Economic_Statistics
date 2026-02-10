---
aliases:
- 货币乘数
- Deposit Multiplier
- Money Multiplier
tags:
- concept
- money and banking
---
# 货币乘数

## 定义

货币乘数（Money Multiplier）表示基础货币（高能货币）变动对货币供应量的放大效应。

$ M = m \times MB $

## 核心思想

由于银行体系的存款创造机制，一笔基础货币能够产生多倍的货币供应量。

## 推导过程

### 1. 两个漏损

1. **现金比率（c）**：$c = C/D$（现金对存款的比率）
2. **超额准备金比率（e）**：$e = ER/D$（超额准备金对存款的比率）

### 2. 货币乘数公式

$ M = \frac{1 + c}{rr + e + c} \times MB $

因此：

$ m = \frac{1 + c}{rr + e + c} $

其中：
- rr：法定准备金率
- e：超额准备金比率
- c：现金比率

## 公式分解

$ m = \frac{1}{\frac{rr + e + c}{1 + c}} = \frac{1}{\frac{rr + e}{1 + c} + \frac{c}{1 + c}} $

分子分母各项的含义：
- $\frac{rr + e}{1 + c}$：存款对总负债的比率
- $\frac{c}{1 + c}$：现金对总负债的比率

## 决定因素

1. **法定准备金率（rr）**：负相关
   - rr 越高，货币乘数越小

2. **超额准备金（ER）**：负相关
   - ER 越高，货币乘数越小

3. **现金持有水平（c）**：负相关
   - c 越高，货币乘数越小

4. **非借入准备金（$MB_n$）**：正相关

### 因素影响示意图

```
货币供给 M
  ↑
  │
  │      MB_n  ↑   ─── 正相关
  │
  │      BR    ↑   ─── 负相关
  │
  │      rr    ↑   ─── 负相关
  │
  │      ER    ↑   ─── 负相关
  │
  │      c     ↑   ─── 负相关
  │
  └────────────────────→
```

## 与存款创造的关系

在简化模型中（假定银行不持有任何超额准备金）：

$ \Delta D = \frac{1}{rr} \times \Delta R $

存款创造停止的条件：所有银行的超额准备金都被用光。

## 与基础货币的关系

$ MB = C + R $
$ M = m \times MB $

货币供应量 = 货币乘数 × 基础货币

## 现实意义

1. **政策传导**：央行通过控制基础货币影响货币供给
2. **放大效应**：银行体系放大基础货币的影响
3. **稳定性**：货币乘数的波动影响货币供给的稳定性

## 局限性

简化模型的假设：
- 假定储户对现金没有偏好
- 假定银行对超额准备金没有偏好
- 与事实不符

## 相关概念

- [[Monetary Base|基础货币]]
- [[Deposit Creation|存款创造]]
- [[Required Reserves|准备金]]

## 应用

1. **货币政策分析**：评估货币政策的效果
2. **货币供应量预测**：预测货币供应量的变化
3. **金融稳定**：分析货币乘数的稳定性

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

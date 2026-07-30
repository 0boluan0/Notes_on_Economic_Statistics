---
aliases:
- Heteroscedasticity Diagnosis
- Heteroskedasticity Diagnosis
- 异方差诊断
- 异方差检验
tags:
- system
- econometrics
---
# Heteroscedasticity Diagnosis

## 诊断目标

判断回归误差方差是否随观测变化，并决定是修正标准误、改模型，还是改估计方法。

## 诊断流程

### Step 1：先看残差图

画 $\hat u_i$ 或 $\hat u_i^2$ 对拟合值、关键解释变量的图。

- 扇形扩散：可能异方差。
- 某类样本波动特别大：可能组间异方差。
- 明显曲线形态：可能是模型设定错误。

### Step 2：做正式检验

- 一般性检验：[[White Test]]。
- 方差和解释变量线性相关的检验：Breusch-Pagan。
- 分组方差差异：Goldfeld-Quandt。

### Step 3：判断后果

若外生性仍成立：

- OLS 系数通常仍无偏、一致。
- 经典标准误错误。
- t/F 检验和置信区间不可靠。
- OLS 不再 BLUE。

### Step 4：选择处理方式

| 情况 | 处理 |
| --- | --- |
| 不知道异方差形式，只关心推断 | [[White Robust Standard Errors]] |
| 方差形式可信 | [[Weighted Least Squares Estimation]] |
| 方差来自遗漏变量或函数形式错误 | 先修正 [[Model Misspecification]] |
| 时间序列同时有自相关 | [[Newey-West]] |

## 稳健性检查

- 同时报经典标准误和稳健标准误，观察结论是否变。
- 不要只因为 White Test 显著就立刻换 WLS；权重错可能更糟。
- 经济解释要说明为什么波动随规模或组别变化。

## 来自课程位置

- [[07_异方差]]

## 关联卡片

## 复现规范

记录输入数据与样本区间、模型/检验设定、阈值或显著性水平、软件版本和处理决策；保留诊断图表与原始输出，使“发现—判断—修正”链条可复核。

- [[Heteroskedasticity]]
- [[White Test Steps]]
- [[White Robust Standard Errors]]
- [[Weighted Least Squares]]

---
aliases:
- ADF Test Steps
- ADF检验步骤
- Augmented Dickey-Fuller检验步骤
tags:
- procedure
- 时间序列
- 计量经济学
---
# ADF Test Steps

## 这张卡什么时候用

当你要判断一条时间序列是否有单位根、是否需要差分、能不能直接建 ARMA 时，用 ADF 检验。

## 输入

- 时间序列 $y_t$；
- 是否包含截距/趋势的判断；
- 最大滞后阶数或信息准则；
- 显著性水平。

## 输出

- 拒绝单位根：序列可视为平稳；
- 不能拒绝单位根：序列可能非平稳，需要差分或进一步分析。

## Step 1. 选检验形式

常见三种：

无截距无趋势：
$$
\Delta y_t=\gamma y_{t-1}+\sum_{i=1}^p\delta_i\Delta y_{t-i}+\varepsilon_t.
$$

有截距：
$$
\Delta y_t=\alpha+\gamma y_{t-1}+\sum_{i=1}^p\delta_i\Delta y_{t-i}+\varepsilon_t.
$$

有截距和趋势：
$$
\Delta y_t=\alpha+\beta t+\gamma y_{t-1}+\sum_{i=1}^p\delta_i\Delta y_{t-i}+\varepsilon_t.
$$

## Step 2. 选滞后阶数

用 AIC/BIC，或从较大阶数往下删。

目标是让残差不再有自相关，同时不要浪费太多自由度。

## Step 3. 估计 ADF 回归

用 OLS 估计，关注 $\gamma$。

原假设：
$$
H_0:\gamma=0
$$
表示存在单位根。

备择假设：
$$
H_1:\gamma<0
$$
表示平稳。

## Step 4. 比较 ADF 临界值

计算统计量：
$$
\tau=\frac{\hat{\gamma}}{se(\hat{\gamma})}.
$$

它不服从普通 t 分布。

若统计量比临界值更负，拒绝单位根。

## Step 5. 写结论

结论要写清楚方向：

- “拒绝 $H_0$，认为序列平稳”；
- “不能拒绝 $H_0$，单位根证据仍在”。

不要写成“接受平稳”或“证明非平稳”。

## 常见错误

- 忘记 ADF 的原假设是单位根。
- 用普通 t 临界值。
- 模型形式选错：该加趋势没加，或不该加趋势硬加。
- 对协整残差使用普通 ADF 临界值。

## 来自课程位置

- [[03_平稳时间序列模型#0.回忆用|时间序列 03：单位根检验回忆]]
- [[07_协整和误差修正模型#3.1 EG两步法|时间序列 07：残差单位根检验]]

## 关联卡片

- [[Augmented Dickey-Fuller Test]]
- [[Unit Root Test]]
- [[Phillips-Perron Test]]
- [[Stationarity Tests Comparison]]
- [[First Difference]]

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

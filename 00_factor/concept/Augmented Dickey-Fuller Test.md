---
aliases:
- Augmented Dickey-Fuller Test
- ADF
- ADF Test
- ADF检验
- 增广迪基-福勒检验
tags:
- concept
- 时间序列
- 计量经济学
---
# Augmented Dickey-Fuller Test

## 先记一句话

ADF 检验就是：**在 Dickey-Fuller 单位根检验里加入滞后差分项，处理误差自相关**。

它检验的是：
$$
H_0:\text{存在单位根}.
$$

## 它是什么

典型 ADF 回归是
$$
\Delta y_t=\alpha+\beta t+\gamma y_{t-1}
+\sum_{i=1}^{p}\delta_i\Delta y_{t-i}
+\varepsilon_t.
$$

关键系数是 $\gamma$：

- $H_0:\gamma=0$，存在单位根；
- $H_1:\gamma<0$，序列平稳。

加入滞后差分项 $\Delta y_{t-i}$ 是为了吸收残差自相关。

## 它解决什么判断

ADF 用来判断序列是否需要差分、能否直接建平稳模型、是否满足协整检验的前置条件。

## 常见误区

- ADF 统计量不是普通 t 分布，临界值要用 ADF/DF 临界值。
- 滞后阶数太少会留下自相关，太多会损失功效。
- 趋势项、截距项的选择要结合图形和经济含义。
- 对 EG 残差做 ADF 时，不能直接用普通 ADF 临界值。

## 来自课程位置

- [[07_协整和误差修正模型#3.1 EG两步法|时间序列 07：EG 残差单位根检验]]
- [[03_平稳时间序列模型#0.回忆用|时间序列 03：平稳性和单位根检验回忆]]

## 关联卡片

- [[Unit Root Test]]
- [[ADF Test Steps]]
- [[Phillips-Perron Test]]
- [[Stationarity]]
- [[Cointegration]]

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

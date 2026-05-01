---
aliases:
- Realized Volatility
- RV
- realized volatility
- 已实现波动率
tags:
- concept
- 金融风险
- 波动建模
---
# Realized Volatility

## 先记一句话

已实现波动率就是：**用日内高频收益平方和估计某一天实际发生的波动**。

它比普通日收益标准差更接近“当天真实波动”。

## 它是什么

把一天分成 $m$ 个小区间，计算高频收益 $u_i$。

已实现方差：
$$
RV_t=\sum_{i=1}^{m}u_{t,i}^2.
$$

已实现波动率：
$$
\sqrt{RV_t}.
$$

## 它解决什么判断

RV 回答：

> 今天实际发生了多少波动？

它常用于检验和校准 GARCH、EWMA 等波动预测模型。

## 常见误区

- 高频越高不一定越好；太高会引入微观结构噪声。
- RV 是 realized，不是 implied；它看的是已经发生的波动。
- 采样频率选择会影响估计。

## 来自课程位置

- [[10_波动率|金融风险管理 10：已实现波动率]]
- [[04_波动建模 Modeling Volatility#1.1 为什么要进行波动建模|时间序列 04：RV 作为高频波动度量]]

## 关联卡片

- [[Historical Volatility]]
- [[Implied Volatility]]
- [[GARCH]]
- [[EWMA]]

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

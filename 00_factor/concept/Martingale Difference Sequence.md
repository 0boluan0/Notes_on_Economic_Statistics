---
aliases:
- 鞅差序列
- MDS
- Martingale Difference Sequence
tags:
- 时间序列
- 概率论
- concept
---
$鞅差序列（Martingale Difference Sequence, MDS）满足 E(\varepsilon_t\mid \mathcal{F}_{t-1})=0，即在给定过去信息下，当前扰动的条件期望为 0。$

## 定义

在滤过 $\{\mathcal{F}_t\}$ 上，若对所有 $t$ 有：
$ E(\varepsilon_t\mid \mathcal{F}_{t-1})=0, $
则 $\{\varepsilon_t\}$ 为鞅差序列。

## 关系

- $若 \{\varepsilon_t\} 为 i.i.d. 且 E\varepsilon_t=0，则必为 MDS；反之不成立。$
- MDS 蕴含零均值与不相关，但允许条件异方差（如 ARCH）。

## Connections

- 相关：[[White Noise|白噪声过程]]、独立同分布（i.i.d.）、[[ARCH]]
- 经济含义：在弱式有效市场中，收益超均值部分常被建模为 MDS。

## $source_notes$

- [[03_平稳时间序列模型#1.2 三种‘没有关系’的辨析]]

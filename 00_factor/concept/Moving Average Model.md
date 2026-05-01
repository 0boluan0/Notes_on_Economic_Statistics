---
aliases:
- Moving Average Model
- MA Model
- MA
- 移动平均模型
tags:
- concept
- 时间序列
---
# Moving Average Model

## 先记一句话

MA 模型就是：**当前值由当前冲击和过去冲击共同解释**。

MA$(q)$ 写作
$$
y_t=\mu+\varepsilon_t+\theta_1\varepsilon_{t-1}+\cdots+\theta_q\varepsilon_{t-q}.
$$

## 它解决什么判断

MA 模型适合描述“冲击影响会持续有限期”的平稳序列。

如果 ACF 在 $q$ 阶后截尾，而 PACF 拖尾，就优先怀疑 MA$(q)$。

## 一个最小例子

MA(1)：
$$
y_t=\mu+\varepsilon_t+\theta\varepsilon_{t-1}.
$$

当前值受本期 shock 和上一期 shock 影响。再早的 shock 不直接进入。

## 平稳性和可逆性

有限阶 MA 通常是平稳的。

但为了让模型表达唯一，需要可逆性条件。直觉上，可逆性保证 MA 可以写成一个收敛的 AR($\infty$) 表示。

## 常见误区

- MA 不是移动平均平滑法；它是 shock 的移动平均。
- MA 阶数主要看 ACF 截尾，不是 PACF。
- MA 部分不负责 ARMA 的平稳性，但负责可逆性。

## 来自课程位置

- [[03_平稳时间序列模型#1.1.2 MA过程|时间序列 03：MA 过程]]
- [[03_平稳时间序列模型#3. ACF|时间序列 03：ACF/PACF 识别]]

## 关联卡片

- [[ARMA]]
- [[Autocorrelation Function]]
- [[Partial Autocorrelation Function]]
- [[White Noise]]

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

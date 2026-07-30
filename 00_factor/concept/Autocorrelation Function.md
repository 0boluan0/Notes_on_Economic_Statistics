---
aliases:
- Autocorrelation Function
- ACF
- autocorrelation function
- 自相关函数
- 自相关函数 ACF
tags:
- concept
- 时间序列
---

# Autocorrelation Function

## 它是什么

ACF 就是：序列和自己滞后多少期之后还剩多少相关性。

## 先记一句话

ACF 就是：**序列和自己滞后多少期之后还剩多少相关性**。

对平稳序列，滞后 $k$ 的自相关是
$$
\rho_k=\frac{\gamma_k}{\gamma_0}.
$$

其中
$$
\gamma_k=\operatorname{Cov}(y_t,y_{t-k}).
$$

## 它解决什么判断

ACF 用来判断：

- 序列有没有记忆；
- ARMA 模型的候选阶数；
- 模型残差是否接近 [[White Noise]]；
- 冲击影响是否逐渐衰减。

## 一个最小例子

白噪声满足：
$$
\rho_0=1,\qquad \rho_k=0\quad(k\neq0).
$$

AR(1) 在平稳时：
$$
\rho_k=a_1^k.
$$

这说明 AR(1) 的记忆是逐渐衰减的。

## 在 ARMA 识别里的作用

| 模型 | ACF 图像 |
| --- | --- |
| AR(p) | 拖尾 |
| MA(q) | q 阶后截尾 |
| ARMA(p,q) | 拖尾 |

所以 ACF 主要帮助识别 MA 阶数，也帮助检查残差是否还有相关结构。

## 常见误区

- ACF 的定义依赖平稳性；非平稳序列的 ACF 常常会慢慢衰减，看起来很“显著”。
- ACF 截尾/拖尾是理论图像，样本图里会有噪声。
- ACF 看的是总相关，包含中间滞后的间接影响；直接影响要看 [[Partial Autocorrelation Function]]。

## 来自课程位置

- [[03_平稳时间序列模型#3. ACF|时间序列 03：ACF 计算与识别]]
- [[03_平稳时间序列模型#0.回忆用|时间序列 03：ACF/PACF 回忆索引]]

## 关联卡片

- [[Partial Autocorrelation Function]]
- [[ARMA]]
- [[Yule-Walker equations]]
- [[White Noise Test]]
- [[Ljung-Box Test]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[White Noise]]、[[Partial Autocorrelation Function]]、[[03_平稳时间序列模型]]、[[ARMA]]、[[Yule-Walker equations]]、[[White Noise Test]]、[[Ljung-Box Test]]。

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

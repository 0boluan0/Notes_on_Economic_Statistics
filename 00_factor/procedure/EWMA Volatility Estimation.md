---
aliases:
- EWMA Volatility Estimation
- EWMA波动率估计
- EWMA estimation
tags:
- procedure
- 金融风险
- 波动建模
---
# EWMA Volatility Estimation

## 这张卡什么时候用

当你需要用最新收益率快速更新波动率估计，尤其是风险监控或 VaR 参数法输入时，用 EWMA。

## 输入

- 收益率序列 $u_t$；
- 上一期方差估计 $h_{t-1}$；
- 衰减因子 $\lambda$。

## 输出

- 本期方差估计 $h_t$；
- 本期波动率 $\sqrt{h_t}$；
- 必要时年化波动率。

## Step 1. 设定衰减因子

常见日频经验值：
$$
\lambda=0.94.
$$

$\lambda$ 越大，历史记忆越长，波动率更新越慢。

## Step 2. 初始化方差

用一个初始窗口的收益率平方均值或样本方差作为 $h_0$。

这个初始值会影响前几期，但递推足够久后影响下降。

## Step 3. 递推更新

使用：
$$
h_t=\lambda h_{t-1}+(1-\lambda)u_{t-1}^2.
$$

这表示：

- 旧方差估计权重为 $\lambda$；
- 最新平方收益权重为 $1-\lambda$。

## Step 4. 转成波动率

方差开根号：
$$
\sigma_t=\sqrt{h_t}.
$$

如果用日数据并需要年化：
$$
\sigma_{\text{annual}}\approx \sigma_t\sqrt{252}.
$$

## Step 5. 检查权重直觉

展开后：
$$
h_t=(1-\lambda)\sum_{i=1}^{\infty}\lambda^{i-1}u_{t-i}^2.
$$

越新的收益平方权重越大。

## 常见错误

- 把 $u_t$ 和 $u_{t-1}$ 的时间点写错。
- 忘记方差和波动率的平方根关系。
- 把 EWMA 当作有长期均值回归的 GARCH；它通常没有固定长期方差。

## 来自课程位置

- [[10_波动率|金融风险管理 10：EWMA]]
- [[04_波动建模 Modeling Volatility#3.1 IGARCH|时间序列 04：EWMA/IGARCH 关系]]

## 关联卡片

- [[EWMA]]
- [[Historical Volatility]]
- [[GARCH]]
- [[IGARCH]]
- [[VaR Parametric Method]]

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

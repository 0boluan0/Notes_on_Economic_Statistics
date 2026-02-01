---
aliases:
- 最小二乘法
- Ordinary least square
tags:
- procedure
---
二乘就是平方的意思，引进的时候翻译不到位。

要求残差平方和最小，即$$
\min \sum_{i=1}^{N} \hat{u}_i^2 = \min \sum_{i=1}^{N} (Y_i - \hat{Y}_i) \hat{u}_i
$$

要求其极小值，只需要一阶偏导得零即可，不需要二阶偏导为正，因为现在要求的是残差平方和极小值，而残差平方和不存在极大值，故其一阶偏导为0时就一定是极小值点。

**由此推知一阶优化条件**：
$$
\frac{\partial \sum_{i=1}^{N} \hat{u}_i^2}{\partial \hat{\beta}_k} = 0
$$

其中 $k = 0, 1$

由一阶优化条件得到正规方程组

- $$
  \sum_{i=1}^{N} \hat{u}_i = 0
  $$
- $$
  \sum_{i=1}^{N} X_i \hat{u}_i = 0
  $$
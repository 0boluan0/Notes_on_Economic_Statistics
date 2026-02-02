---
aliases:
- 马哈拉诺比斯距离
tags:
- concept
- multivariate statistics
---
# 马哈拉诺比斯距离

## 定义

马哈拉诺比斯距离是考虑变量相关性和不同尺度后，衡量观测值与均值之间距离的度量。

对于随机向量 $X \sim N_p(\mu, \Sigma)$：

$$ D^2 = (X - \mu)'\Sigma^{-1}(X - \mu) $$

## 几何意义

### 加权距离
- 普通欧氏距离：$||X - \mu||^2 = (X - \mu)'(X - \mu)$
- 马哈拉诺比斯距离：$(X - \mu)'\Sigma^{-1}(X - \mu)$

协方差矩阵 $\Sigma^{-1}$ 充当了"权重矩阵"，考虑了：
1. **变量尺度**：不同变量的方差不同
2. **变量相关性**：变量之间的协方差关系

### 椭球距离
在多元正态分布的等概率密度曲线上：
- 固定 $D^2 = c^2$ 得到的是一个椭圆（或椭球）
- 椭圆形状由 $\Sigma$ 决定

## 统计性质

### 1. 服从卡方分布

如果 $X \sim N_p(\mu, \Sigma)$，则：

$$ D^2 \sim \chi^2_p $$

这是马哈拉诺比斯距离最重要的性质。

### 2. 标准化
可以通过标准化将其转化为标准正态距离：

$$ Z = \Sigma^{-1/2}(X - \mu) $$
$$ D^2 = Z'Z = ||Z||^2 $$

## 应用

### 1. 异常值检测
- 计算每个观测的马哈拉诺比斯距离
- 如果 $D^2 > \chi^2_{p,\alpha}$，则认为是异常值

### 2. 多元正态性检验
- 将每个观测的 $D^2$ 按大小排序
- 与 $\chi^2_p$ 分布的分位数比较
- 绘制 $\chi^2$ QQ 图检验正态性

### 3. 分类和判别
- 在判别分析中用作距离度量
- 用于确定观测属于哪个类别

### 4. 置信椭球构造
- 构造均值的置信区域
- 椭球边界由 $D^2 = \chi^2_{p,\alpha}$ 确定

## 与欧氏距离的比较

| 特征 | 欧氏距离 | 马哈拉诺比斯距离 |
|------|---------|-----------------|
| 公式 | $(x-y)'(x-y)$ | $(x-y)'\Sigma^{-1}(x-y)$ |
| 考虑尺度 | 否 | 是 |
| 考虑相关性 | 否 | 是 |
| 分布假设 | 无 | 假设多元正态 |

## 计算示例

### 二元情形
对于 $X = (X_1, X_2)' \sim N_2(\mu, \Sigma)$：

$$ \Sigma = \begin{pmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2 \end{pmatrix} $$

$$ \Sigma^{-1} = \frac{1}{1-\rho^2} \begin{pmatrix} \frac{1}{\sigma_1^2} & -\frac{\rho}{\sigma_1\sigma_2} \\ -\frac{\rho}{\sigma_1\sigma_2} & \frac{1}{\sigma_2^2} \end{pmatrix} $$

$$ D^2 = \frac{1}{1-\rho^2}\left[\frac{(X_1-\mu_1)^2}{\sigma_1^2} + \frac{(X_2-\mu_2)^2}{\sigma_2^2} - \frac{2\rho(X_1-\mu_1)(X_2-\mu_2)}{\sigma_1\sigma_2}\right] $$

## 相关概念

- [[00_factor/concept/Multivariate Normal Distribution|多元正态分布]]
- [[00_factor/concept/Wishart Distribution|Wishart 分布]]
- [[Hotelling T2 Test|Hotelling T² 检验]]

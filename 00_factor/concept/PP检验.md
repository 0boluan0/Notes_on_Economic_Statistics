---
aliases:
  - Phillips-Perron Test
  - 菲利普斯-佩龙检验
tags:
  - 计量经济学
  - 时间序列
  - 单位根检验
---

PP检验（Phillips-Perron Test）是Phillips和Perr提出的单位根检验方法，通过直接修正DF检验统计量的标准误来处理自相关和异方差问题。

## 基本思想

### 与ADF检验的对比

- **ADF检验**：通过在回归中加入滞后项消除自相关
- **PP检验**：不加入滞后项，直接修正DF统计量的标准误

### 非参数化方法

PP检验采用非参数化方法：
- 不设定误差项的特定形式
- 使用长方差估计修正标准误
- 更一般化的处理方式

## 检验形式

### 三种形式（与DF检验相同）

#### 1. 无常数项和趋势项

$\Delta y_t = \gamma y_{t-1} + \epsilon_t$

#### 2. 有常数项，无趋势项

$\Delta y_t = \alpha + \gamma y_{t-1} + \epsilon_t$

#### 3. 有常数项和趋势项

$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \epsilon_t$

原假设：$H_0: \gamma = 0$（存在单位根）
备择假设：$H_1: \gamma < 0$（平稳）

## 统计量的修正

### 1. 基本DF统计量

估计基本DF回归，得到：
- $\hat{\gamma}$：$\gamma$的OLS估计量
- $SE(\hat{\gamma})$：$\hat{\gamma}$的OLS标准误

计算t统计量：
$t_{\gamma} = \frac{\hat{\gamma}}{SE(\hat{\gamma})}$

### 2. 修正标准误

PP检验修正标准误以考虑自相关和异方差：

$SE(\hat{\gamma})_{\text{corrected}} = \sqrt{\frac{\hat{\sigma}^2_{\text{long}}}{\sum_{t=1}^T y_{t-1}^2}}$

其中$\hat{\sigma}^2_{\text{long}}$是长方差估计。

### 长方差估计

使用Newey-West方法：

$\hat{\sigma}^2_{\text{long}} = \hat{\sigma}^2 \left[1 + 2\sum_{j=1}^l w_j \hat{\rho}_j\right]$

其中：
- $\hat{\sigma}^2$是误差项方差的OLS估计
- $\hat{\rho}_j$是误差项的j阶自相关系数
- $w_j$是权重（通常用Bartlett权重）
- $l$是截断滞后

### 3. 修正统计量

#### Z_alpha统计量

$Z_{\alpha} = T \hat{\gamma} - \frac{1}{2}\left[T^2 \frac{SE(\hat{\gamma})_{\text{corrected}}^2 - SE(\hat{\gamma})^2}{\hat{\sigma}^2}\right]$

#### Z_t统计量

$Z_t = \frac{\hat{\gamma}}{SE(\hat{\gamma})_{\text{corrected}}} - \frac{1}{2}\left[\frac{T^2 \cdot SE(\hat{\gamma})_{\text{corrected}}^2 - \sum_{t=1}^T \hat{\epsilon}_t^2}{\sum_{t=1}^T y_{t-1}^2}\right]\frac{SE(\hat{\gamma})_{\text{corrected}}}{SE(\hat{\gamma})}$

## 检验步骤

### 标准流程

1. **选择检验形式**
   - 与ADF检验相同
   - 根据数据特征选择

2. **估计基本DF回归**
   - OLS估计DF方程
   - 得到$\hat{\gamma}$和残差

3. **计算长方差**
   - 使用Newey-West方法
   - 选择适当的截断滞后

4. **修正统计量**
   - 计算$Z_{\alpha}$或$Z_t$

5. **比较临界值**
   - 使用PP检验临界值表
   - 与DF临界值类似

6. **做出判断**
   - 若统计量 < 临界值，拒绝原假设（平稳）
   - 若统计量 > 临界值，不能拒绝原假设（非平稳）

## 与ADF检验的比较

| 特征 | ADF检验 | PP检验 |
|------|----------|---------|
| 处理方法 | 参数化（加滞后项） | 非参数化（修正标准误） |
| 滞后阶数 | 需要选择 | 不需要选择 |
| 模型复杂度 | 较复杂 | 相对简单 |
| 误差项假设 | 设定AR形式 | 不设定特定形式 |
| 适用性 | 大多数时间序列 | 异方差严重时 |
| 临界值 | ADF临界值表 | 类似DF临界值 |

## 优缺点

### 优点

1. **不需要选择滞后阶数**
   - 避免滞后阶数选择困难
   - 减少主观性

2. **处理异方差**
   - 对异方差问题更稳健
   - 适用于异方差严重的数据

3. **非参数化**
   - 不假设误差项特定形式
   - 更一般化

### 缺点

1. **小样本性质**
   - 小样本下性质可能不如ADF
   - 截断滞后选择困难

2. **截断滞后**
   - 仍需选择截断滞后
   - 影响检验结果

3. **理论支持**
   - ADF检验理论基础更完善
   - 文献中使用更广泛

## 截断滞后的选择

### 方法

1. **经验法则**
   - $l = \text{int}(4(T/100)^{1/4})$
   - $l = \text{int}(12(T/100)^{1/4})$

2. **数据驱动**
   - 根据自相关情况调整
   - 考虑样本量

### 原则

- 不能太大：损失太多自由度
- 不能太小：无法充分捕捉自相关

## 应用场景

### 1. 异方差数据

- 误差项方差随时间变化
- PP检验更合适

### 2. 复杂自相关

- 自相关结构复杂
- 难以用AR(p)捕捉

### 3. 大样本

- 大样本下PP检验表现良好
- 非参数化方法优势明显

## 软件实现

### 主要软件

大多数计量经济学软件都提供PP检验：
- Stata：pperron命令
- EViews：Unit Root Test → Phillips-Perron
- R：ur.pp函数
- Python：statsmodels.tsa.stattools PhillipsPerron

### 输出内容

- Z_alpha统计量
- Z_t统计量
- p值
- 滞后截断
- 检验形式

## 实践建议

### 1. 同时使用ADF和PP检验

比较两个检验结果：
- 如果结果一致，结论更可靠
- 如果结果不一致，需要谨慎分析

### 2. 考虑数据特征

- 观察数据图形
- 检查异方差和自相关
- 选择合适检验

### 3. 报告结果

在实证研究中：
- 报告使用的检验方法
- 报告检验形式
- 报告关键参数（滞后阶数、截断等）

相关链接: [[ADF检验]], [[单位根检验]], [[伪回归]]

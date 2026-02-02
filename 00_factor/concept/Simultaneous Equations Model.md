---
aliases:
- 联立方程模型
- Simultaneous Equations Model
- Simultaneous
tags:
- 计量经济学
- 计量方法
- 经济
- concept
---
联立方程模型（Simultaneous Equations Model, SEM）是指由多个相互关联的方程组成的计量经济模型，其中某些变量在多个方程中同时被决定。

## 基本结构

一个包含m个方程的SEM：

$$y_1 = \beta_{12}y_2 + \gamma_{11}x_1 + \cdots + \gamma_{1k}x_k + \varepsilon_1$$
$$y_2 = \beta_{21}y_1 + \gamma_{21}x_1 + \cdots + \gamma_{2k}x_k + \varepsilon_2$$
$$\vdots$$
$$y_m = \beta_{m1}y_1 + \cdots + \gamma_{m1}x_1 + \cdots + \gamma_{mk}x_k + \varepsilon_m$$

其中：
- y₁, ..., y_m：内生变量（m个）
- x₁, ..., x_k：外生变量（k个）
- β_ij：内生变量系数
- γ_ij：外生变量系数
- ε_i：误差项

## 内生性与外生性

### 内生变量
- 在系统中由模型决定
- 与误差项相关
- 例如：供需模型中的价格和数量

### 外生变量
- 在系统外决定
- 与误差项不相关
- 例如：供需模型中的收入、替代品价格

### 前定变量
- 包括外生变量和内生变量的滞后项
- 与当前误差项不相关

## 内生性问题

在联立方程中，内生变量与误差项相关，导致OLS估计有偏且不一致。

示例：简单供需模型

需求：$Q = \alpha_0 + \alpha_1 P + \alpha_2 Y + \varepsilon_D$
供给：$Q = \beta_0 + \beta_1 P + \beta_2 W + \varepsilon_S$

其中P和Q是内生变量，Y和W是外生变量。

## 识别条件

### 阶条件（必要条件）

方程可识别的必要条件：

$$k \geq m-1$$

其中k是方程中不包含的外生变量数，m是内生变量总数。

- k = m-1：恰好识别
- k > m-1：过度识别
- k < m-1：不可识别

### 秩条件（充分条件）

构造包含所有外生变量的矩阵，计算其秩。

**秩条件**：秩 = m-1（充要条件）

## 估计方法

### 1. ILS（间接最小二乘法）

适用于恰好识别的方程：
1. 求出内生变量的简化式（Reduced Form）
2. 用OLS估计简化式
3. 导出结构式参数

### 2. 2SLS（两阶段最小二乘法）

适用于恰好识别或过度识别：
1. 第一阶段：用外生变量对内生变量回归，得到预测值
2. 第二阶段：用预测值替代内生变量进行OLS

### 3. 3SLS（三阶段最小二乘法）

考虑误差项协方差矩阵，更有效但计算复杂。

### 4. FIML（完全信息最大似然）

同时估计所有方程和误差协方差矩阵。

### 5. LIML（有限信息最大似然）

逐个方程估计，更稳健。

## 简化式与结构式

### 简化式（Reduced Form）

$$y_i = \pi_{i1}x_1 + \cdots + \pi_{ik}x_k + v_i$$

内生变量仅表示为外生变量的函数。

### 结构式（Structural Form）

包含经济理论关系的原始方程形式。

## 应用

1. **宏观经济模型**：IS-LM模型、AD-AS模型
2. **市场分析**：供需模型、竞争均衡
3. **政策模拟**：分析政策变化的影响

## 软件实现

- Stata：reg3, sureg命令
- R：systemfit包
- EViews：System对象

相关链接: [[00_factor/concept/Endogeneity|内生性]], [[00_factor/concept/Instrumental Variable|工具变量]], [[2SLS]]

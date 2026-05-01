---
aliases:
- Macaulay Duration Calculation
- 马考利久期计算
- 麦考利久期计算
tags:
- procedure
- fixed-income
---
# Macaulay Duration Calculation

## 这张卡什么时候用

给出债券现金流、到期收益率和付息频率，要求计算 Macaulay duration 时使用。定义见 [[Macaulay Duration]]。

## 输入

- 每期现金流 $CF_t$。
- 年化到期收益率 $y$。
- 每年付息次数 $m$。
- 期数 $n$。

## 输出

- 债券价格 $P$。
- Macaulay duration $D_M$，通常以年为单位。

## Step 1：列现金流

按每个付息周期列出 $CF_1,\dots,CF_n$。最后一期通常包含票息和本金。

## Step 2：折现现金流

周期收益率为 $y/m$：

$$
PV_t=\frac{CF_t}{(1+y/m)^t}
$$

价格为：

$$
P=\sum_{t=1}^{n}PV_t
$$

## Step 3：计算现值权重

$$
w_t=\frac{PV_t}{P}
$$

检查所有权重应加总为 1。

## Step 4：计算加权平均时间

周期口径：

$$
D_M^{period}=\sum_{t=1}^{n}t w_t
$$

年口径：

$$
D_M=\frac{D_M^{period}}{m}
$$

## 检查点

- 零息债券的 Macaulay duration 等于剩余期限。
- 附息债券的 duration 通常小于到期期限。
- 现金流周期和收益率周期必须一致。

## 常见错误

- 忘记最后一期本金。
- 半年付息却直接用年收益率折现。
- 把 Macaulay duration 直接用于价格百分比变化；价格敏感度见 [[Modified Duration Calculation]]。

## 来自课程位置

- [[09_利率风险]]

## 关联卡片

- [[Macaulay Duration]]
- [[Modified Duration]]
- [[Convexity]]

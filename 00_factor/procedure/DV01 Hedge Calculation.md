---
aliases:
- DV01 Hedge Calculation
- BPV Hedge Calculation
- DV01对冲计算
- 基点价值对冲计算
tags:
- procedure
- fixed-income
- risk-management
---
# DV01 Hedge Calculation

## 这张卡什么时候用

题目给出组合每 1bp 利率变化的损益，或给出对冲工具的 DV01/BPV，要求算多少期货、互换名义本金或债券头寸来对冲时使用。

## 输入

- 被对冲组合的 DV01，记为 $DV01_P$。
- 对冲工具单份或单位名义本金的 DV01，记为 $DV01_H$。
- 头寸方向：利率上升时组合是亏还是赚。

## 输出

- 对冲工具数量或名义本金 $N$。
- 对冲方向：买入、卖出、收固定、付固定等。

## Step 1：统一符号

先规定利率上升 1bp 时的价值变化为 DV01 符号。

- 若组合利率上升亏损，则 $DV01_P<0$。
- 对冲头寸应使总 DV01 接近 0。

## Step 2：写出中性条件

$$
DV01_P+N\cdot DV01_H=0
$$

所以：

$$
N=-\frac{DV01_P}{DV01_H}
$$

## Step 3：判断方向

如果 $N>0$，持有题目定义下的正向对冲工具；如果 $N<0$，做相反方向。

互换题里要把方向翻译成“收固定/付固定”。固定利率债券式现金流通常在利率上升时价值下降；想对冲利率上升损失，需要持有利率上升时盈利的方向。

## Step 4：检查剩余风险

DV01 对冲只中和 1bp 小幅平行移动的一阶风险。还需要检查：

- [[Convexity]]：大幅利率变化的二阶风险。
- [[Key Rate Duration]]：曲线非平行移动。
- [[Basis Risk]]：对冲工具和被对冲头寸参考利率不同。

## 常见错误

- 只匹配绝对值，不判断头寸方向。
- 把 DV01 当成每 1% 利率变化金额。
- 用单一期限工具对冲整条收益率曲线敞口。

## 来自课程位置

- [[09_利率风险]]

## 关联卡片

- [[Basis Point Value (BPV)]]
- [[Dollar Duration]]
- [[Modified Duration]]
- [[Yield Curve Risk]]

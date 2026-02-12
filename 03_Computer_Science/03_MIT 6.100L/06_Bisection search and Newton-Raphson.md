---
date: 2026-02-13
科目: MIT 6.100L
---

# Bisection search and Newton-Raphson

## 本讲主线

- 为什么“只会枚举”不够快，必须优化 guess 的生成方式
- 二分查找如何利用有序性与方向反馈，把搜索空间每次砍半
- Newton-Raphson 如何用函数斜率快速修正 guess，实现更快收敛

## 上节回顾（与本讲的连接）

在 [[05_Floats and approximation methods]] 里，我们已经得到两个关键结论：

- 很多问题无法得到“精确可表示”的答案，只能追求 close enough
- 浮点运算存在表示误差，所以判断条件通常是 `abs(error) < epsilon`，而不是 `==`

本讲继续做同一件事：  
**找近似解**，但改进“猜测策略”，让算法从“能用”变成“高效可扩展”。

## 1. 为什么需要更聪明的搜索

### 1.1 近似法的问题不是“对不对”，而是“慢不慢”

例如平方根问题：

- 近似法（每次 `guess += increment`）可以最终达到 epsilon 要求
- 但步长越小，循环次数越大
- 为了精度而减小步长，往往会把时间成本推高到不可接受

这引出本讲核心：

> 同样是迭代算法，`generate guesses` 的方式决定了速度上限。

### 1.2 课堂例子（猜书页）

如果你只知道“对/错”，最稳妥方法是线性试：1, 2, 3, ...

如果每次还能知道“太大/太小”，就能立即排除一半范围。  
这就是二分查找可用的根本原因：**反馈不仅判断正确性，还提供方向信息**。

## 2. Bisection search（二分查找）核心机制

### 2.1 算法直觉

假设答案在区间 `[low, high]`：

1. 取中点 `guess = (low + high)/2`
2. 如果命中（或误差足够小），结束
3. 如果 guess 偏小，只保留右半区间
4. 如果 guess 偏大，只保留左半区间
5. 重复

每一步都让搜索空间缩小为原来的一半。

### 2.2 何时可以用二分查找

必须满足：

- 搜索空间有序（能比较大小）
- 能判断当前 guess 是偏高还是偏低
- 已知答案落在某个边界区间内

如果只知道“对/错”，却不知道方向（例如 4 位 PIN 码仅返回正确或错误），就不能直接使用二分查找。

## 3. 用二分查找求平方根（标准版）

```python
x = 54321
epsilon = 0.01
num_guesses = 0

low = 0.0
high = x
guess = (high + low) / 2.0

while abs(guess**2 - x) >= epsilon:
    if guess**2 < x:
        low = guess
    else:
        high = guess

    guess = (high + low) / 2.0
    num_guesses += 1

print('num_guesses =', num_guesses)
print(guess, 'is close to square root of', x)
```

### 3.1 这段代码的关键不变量

- `guess` 始终位于 `[low, high]` 内部
- 如果 `guess**2 < x`，真正答案一定在 `[guess, high]`
- 如果 `guess**2 > x`，真正答案一定在 `[low, guess]`

因此每轮都缩区间，但不会丢掉答案。

### 3.2 停止条件为什么是误差而不是相等

- `guess` 是浮点数
- 目标通常无法精确表示
- 所以应以 `abs(guess**2 - x) < epsilon` 作为“足够好”

这和上一讲的浮点误差认知完全一致。

## 4. 复杂度：为什么它快很多

设初始搜索空间大小为 `N`：

- 第 1 步后变为 `N/2`
- 第 2 步后变为 `N/4`
- 第 `k` 步后变为 `N/2^k`

当只剩一个候选量级时，`N/2^k ≈ 1`，可得 `k ≈ log2(N)`。

> 二分查找的步数增长是对数级；  
> 线性枚举的步数增长是线性级。

课件给出的对比结论：  
对 `x = 54321`，暴力近似可能达到千万级猜测；二分大约几十次即可收敛。

## 5. 边界场景：`0 < x < 1` 的修正

### 5.1 原代码为什么会出错

若 `x = 0.5` 且仍设 `low = 0, high = x`，则搜索区间是 `[0, 0.5]`。  
但 `sqrt(0.5) ≈ 0.707...`，真实答案不在区间内，算法不会正确收敛到目标附近。

### 5.2 全正数版本（推荐模板）

```python
x = 0.5
epsilon = 0.01

if x >= 1:
    low = 1.0
    high = x
else:
    low = x
    high = 1.0

guess = (high + low) / 2.0

while abs(guess**2 - x) >= epsilon:
    if guess**2 < x:
        low = guess
    else:
        high = guess
    guess = (high + low) / 2.0

print(f'{guess} is close to square root of {x}')
```

### 5.3 适用范围说明

- 该版本覆盖全部正实数 `x > 0`
- `x = 0` 可直接返回 `0`
- 负数在实数范围无平方根，本讲不处理复数情形

## 6. 迁移：用 bisection 求立方根

思路不变，只改目标函数：

- 平方根：比较 `guess**2` 与 `x`
- 立方根：比较 `guess**3` 与 `cube`

```python
cube = 27.0
epsilon = 0.01
low = 0.0
high = cube
guess = (low + high) / 2.0

while abs(guess**3 - cube) >= epsilon:
    if guess**3 < cube:
        low = guess
    else:
        high = guess
    guess = (low + high) / 2.0

print(guess, 'is close to cube root of', cube)
```

要点：二分查找是“区间缩减框架”，可复用于多种单调目标函数。

## 7. Newton-Raphson：更激进的收敛策略

### 7.1 思想来源

求 `k` 的平方根，可转成求方程 `p(x) = x^2 - k = 0` 的根。  
Newton-Raphson 用当前点的切线来近似函数，并把切线与 x 轴交点作为下一次 guess。

对 `p(x) = x^2 - k`：

- `p'(x) = 2x`
- 迭代式为  
  `g <- g - p(g)/p'(g)`  
  `g <- g - (g^2 - k)/(2g)`

### 7.2 代码实现（平方根场景）

```python
epsilon = 0.01
k = 24.0
guess = k / 2.0
num_guesses = 0

while abs(guess**2 - k) >= epsilon:
    num_guesses += 1
    guess = guess - (((guess**2) - k) / (2 * guess))

print('num_guesses =', num_guesses)
print('Square root of', k, 'is about', guess)
```

### 7.3 与二分查找的区别

- 二分查找依赖“区间 + 单调方向”
- Newton-Raphson 依赖“函数可导 + 导数信息”
- 在条件合适时，Newton-Raphson 往往更快，但对初值与函数形态更敏感

## 8. 四种迭代找根方法对比

| 方法 | 核心做法 | 需要有序区间 | 需要导数 | 收敛速度直觉 | 典型风险 |
|---|---|---|---|---|---|
| Guess-and-check | 一个个试候选 | 否 | 否 | 慢（线性） | 搜索空间大时不可用 |
| Approximation increment | 固定步长逼近 | 否 | 否 | 依赖步长，常偏慢 | 精度与速度强耦合 |
| Bisection search | 每轮砍半区间 | 是 | 否 | 快（对数） | 前提不满足时失效 |
| Newton-Raphson | 用切线更新 guess | 否（通常） | 是 | 常常非常快 | 初值差或导数异常可能不稳 |

## 9. 课堂练习（You Try It）

- [ ] 用二分查找计算 `sqrt(2)`，并统计 `num_guesses`
- [ ] 测试 `x = 0.25`，确认 `0 < x < 1` 修正版能正常收敛
- [ ] 将平方根代码改为立方根版本，测试 `cube = 125`
- [ ] 用 Newton-Raphson 求 `sqrt(24)`，与二分查找的迭代次数比较
- [ ] 思考：为什么 PIN 码问题无法直接二分？

## 小结

- 近似算法的效率核心在“如何生成下一次 guess”
- 二分查找通过“方向反馈 + 区间砍半”把复杂度降为 `O(log N)`
- Newton-Raphson 通过导数信息做更聪明更新，通常收敛更快
- 浮点问题里永远要把“误差阈值 + 停止条件”作为算法的一部分

下一讲将进入 decomposition 与 abstraction（把程序拆分成可复用部件，并隐藏不必要细节）。

## 资料

- ![[MIT 6.100L-slides/mit6_100l_lec06.pdf]]

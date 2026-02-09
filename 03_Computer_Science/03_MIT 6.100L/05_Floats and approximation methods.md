---
date: 2026-02-09
科目: MIT 6.100L
---

# Floats, approximation methods, and early numerical thinking

## 本讲主线

- 浮点数表示及其误差源头
- 算法参数：增量与 epsilon 的权衡
- 从穷举走向 bisection/newton 的分步优化

## 1. 浮点数为什么不精确

- 0.1 无法在有限二进制位精确表示；单次加法会留下小误差
- 任何用 == 比较 floats 都是危险的

```python
x = 0
for _ in range(10):
    x += 0.1

print(x == 1)     # False
print(x, 10 * 0.1)
```

> [!important] Big Idea
> 浮点是“近似表示的实数”，多次运算会放大微小误差，比较时应使用“近似相等”而非 ==。

### 实数与二进制

- 小数部分可以写成一串 `a*2^{-1} + b*2^{-2} + ...`
- 整数部分的转换：持续 `% 2` 取最低位并用 `// 2` 缩减
- 小数部分需要乘以 2 的幂再除回去才能恢复，但不总能得到整数

## 2. Approximation methods（近似算法）

- 不再寻找精确答案，而是设定 `epsilon`，只要 `|guess**2 - x| < epsilon` 就收手
- 需要两个参数：`epsilon` 决定精度，`increment` 决定搜索步长
- 减小 `increment` 会明显降低性能；增大 `epsilon` 会牺牲精度

```python
x = 36
epsilon = 0.01
increment = 0.0001
guess = 0.0
while abs(guess**2 - x) >= epsilon and guess**2 <= x:
    guess += increment
print(guess, 'is close to', x)
```

- 还需要防止“跳过”近似区间，否则可能永远无法跳出 while
- 增加 `guess**2 <= x` 条件可以阻止过冲，并能报告失败

> [!tip] Big Idea
> 近似算法是“guess-and-check + float increment”，但无法“检查是否正确”，只能检查是否“足够接近”。

## 3. “好”算法：bisection、Newton-Raphson

### Bisection search

- 需要：有序搜索区间 + 能告诉你猜测是偏大还是偏小
- 每次猜 midpoint，成功则终止；否则根据反馈缩小区间到一半
- 复杂度从线性降为 `O(log N)`，比如 `√54321` 从 23M 次降到 30 次

```python
low = 0
high = x if x >= 1 else 1
guess = (high + low) / 2.0
while abs(guess**2 - x) >= epsilon:
    if guess**2 < x:
        low = guess
    else:
        high = guess
    guess = (high + low) / 2.0
```

- `x < 1` 时要调整初始 `[low, high]`

### Newton-Raphson

- 通用 root finding：`guess <- guess - (guess**2 - k)/(2*guess)`
- 迭代次数通常远低于 bisection，但需要先验导数公式

## 4. 课堂练习（You Try It）

- [ ] 用 `epsilon` + `increment` 写一个“good enough” 的平方根，测试 `12345`、`54321`
- [ ] 实现能早早停止的 `while abs(...) < epsilon and guess**2 <= x` 版，并观察 `guess` 数量
- [ ] 比较 `increment=0.0001` 与 `0.00001` 的运行速度/命中情况
- [ ] 用 `while` 实现 `num % 2` 还是 `range` 选择判定奇数
- [ ] “Never use == ”：设计一个条件测试 0.1 的多次累加

## 小结

- 浮点数是有限位的近似，千万别用 == ；用 `abs(a - b) < epsilon`
- 近似算法需要调两个参数；减小 `increment` 或 `epsilon` 的代价是速度
- Bisection/newton 通过 smarter guess 减少迭代次数

## 资料

- ![[MIT 6.100L-slides/mit6_100l_lec05.pdf]]

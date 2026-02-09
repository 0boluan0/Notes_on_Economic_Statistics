---
date: 2026-02-10
科目: MIT 6.100L
---

# Bisection search, Newton-Raphson, and the next steps in abstraction

## 本讲主线

- 从 approximation 提升到 bisection，从线性搜到对数缩减
- 稳定控制搜索区间：`low`/`high` 与 `epsilon` 的交互
- 引入 Newton-Raphson 作为更聪明的 guess generator

## 1. Bisection search

- 适用于有“端点 + 有序” + 能反馈“太大/太小”的问题
- 每次从 `[low, high]` 取 midpoint；若 `guess**2 < x`，`low=guess`；否则 `high=guess`
- 保证每轮至少减半搜索空间，迭代次数为 `O(log N)`

```python
low = 0 if x >= 1 else x
high = x if x >= 1 else 1
guess = (low + high)/2.0
while abs(guess**2 - x) >= epsilon:
    if guess**2 < x:
        low = guess
    else:
        high = guess
    guess = (low + high)/2.0
```

- `x < 1` 时要把 `low` 设置为 `x`（因为根在 `(x, 1]`）
- 计算量从 `N` 变成 `log2 N`，例如 `√54321` 由 23M 次变 30 次

> [!important] Big Idea
> 精心生成猜测可以把线性搜索降到对数级，是算法中的常见跃迁。

### 练习题

- 三位 PIN：只能知道“是否正确”。能否用 bisection 快速缩小解空间？
- 极端猜数：友人想一个 0-10 的十进制，告诉你“高/低/对”，你如何用对数步猜中？

## 2. approximate search vs bisection

- `while abs(...) >= epsilon` 与 `guess**2 <= x` 组合，防止 jump over epsilon
- 线性搜索在 `x=54321` 上需要百万级猜测；bisection 所需猜测是 30
- `num_guesses` 变量可以帮助对比效率

## 3. Newton-Raphson（牛顿法）

- 目标：find root of `x^2 - k`
- 迭代公式：`guess ← guess - ((guess**2 - k)/(2*guess))`
- 通常收敛更快，比 bisection 更聪明，但依赖导数

```python
while abs(guess**2 - k) >= epsilon:
    guess = guess - (((guess**2) - k)/(2*guess))
```

- Newton-Raphson 属于“迭代算法”家族：exhaustive, approximation, bisection, Newton

## 4. 课堂练习（You Try It）

- [ ] 用 bisection 写 `cube root` 的近似算法（`low=0`, `high=cube`, `epsilon` 可调）
- [ ] 把 `x < 1` 也支持进去：一个程序根据 `x` 选择 `[low, high]`
- [ ] 计算 `abs(guess**2 - x)` 每 100k 步打印，观察 overshoot 何时发生
- [ ] 写 Newton-Raphson 版本，记录 `num_guesses`
- [ ] 比较线性、bisection、Newton 在同一个 `x` 上的速度差别

## 5. 抽象与分解的预告

- 计算机科学讲“黑盒 + 接口”：将复杂系统拆成自包含部分
- 一个函数就是黑盒：通过 `docstring` 描述输入/输出
- 抽象让用户只关心“输入→输出”，不必理会内部细节
- 分解则用函数组织大程序（也引出下一讲的 `sum_odd` 例子）

## 小结

- Bisection 需要 ordering/feedback，能把线性搜索变为 log
- Newton-Raphson 是智能猜测生成器，利用导数调整
- 任何迭代算法都是：生成 guess → 检查 → 继续／返回
- 通过抽象与分解，我们得以构建可管理的大规模程序

## 资料

- ![[MIT 6.100L-slides/mit6_100l_lec06.pdf]]

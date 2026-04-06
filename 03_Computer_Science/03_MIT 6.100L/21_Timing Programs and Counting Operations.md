---
aliases:
  - MIT 6.100L Lecture 21
  - 6.100L L21
  - Timing Programs and Counting Operations
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 21
---

# Lecture 21: Timing Programs and Counting Operations

> [!tip] Hint
> - 这节课开头先从“程序不仅要对，还要快”切换课程目标，复杂度单元正式开始。
> - correctness 之前一直是主角，但现在老师要你开始关心 efficiency。
> - time efficiency 和 space efficiency 往往有 tradeoff，Fibonacci memoization 被拿来做第一例。
> - 第一种测效率的方法是直接 timing，看看程序在不同输入规模上要多久。
> - `c_to_f`、`mysum`、`square` 是三类代表：constant、linear、quadratic。
> - timing 很直观，但会受机器状态、实现细节和小样本误差影响。
> - 第二种方法是 counting operations，把代码里关键操作一项项数出来。
> - counting 比 timing 更抽象，但更稳定，也更靠近后面要讲的复杂度理论。
> - 老师这讲一直在让你比较“输入扩大 10 倍时，时间或操作数大约扩大多少倍”。
> - 听完这节课，你应该知道为什么“跑起来快不快”不能只靠一次 timing 结果下结论。

## Lecture flow

### 1. 课程目标从 correctness 转向 efficiency
Lecture 21 一开始就说明要换挡。

前面 problem sets 和 quizzes 重点一直是：

- 程序是否正确

但现实里的程序通常还得考虑：

- 时间上够不够快
- 空间上占不占资源

这节课因此正式开启复杂度单元，开始讨论 how efficient our programs are。

### 2. 先讲 time vs space tradeoff
老师先没有直接上 timing，而是先给一个动机：  
程序效率常常不是单维度的。

最典型例子就是前面见过的 Fibonacci：

- naive recursion：省空间，但会做海量重复工作
- memoization：额外占用字典空间，但速度快很多

这让课程一开始就把一个很重要的观念钉住：

> [!note]
> 更快和更省内存往往不能同时极致；很多算法是在时间和空间之间做取舍。

### 3. 第一条路线：直接 timing 程序
接着老师进入第一种评估效率的方法：  
直接计时。

代码里用到 `time.time()`，大致做法是：

1. 记录开始时间
2. 运行函数
3. 记录结束时间
4. 相减得到耗时

这在直觉上最容易理解，因为你直接看到“这段代码花了多久”。

### 4. 用三类函数建立 timing 直觉
老师故意选了三种非常不同的函数：

- `c_to_f(c)`：常数时间
- `mysum(x)`：线性时间
- `square(n)`：双重循环，近似二次时间

然后让输入规模按：

- `1`
- `10`
- `100`
- `1000`

这样逐步扩大，观察 timing 的变化。

课堂此时在训练的不是精准测量，而是量级直觉：

- 常数函数几乎不怎么变
- 线性函数增长比较平稳
- 二次函数会很快变得难以忍受

### 5. timing 的长处：真实、直观
老师并没有一开始就批 timing，而是先承认它的优点。

timing 的好处包括：

- 贴近真实运行环境
- 不需要先抽象出理论模型
- 对新手特别直观

如果你只是粗略比较两个实现，timing 常常是第一步。

### 6. timing 的局限：噪声很多，解释不稳定
但老师很快也指出 timing 的局限。

例如：

- 机器当前负载
- Python 解释器本身的细节
- 运行时缓存
- 样本太小导致的测量误差

都会让 timing 结果抖动。

所以：

- 很小的输入规模上，timing 可能几乎看不出区别
- 同一函数多跑几次，也可能拿到不同时间

这就引出第二条路线。

### 7. 第二条路线：counting operations
老师随后转到 counting operations。

思想很简单：

- 不直接问“花了多少秒”
- 而是问“执行了多少基本操作”

这样做的好处是：

- 更抽象
- 更稳定
- 不那么依赖机器和环境噪声

当然代价是你得自己建模“哪些东西算一次操作”。

### 8. 从常数函数开始数：`c_to_f`
以最简单的 `c_to_f(c)` 为例，老师把几步乘法、除法、加法粗略记成固定个数操作。

于是无论输入 `c` 是多少，操作数都差不多保持不变。

这一点和 timing 对应起来，就形成了第一个复杂度直觉：

- 不是所有函数的运行成本都随输入值本身变化

### 9. 线性例子：`mysum(x)`
对于：

```python
def mysum(x):
    total = 0
    for i in range(x + 1):
        total += i
```

老师数操作时会把：

- 初始化
- 每轮循环中的赋值、加法、比较

都记进去。

虽然具体常数项可以不同，但核心现象很清楚：

- 输入 `x` 扩大大约 10 倍
- 操作数也会大约扩大 10 倍

这就是线性增长。

### 10. 二次例子：`square(n)`
对双重循环的 `square(n)` 来说，情况就明显不同了。

```python
for i in range(n):
    for j in range(n):
        ...
```

这里老师反复问的是：

- 外层循环跑几次
- 每次外层里，内层又跑几次

所以总操作量大约和 `n * n` 成正比。

这就是为什么当输入扩大 10 倍时：

- 线性函数可能只慢约 10 倍
- 二次函数可能慢约 100 倍

### 11. timing 和 counting 的关系
这节课并不是让你二选一，而是在建立比较框架。

它们的关系可以概括为：

- timing：真实世界表现，直观但 noisy
- counting：抽象模型，更稳定但需要人为设定

老师希望你意识到：

- 仅靠 timing 看一眼，不够稳
- 仅靠 operation counting，不够贴近真实机器

后面的复杂度理论，其实是在 counting 的基础上进一步抽象。

### 12. 课程这时还没正式进入 Big Theta，但地基已经搭好了
Lecture 21 的任务不是正式定义 Big O / Theta。  
它更像是先让你接受：

- 程序效率可以系统比较
- 输入规模变化比单次运行结果更重要
- constant / linear / quadratic 的增长差异是真实会咬人的

这让下节课进入 order of growth 时不会显得凭空抽象。

## Exercise log

> [!warning] No official finger exercise
> 这讲官方没有单独的 finger exercise 文件。

如果要按课堂内容做一个最像 finger exercise 的自测，最合适的是自己完成下面两步：

- 选 `c_to_f`、`mysum`、`square` 三个函数，预测输入扩大 10 倍时 timing 和 ops 分别会怎样变化。
- 不看代码注释，自己为 `mysum` 或 `square` 重做一遍 operation counting。

这两步正好对应本讲的两条主线：

- timing
- counting

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec21.pdf|Lecture 21 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec21_code.py|Lecture 21 code (py)]]
- Finger exercise: no official file for this lecture
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec21_transcript.pdf|Lecture 21 transcript]]
- Recitation: none attached to this lecture week
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 11)

## Review checklist
- [ ] 我能解释为什么课程会从 correctness 转向 efficiency。
- [ ] 我能说明 time efficiency 和 space efficiency 的 tradeoff。
- [ ] 我能用 Fibonacci memoization 解释“拿空间换时间”。
- [ ] 我能说出 timing 程序的基本步骤。
- [ ] 我能比较 `c_to_f`、`mysum`、`square` 在 timing 上的大致增长差异。
- [ ] 我能解释 timing 为什么会 noisy。
- [ ] 我能说明 counting operations 的基本思想。
- [ ] 我能手动数出一个简单循环函数的大致操作数表达式。
- [ ] 我能解释为什么输入扩大 10 倍时，线性和二次增长会表现得很不一样。
- [ ] 我能按课堂顺序复述：motivation -> timing -> limitations -> counting。

> [!warning] Common mistakes
> - 用一次 timing 结果就断定两个程序谁更优。
> - 数操作时把所有常数差异都当成最重要部分。
> - 只看输入值本身，而不看“输入规模”到底该怎么定义。
> - 把时间快慢和空间占用想成永远同步改进。

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
> <!-- bilingual-en:start -->
> - This lecture formally opens the complexity unit by shifting the goal from “a program must be correct” to “a program must also be fast.”
> - Correctness has been the main focus so far, but now the instructor emphasizes efficiency as well.
> - Time efficiency and space efficiency often trade off; Fibonacci memoization is used as the first example.
> - The first method for measuring efficiency is timing, observing how long a program takes with different input sizes.
> - `c_to_f`, `mysum`, and `square` illustrate constant-, linear-, and quadratic-time growth.
> - Timing is intuitive, but machine load, implementation details, and measurement error on small inputs can all affect the result.
> - The second method involves counting operations, tallying key operations in the code.
> - Counting operations is more abstract than timing but more stable and closer to the complexity theory that will follow.
> - Throughout the lecture, the instructor asks what happens to runtime or operation count when the input grows tenfold.
> - By the end of the lecture, you should understand why a single timing result is not enough to decide whether a program is fast.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 课程目标从 correctness 转向 efficiency
<!-- bilingual-en:start -->
*1. Shifting Course Objectives from Correctness to Efficiency*
<!-- bilingual-en:end -->
Lecture 21 一开始就说明要换挡。
<!-- bilingual-en:start -->
Lecture 21 begins with a shift in gears.
<!-- bilingual-en:end -->

前面 problem sets 和 quizzes 重点一直是：
<!-- bilingual-en:start -->
The focus of previous problem sets and quizzes has been on:
<!-- bilingual-en:end -->

- 程序是否正确
<!-- bilingual-en:start -->
- Program correctness
<!-- bilingual-en:end -->

但现实里的程序通常还得考虑：
<!-- bilingual-en:start -->
In practice, however, a program must also consider:
<!-- bilingual-en:end -->

- 时间上够不够快
- 空间上占不占资源
<!-- bilingual-en:start -->
- Whether it runs fast enough
- How much memory it uses
<!-- bilingual-en:end -->

这节课因此正式开启复杂度单元，开始讨论 how efficient our programs are。
<!-- bilingual-en:start -->
This lecture thus formally introduces the complexity unit, beginning a discussion on how efficient our programs are.
<!-- bilingual-en:end -->

### 2. 先讲 time vs space tradeoff
<!-- bilingual-en:start -->
*2. Balancing Time and Space Tradeoffs*
<!-- bilingual-en:end -->
老师先没有直接上 timing，而是先给一个动机：  
程序效率常常不是单维度的。
<!-- bilingual-en:start -->
Rather than jumping straight into timing, the instructor first establishes the motivation:
Program efficiency is rarely one-dimensional.
<!-- bilingual-en:end -->

最典型例子就是前面见过的 Fibonacci：
<!-- bilingual-en:start -->
The familiar Fibonacci example makes this tradeoff concrete:
<!-- bilingual-en:end -->

- naive recursion：省空间，但会做海量重复工作
- memoization：额外占用字典空间，但速度快很多
<!-- bilingual-en:start -->
- Naive recursion: saves space but performs a great deal of redundant work
- Memoization: uses extra dictionary space but runs much faster
<!-- bilingual-en:end -->

这让课程一开始就把一个很重要的观念钉住：
<!-- bilingual-en:start -->
This drives home an important concept early in the course:
<!-- bilingual-en:end -->

> [!note]
> 更快和更省内存往往不能同时极致；很多算法是在时间和空间之间做取舍。
> <!-- bilingual-en:start -->
> Speed and memory efficiency usually cannot both be maximized; many algorithms trade one for the other.
> <!-- bilingual-en:end -->

### 3. 第一条路线：直接 timing 程序
<!-- bilingual-en:start -->
*3. First Approach: Directly Timing Programs*
<!-- bilingual-en:end -->
接着老师进入第一种评估效率的方法：  
直接计时。
<!-- bilingual-en:start -->
The instructor then introduces the first way to evaluate efficiency:
time the program directly.
<!-- bilingual-en:end -->

代码里用到 `time.time()`，大致做法是：
<!-- bilingual-en:start -->
Using `time.time()`, the general approach is:
<!-- bilingual-en:end -->

1. 记录开始时间
2. 运行函数
3. 记录结束时间
4. 相减得到耗时
<!-- bilingual-en:start -->
1. Record start time
2. Run function
3. Record end time
4. Subtract to get duration
<!-- bilingual-en:end -->

这在直觉上最容易理解，因为你直接看到“这段代码花了多久”。
<!-- bilingual-en:start -->
This is intuitive because it directly tells you “how long this code took.”
<!-- bilingual-en:end -->

### 4. 用三类函数建立 timing 直觉
<!-- bilingual-en:start -->
*4. Building Timing Intuition with Three Types of Functions*
<!-- bilingual-en:end -->
老师故意选了三种非常不同的函数：
<!-- bilingual-en:start -->
The instructor intentionally selects three very different functions:
<!-- bilingual-en:end -->

- `c_to_f(c)`：常数时间
- `mysum(x)`：线性时间
- `square(n)`：双重循环，近似二次时间
<!-- bilingual-en:start -->
- `c_to_f(c)`: Constant time
- `mysum(x)`: Linear time
- `square(n)`: Double loop, approximately quadratic time
<!-- bilingual-en:end -->

然后让输入规模按：
<!-- bilingual-en:start -->
Then run with input sizes scaled by:
<!-- bilingual-en:end -->

- `1`
- `10`
- `100`
- `1000`

这样逐步扩大，观察 timing 的变化。
<!-- bilingual-en:start -->
Increasing the input in steps like this makes the change in runtime visible.
<!-- bilingual-en:end -->

课堂此时在训练的不是精准测量，而是量级直觉：
<!-- bilingual-en:start -->
The point here is not precise measurement, but intuition about scale:
<!-- bilingual-en:end -->

- 常数函数几乎不怎么变
- 线性函数增长比较平稳
- 二次函数会很快变得难以忍受
<!-- bilingual-en:start -->
- Constant functions hardly change at all
- Linear functions grow steadily
- Quadratic functions quickly become impractical
<!-- bilingual-en:end -->

### 5. timing 的长处：真实、直观
<!-- bilingual-en:start -->
*5. Timing's Strengths: Real and Intuitive*
<!-- bilingual-en:end -->
老师并没有一开始就批 timing，而是先承认它的优点。
<!-- bilingual-en:start -->
The instructor does not begin by criticizing timing; first, she acknowledges its advantages.
<!-- bilingual-en:end -->

timing 的好处包括：
<!-- bilingual-en:start -->
The benefits of timing include:
<!-- bilingual-en:end -->

- 贴近真实运行环境
- 不需要先抽象出理论模型
- 对新手特别直观
<!-- bilingual-en:start -->
- Reflects performance in a real execution environment
- Requires no theoretical model up front
- Is especially intuitive for beginners
<!-- bilingual-en:end -->

如果你只是粗略比较两个实现，timing 常常是第一步。
<!-- bilingual-en:start -->
For a rough comparison of two implementations, timing is often the natural first step.
<!-- bilingual-en:end -->

### 6. timing 的局限：噪声很多，解释不稳定
<!-- bilingual-en:start -->
*6. Limitations of timing: noisy measurements and unstable conclusions*
<!-- bilingual-en:end -->
但老师很快也指出 timing 的局限。
<!-- bilingual-en:start -->
The instructor soon points out timing's limitations as well.
<!-- bilingual-en:end -->

例如：
<!-- bilingual-en:start -->
For example:
<!-- bilingual-en:end -->

- 机器当前负载
- Python 解释器本身的细节
- 运行时缓存
- 样本太小导致的测量误差
<!-- bilingual-en:start -->
- Current machine load
- Details of the Python interpreter itself
- Runtime caching
- Measurement errors due to small sample sizes
<!-- bilingual-en:end -->

都会让 timing 结果抖动。
<!-- bilingual-en:start -->
All these factors cause timing results to fluctuate.
<!-- bilingual-en:end -->

所以：
<!-- bilingual-en:start -->
Therefore:
<!-- bilingual-en:end -->

- 很小的输入规模上，timing 可能几乎看不出区别
- 同一函数多跑几次，也可能拿到不同时间
<!-- bilingual-en:start -->
- With very small inputs, timing may reveal almost no difference
- Even running the same function multiple times can yield different timings
<!-- bilingual-en:end -->

这就引出第二条路线。
<!-- bilingual-en:start -->
This leads to the second approach.
<!-- bilingual-en:end -->

### 7. 第二条路线：counting operations
<!-- bilingual-en:start -->
*7. Second approach: counting operations*
<!-- bilingual-en:end -->
老师随后转到 counting operations。
<!-- bilingual-en:start -->
The instructor then turns to counting operations.
<!-- bilingual-en:end -->

思想很简单：
<!-- bilingual-en:start -->
The idea is simple:
<!-- bilingual-en:end -->

- 不直接问“花了多少秒”
- 而是问“执行了多少基本操作”
<!-- bilingual-en:start -->
- Do not ask, “How many seconds did it take?”
- Ask instead, “How many basic operations were executed?”
<!-- bilingual-en:end -->

这样做的好处是：
<!-- bilingual-en:start -->
The benefits of this approach are:
<!-- bilingual-en:end -->

- 更抽象
- 更稳定
- 不那么依赖机器和环境噪声
<!-- bilingual-en:start -->
- More abstract
- More stable
- Less dependent on machine and environmental noise
<!-- bilingual-en:end -->

当然代价是你得自己建模“哪些东西算一次操作”。
<!-- bilingual-en:start -->
The tradeoff is that you must decide what counts as a single operation.
<!-- bilingual-en:end -->

### 8. 从常数函数开始数：`c_to_f`
<!-- bilingual-en:start -->
*8. Starting with constant functions: `c_to_f`*
<!-- bilingual-en:end -->
以最简单的 `c_to_f(c)` 为例，老师把几步乘法、除法、加法粗略记成固定个数操作。
<!-- bilingual-en:start -->
For the simple function `c_to_f(c)`, the instructor treats its multiplications, division, and addition as a fixed number of operations.
<!-- bilingual-en:end -->

于是无论输入 `c` 是多少，操作数都差不多保持不变。
<!-- bilingual-en:start -->
Regardless of the input `c`, the number of operations remains roughly constant.
<!-- bilingual-en:end -->

这一点和 timing 对应起来，就形成了第一个复杂度直觉：
<!-- bilingual-en:start -->
Combined with the timing results, this gives the first intuition about growth:
<!-- bilingual-en:end -->

- 不是所有函数的运行成本都随输入值本身变化
<!-- bilingual-en:start -->
- A function's running cost does not necessarily grow with the numerical value of its input
<!-- bilingual-en:end -->

### 9. 线性例子：`mysum(x)`
<!-- bilingual-en:start -->
*9. Linear example: `mysum(x)`*
<!-- bilingual-en:end -->
对于：
<!-- bilingual-en:start -->
For:
<!-- bilingual-en:end -->

```python
def mysum(x):
    total = 0
    for i in range(x + 1):
        total += i
```

老师数操作时会把：
<!-- bilingual-en:start -->
When counting operations, the instructor includes:
<!-- bilingual-en:end -->

- 初始化
- 每轮循环中的赋值、加法、比较
<!-- bilingual-en:start -->
- Initialization
- Assignments, additions, comparisons in each loop iteration
<!-- bilingual-en:end -->

都记进去。
<!-- bilingual-en:start -->
All of these count toward the total.
<!-- bilingual-en:end -->

虽然具体常数项可以不同，但核心现象很清楚：
<!-- bilingual-en:start -->
Although specific constant terms may vary, the core phenomenon is clear:
<!-- bilingual-en:end -->

- 输入 `x` 扩大大约 10 倍
- 操作数也会大约扩大 10 倍
<!-- bilingual-en:start -->
- Input `x` grows roughly tenfold
- The operation count also grows roughly tenfold
<!-- bilingual-en:end -->

这就是线性增长。
<!-- bilingual-en:start -->
That is linear growth.
<!-- bilingual-en:end -->

### 10. 二次例子：`square(n)`
<!-- bilingual-en:start -->
*10. Quadratic Example: `square(n)`*
<!-- bilingual-en:end -->
对双重循环的 `square(n)` 来说，情况就明显不同了。
<!-- bilingual-en:start -->
For the double-loop `square(n)`, things are quite different.
<!-- bilingual-en:end -->

```python
for i in range(n):
    for j in range(n):
        ...
```

这里老师反复问的是：
<!-- bilingual-en:start -->
The instructor repeatedly asks:
<!-- bilingual-en:end -->

- 外层循环跑几次
- 每次外层里，内层又跑几次
<!-- bilingual-en:start -->
- How many times does the outer loop run?
- For each outer-loop iteration, how many times does the inner loop run?
<!-- bilingual-en:end -->

所以总操作量大约和 `n * n` 成正比。
<!-- bilingual-en:start -->
Therefore, the total number of operations is roughly proportional to `n * n`.
<!-- bilingual-en:end -->

这就是为什么当输入扩大 10 倍时：
<!-- bilingual-en:start -->
This explains why, when the input size grows tenfold:
<!-- bilingual-en:end -->

- 线性函数可能只慢约 10 倍
- 二次函数可能慢约 100 倍
<!-- bilingual-en:start -->
- A linear function may take about 10 times as long
- A quadratic function may take about 100 times as long
<!-- bilingual-en:end -->

### 11. timing 和 counting 的关系
<!-- bilingual-en:start -->
*11. The Relationship Between Timing and Counting*
<!-- bilingual-en:end -->
这节课并不是让你二选一，而是在建立比较框架。
<!-- bilingual-en:start -->
This lecture does not ask you to choose between timing and counting; it builds a framework for comparing them.
<!-- bilingual-en:end -->

它们的关系可以概括为：
<!-- bilingual-en:start -->
Their relationship can be summarized as:
<!-- bilingual-en:end -->

- timing：真实世界表现，直观但 noisy
- counting：抽象模型，更稳定但需要人为设定
<!-- bilingual-en:start -->
- Timing: Real-world performance, intuitive but noisy
- Counting: An abstract model that is more stable but requires a chosen definition of an operation
<!-- bilingual-en:end -->

老师希望你意识到：
<!-- bilingual-en:start -->
The instructor wants you to recognize that:
<!-- bilingual-en:end -->

- 仅靠 timing 看一眼，不够稳
- 仅靠 operation counting，不够贴近真实机器
<!-- bilingual-en:start -->
- Relying solely on timing for a single measurement is not reliable
- Operation counting alone does not fully capture performance on a real machine
<!-- bilingual-en:end -->

后面的复杂度理论，其实是在 counting 的基础上进一步抽象。
<!-- bilingual-en:start -->
The complexity theory introduced later abstracts this counting framework one step further.
<!-- bilingual-en:end -->

### 12. 课程这时还没正式进入 Big Theta，但地基已经搭好了
<!-- bilingual-en:start -->
*12. The lecture has not formally introduced Big Theta yet, but the foundation is in place*
<!-- bilingual-en:end -->
Lecture 21 的任务不是正式定义 Big O / Theta。  
它更像是先让你接受：
<!-- bilingual-en:start -->
Lecture 21 is not meant to define Big O or Theta formally.
Its purpose is to establish that:
<!-- bilingual-en:end -->

- 程序效率可以系统比较
- 输入规模变化比单次运行结果更重要
- constant / linear / quadratic 的增长差异是真实会咬人的
<!-- bilingual-en:start -->
- Program efficiency can be systematically compared
- Changes in input size are more important than single-run results
- Differences in constant, linear, and quadratic growth rates have real consequences
<!-- bilingual-en:end -->

这让下节课进入 order of growth 时不会显得凭空抽象。
<!-- bilingual-en:start -->
As a result, the next lecture's treatment of order of growth will not seem abstract or unmotivated.
<!-- bilingual-en:end -->

## Exercise log

> [!warning] No official finger exercise
> 这讲官方没有单独的 finger exercise 文件。
> <!-- bilingual-en:start -->
> There is no separate finger exercise file for this lecture.
> <!-- bilingual-en:end -->

如果要按课堂内容做一个最像 finger exercise 的自测，最合适的是自己完成下面两步：
<!-- bilingual-en:start -->
The closest equivalent to a finger exercise is to complete these two steps yourself:
<!-- bilingual-en:end -->

- 选 `c_to_f`、`mysum`、`square` 三个函数，预测输入扩大 10 倍时 timing 和 ops 分别会怎样变化。
- 不看代码注释，自己为 `mysum` 或 `square` 重做一遍 operation counting。
<!-- bilingual-en:start -->
- For `c_to_f`, `mysum`, and `square`, predict how runtime and operation count change when the input grows tenfold.
- Without looking at code comments, manually redo the operation counting for either `mysum` or `square`.
<!-- bilingual-en:end -->

这两步正好对应本讲的两条主线：
<!-- bilingual-en:start -->
These two steps directly correspond to the lecture's two main threads:
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
- [ ] Can I explain why the course shifts from correctness to efficiency?
- [ ] Can I describe the trade-off between time efficiency and space efficiency?
- [ ] Can I use Fibonacci memoization to explain 'trading space for time'?
- [ ] Can I outline the basic steps of timing a program?
- [ ] Can I compare how the runtimes of `c_to_f`, `mysum`, and `square` grow as their inputs grow?
- [ ] Can I explain why timing can be noisy?
- [ ] Can I describe the fundamental idea behind counting operations?
- [ ] Can I manually derive an approximate operation-count expression for a simple loop-based function?
- [ ] Can I explain why linear and quadratic growth behave so differently when the input grows tenfold?
- [ ] Can I reconstruct the lecture sequence: motivation -> timing -> limitations -> counting?
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 用一次 timing 结果就断定两个程序谁更优。
> - 数操作时把所有常数差异都当成最重要部分。
> - 只看输入值本身，而不看“输入规模”到底该怎么定义。
> - 把时间快慢和空间占用想成永远同步改进。
> <!-- bilingual-en:start -->
> - Inferring which program is better from a single timing run.
> - Treating differences in constants as the most important part of operation counting.
> - Looking only at an input's numerical value without defining what “input size” means.
> - Assuming that runtime and memory use always improve together.
> <!-- bilingual-en:end -->

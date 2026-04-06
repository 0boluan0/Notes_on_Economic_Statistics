---
aliases:
  - MIT 6.100L Lecture 15
  - 6.100L L15
  - Recursion
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 15
---

# Lecture 15: Recursion

> [!tip] Hint
> - 这节课一开始就提醒你：recursion 不是更高级的 loop，而是换一种想问题的方式。
> - 老师先回顾 iterative algorithms：状态变量、循环更新、停止条件、累计结果。
> - 然后故意用“乘法只允许靠加法实现”这个例子，把 iterative 写法和 recursive 写法并排比较。
> - 递归最重要的两个部分是 base case 和 recursive step，缺一个都不行。
> - `mult_recur(a, b)` 里真正变化的是问题规模：`b` 在不断减小，直到碰到 base case。
> - 课堂花很长时间 trace function calls，是为了让你接受“每次调用都是独立的一层工作”。
> - 老师用了批改作业/层层转交任务的类比来解释为什么递归会形成调用链。
> - `power_recur` 让你看到一个递归函数可以有不止一个 base case。
> - factorial 例子把“调用链下去，再沿调用链返回”这件事讲得最完整。
> - 听完这节课，你应该能解释 recursion 的执行过程，而不只是会照抄 `return n * f(n-1)`。

## Lecture flow

### 1. 先承认：recursion 会迫使你换脑子
Lecture 15 一开始老师就把预期管理说得很直白：

- recursion 是一种 programming technique
- 它不一定一上手就自然
- 你需要暂时放下之前对 loops 的默认思路

这段开场很重要，因为课程不是把 recursion 当成“更快的 loop”，而是把它当成另一种 problem-solving lens。

### 2. 先复习 iterative algorithms 是怎么想的
为了让对比更清楚，老师没有直接写递归，而是先回顾迭代算法的思维方式。

迭代算法通常依赖：

- 某些状态变量
- 每轮循环更新这些状态
- 一个停止条件
- 最后累积出答案

老师这里其实是在提醒你，前面十几讲你已经非常熟悉这种模式了：

- running sum
- counters
- while 条件
- for exhaust sequence

所以 recursion 的进入点必须通过比较，才能看出差异。

### 3. 先用乘法的迭代版建立共同问题
老师选的例子是：  
假装我们不会用 `*`，只会用加法，如何算 `a * b`。

先给出迭代版：

```python
def mult_iter(a, b):
    result = 0
    while b > 0:
        result += a
        b -= 1
    return result
```

这里完全是熟悉的 iterative thinking：

- `result` 记录当前累积值
- `b` 记录还要加几次
- 每轮做一次相同动作
- 直到 `b == 0`

### 4. 递归版不是“循环换写法”，而是“问题换表述”
老师接着问：  
如果用递归想同一个问题，会怎么想？

关键观察是：

- `a * b`
- 可以看成 `a + a * (b - 1)`

也就是说，你不再想“我要循环 b 次”，而是想：

- 如果我已经知道 `a * (b - 1)` 怎么求
- 那 `a * b` 只差再加一个 `a`

这就是递归思维的入口：

> [!note]
> 把当前问题写成“一个更小的同类问题 + 一点额外工作”。

### 5. base case：递归必须有一个不用再递归的地方
一旦有了递归表达式，还差最关键的一步：base case。

例如：

```python
def mult_recur(a, b):
    if b == 1:
        return a
    else:
        return a + mult_recur(a, b - 1)
```

这里：

- `b == 1` 是 base case
- `a + mult_recur(a, b - 1)` 是 recursive step

为什么 base case 必须存在？

- 否则问题规模只会一直缩小
- 但永远没有真正停下来的点
- 最后递归会无限展开

### 6. 递归还有第二个要求：每一步都要朝 base case 靠近
老师在讲完 base case 之后，又单独强调了另一个常见失误：

- 不只是要有 base case
- recursive step 还必须真正 **推进 toward base case**

在 `mult_recur(a, b)` 里，体现为：

- `b` 每次减 1
- 迟早会到 `1`

如果参数没朝 base case 靠近，即使你写了 base case，也可能永远到不了。

### 7. 课堂花很大篇幅 trace 调用链
这节课最容易让人着急，但又最重要的部分，就是老师手动 trace `mult_recur(5, 4)` 之类的调用。

她不断追问：

- 当前函数调用知道什么
- 它不知道什么
- 它下一步要把工作交给谁
- 谁会把结果返回给它

这是为了让你真正建立对 recursion execution 的感觉：

- 每次函数调用都是独立的一层
- 每层有自己的参数和局部环境
- 当前层只负责“做自己这一步，再等更小问题回来”

### 8. 调用链怎么“下去”又怎么“回来”
如果 trace `mult_recur(5, 4)`：

```python
mult_recur(5, 4)
= 5 + mult_recur(5, 3)
= 5 + (5 + mult_recur(5, 2))
= 5 + (5 + (5 + mult_recur(5, 1)))
= 5 + (5 + (5 + 5))
```

你会看到两个阶段：

- 向下展开：不断制造更小调用
- 向上返回：base case 出现后，结果一层层带回来

这也是为什么 recursion 往往让人感觉“先悬空着一堆没完成的工作”。

### 9. 课堂中的类比：递归像一层层把任务交下去
老师在课堂中用了一个非常形象的类比：  
像批改作业或层层找助教帮忙那样。

直觉上可以理解为：

- 当前人接到任务
- 发现还得先解决一个更小的子任务
- 把子任务继续交下去
- 最底层的人先完成最小任务
- 然后答案再一层层返回

这个类比虽然不影响写代码，但对理解“为什么每层调用都像在等待下一级完成”非常有帮助。

### 10. `power_recur`：一个递归函数可以有多个 base case
讲完乘法之后，老师把递归模式推广到幂函数：

```python
def power_recur(n, p):
    if p == 0:
        return 1
    elif p == 1:
        return n
    else:
        return n * power_recur(n, p - 1)
```

这个例子的重要性在于：

- 让你看到 base case 不一定只有一个
- 让你开始把“数学定义”直接翻译成递归结构

这里课堂的视角已经从“trace 一次递归”转成“如何设计递归函数”。

### 11. factorial：最标准的递归定义
接着老师把焦点放到 factorial。

数学定义本身就很递归：

- `0! = 1` 或 `1! = 1`
- `n! = n * (n - 1)!`

所以它特别适合拿来讲 recursion。

代码形式自然写成：

```python
def fact_recur(n):
    if n == 1:
        return 1
    else:
        return n * fact_recur(n - 1)
```

课堂这里又一次长时间 trace，是为了让你看到 factorial 的调用链如何层层展开、再层层返回。

### 12. 每个函数调用都是独立实例
老师在 factorial 追踪里不断强调一个容易忽视的事实：

- 虽然每层调用名字都叫 `fact_recur`
- 但每一层都是不同的 function call
- 每一层只知道自己的 `n`

所以递归不是“一个函数在一张纸上自言自语”，而是“一连串独立调用形成的链”。

这也是为什么 trace 时要认真区分：

- 当前层参数是什么
- 谁调用了我
- 我还欠着谁一个返回值

### 13. 递归与迭代比较：不一定更快，但有时更自然
课程最后没有把 recursion 神化。  
老师很明确地说：

- 有些问题递归写起来更自然
- 但迭代版往往更直接、更高效
- 不是每个问题都必须递归

因此这节课的真正目标不是让你以后“逢题必递归”，而是多掌握一种拆问题的方式。

## Exercise log

> [!example] Finger exercise 15
> 官方练习是 `recur_power(base, exp)`：
> - `exp >= 0`
> - 用 recursion 计算 `base ** exp`
> - 提示明确给出：base case 在 `exp == 0`

这题之所以合适，是因为它几乎是课堂上 `power_recur` 的直接练习版。  
它会检查你有没有真正掌握：

- base case 写什么
- recursive step 如何朝 base case 缩小

官方解法是：

```python
if exp <= 0:
    return 1
return base * recur_power(base, exp - 1)
```

如果你写这题时还是下意识想上 `for` 或 `while`，说明你的脑子还没切进递归模式。

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec15.pdf|Lecture 15 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec15_code.py|Lecture 15 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex15_sol.pdf|Lecture 15 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec15_transcript.pdf|Lecture 15 transcript]]
- Recitation 8: [[MIT 6.100L-recitations/mit6_100l_rec08.zip|Recitation 08 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 6.1)

## Review checklist
- [ ] 我能解释 recursion 和 iteration 在思考问题方式上的差别。
- [ ] 我能说清 iterative algorithm 常见的状态变量、更新和停止条件。
- [ ] 我能把乘法问题分别写成 iterative 和 recursive 的表述。
- [ ] 我能解释 base case 为什么不可缺少。
- [ ] 我能说明 recursive step 为什么必须朝 base case 推进。
- [ ] 我能手动 trace `mult_recur` 或 `fact_recur` 的调用链。
- [ ] 我能解释“每个函数调用都是独立的一层”这件事。
- [ ] 我能设计一个有两个 base case 的递归函数，比如 `power_recur`。
- [ ] 我能比较 factorial 的 iterative 和 recursive 版本各自的直观优缺点。
- [ ] 我能按课堂顺序复述：iterative recap -> multiplication -> base case/recursive step -> call tracing -> power -> factorial。

> [!warning] Common mistakes
> - 只写 recursive step，不写 base case。
> - 写了 base case，但参数变化方向根本到不了它。
> - 递归 trace 时把所有调用混成同一层。
> - 一看到 recursion 就把它当成“更短的 loop”，却没有真正换问题表述。

---
aliases:
  - MIT 6.100L Lecture 05
  - 6.100L L05
  - Floats and Approximation Methods
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 05
---

# Lecture 05: Floats and Approximation Methods

> [!tip] Hint
> - 我能解释为什么 float 不是实数本身，而是有限精度近似。
> - 我能说明 approximation algorithm 里 epsilon 的角色是什么。
> - 我能分清‘找到精确解’和‘找到足够好的近似解’的程序结构差异。
> - 我能解释为什么步长越小不一定越好，它会影响速度和可达精度。
> - 我能围绕本讲的主轴 “Float 是近似表示，不是精确实数” / “Approximation method：用小步子逼近目标” / “数值算法的正确性来自 stopping rule”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释为什么 `0.1` 的累加会暴露浮点误差。
> - 我能说明 approximation method 里 epsilon 与 increment 分别控制什么。
> - 我能写出一个有成功与失败两种退出路径的近似程序。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 3.2-3.3
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Float 是近似表示，不是精确实数 / Approximation method：用小步子逼近目标 / 数值算法的正确性来自 stopping rule
> - Lecture 4 还在枚举整数解，这一讲开始接受‘只要足够接近就行’的连续近似思路。
> - 浮点误差不是边角料，而是写数值程序时必须直面的事实。
> - 后面的 bisection search、Newton-Raphson 都是在改进这一讲的 approximation 思路。

## Core ideas
### Float 是近似表示，不是精确实数
很多初学者第一次被 `0.1 + 0.1 + ...` 吓到，其实这不是 Python 的怪脾气，而是有限精度表示的普遍现象。
- 浮点数能表示的实数是离散的一小部分，所以某些十进制小数无法被精确存储。
- 因此比较两个 float 时，通常不该追求 `==`，而要比较它们是否在允许误差范围内足够接近。
- 只要程序在处理中使用 float，就要默认误差会传播；关键是管理误差，而不是假装误差不存在。
- 数值程序的输出要看是否满足问题需求，而不是是否达到数学上的完美精确。

> [!note] What to internalize
> - One-sentence takeaway: 很多初学者第一次被 `0.1 + 0.1 + ...` 吓到，其实这不是 Python 的怪脾气，而是有限精度表示的普遍现象。
> - Review anchor: 浮点数能表示的实数是离散的一小部分，所以某些十进制小数无法被精确存储。
> - Review anchor: 因此比较两个 float 时，通常不该追求 `==`，而要比较它们是否在允许误差范围内足够接近。

从做题角度看，只要题目在考“Float 是近似表示，不是精确实数”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：很多初学者第一次被 `0.1 + 0.1 + ...` 吓到，其实这不是 Python 的怪脾气，而是有限精度表示的普遍现象。

### Approximation method：用小步子逼近目标
当问题的解不一定是整数时，最直接的办法是从某个起点开始按固定步长移动，直到误差足够小。
- 需要明确四个量：目标值、当前 guess、步长 increment、可接受误差 epsilon。
- 循环 guard 通常写成 `abs(guess**2 - x) >= epsilon`，也就是‘只要还不够好，就继续试’。
- 固定步长法的优势是简单；缺点是步长小会非常慢，步长大又可能跳过好答案。
- 如果 guess 已经越界仍未达到误差要求，程序必须有失败分支，而不是无限尝试。

> [!note] What to internalize
> - One-sentence takeaway: 当问题的解不一定是整数时，最直接的办法是从某个起点开始按固定步长移动，直到误差足够小。
> - Review anchor: 需要明确四个量：目标值、当前 guess、步长 increment、可接受误差 epsilon。
> - Review anchor: 循环 guard 通常写成 `abs(guess**2 - x) >= epsilon`，也就是‘只要还不够好，就继续试’。

从做题角度看，只要题目在考“Approximation method：用小步子逼近目标”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：当问题的解不一定是整数时，最直接的办法是从某个起点开始按固定步长移动，直到误差足够小。

### 数值算法的正确性来自 stopping rule
近似算法不再追求‘恰好命中’，所以你要更认真地定义什么时候可以停、什么时候必须承认失败。
- `epsilon` 决定了答案的精细程度；它不是越小越好，而是要和成本、需求匹配。
- 除了误差条件，还常常需要配合边界条件，例如不让 guess 走到明显不合理的范围之外。
- 一个近似算法通常同时存在两种退出方式：成功逼近目标，或者发现当前策略不足以找到可接受答案。
- 所以数值程序的输出应该同时报告结果和一些诊断信息，例如猜测次数、最后的误差。

> [!note] What to internalize
> - One-sentence takeaway: 近似算法不再追求‘恰好命中’，所以你要更认真地定义什么时候可以停、什么时候必须承认失败。
> - Review anchor: `epsilon` 决定了答案的精细程度；它不是越小越好，而是要和成本、需求匹配。
> - Review anchor: 除了误差条件，还常常需要配合边界条件，例如不让 guess 走到明显不合理的范围之外。

从做题角度看，只要题目在考“数值算法的正确性来自 stopping rule”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：近似算法不再追求‘恰好命中’，所以你要更认真地定义什么时候可以停、什么时候必须承认失败。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - Example: successive addition
> - 0.125 is a perfect power of 2
> - x = 0
> - for i in range(10)
> - x += 0.125
> - print(x == 1.25)
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 用固定步长近似平方根
> ```python
> x = 25
> epsilon = 0.01
> guess = 0.0
> increment = 0.0001
> while abs(guess**2 - x) >= epsilon and guess**2 <= x:
>     guess += increment
> print(guess)
> ```
> 这是最朴素的连续搜索。它展示了为什么 approximation 比整数枚举更灵活，但也暴露了速度瓶颈。

> [!example] 不要直接用 `==` 比较浮点
> ```python
> x = 0.0
> for _ in range(10):
>     x += 0.1
> print(x == 1.0)
> print(abs(x - 1.0) < 1e-9)
> ```
> 第一行比较经常是 `False`，第二行更接近数值计算的正确思维：比较距离，而不是比较字面值完全一致。

## Exercise log
> [!note] Finger exercise snapshot
> - Official prompt: Assume you are given a string variaboe named my_str . Write a piece of Python code that prints out a new string containing the even indexed characters of my_str . For exampoe, if my_str = "abcdefg" then your code shouod...
> - What this is really testing: whether you can compress the lecture into one small, high-frequency coding move without needing the slides beside you.
> - Where to revisit if this feels shaky: go back to the first two Core ideas sections in this note, then rerun the official lecture code once with your own input.
> - Homework bridge: this lecture is directly connected to the following calendar milestones: PS 1 halfway hand-in due.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: calendar shows this lecture touching the following milestones: PS 1 halfway hand-in due。读完本讲后，不应只会解释概念，还应能把它们搬到更长的程序里。
> - Recitation connection: there is no recitation attached to this lecture week in the official calendar.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec05.pdf|Lecture 05 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec05_code.py|Lecture 05 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex05_sol.pdf|Lecture 05 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec05_transcript.pdf|Lecture 05 transcript]]
- Recitation: none attached to this lecture week
- PS 1 halfway hand-in due: [[MIT 6.100L-problem-sets/mit6_100l_ps1.pdf|PS1 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps1_code.zip|PS1 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 3.2-3.3)

## Review checklist
- [ ] 我能解释为什么 `0.1` 的累加会暴露浮点误差。
- [ ] 我能说明 approximation method 里 epsilon 与 increment 分别控制什么。
- [ ] 我能写出一个有成功与失败两种退出路径的近似程序。
- [ ] 我能解释为什么小步长法正确但可能极慢。
- [ ] 我能说明何时应该用“足够接近”取代“完全相等”。
- [ ] 我能围绕“Float 是近似表示，不是精确实数”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Approximation method：用小步子逼近目标”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：把 float 当成精确实数，写大量 `==` 判断。
- [ ] 我能说出并避免这个高频误区：只关心 epsilon，不关心步长和边界条件，导致程序极慢或失败。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 把 float 当成精确实数，写大量 `==` 判断。
> - 只关心 epsilon，不关心步长和边界条件，导致程序极慢或失败。
> - 近似算法没有失败分支，最后得到一个看起来像答案但其实不可信的结果。

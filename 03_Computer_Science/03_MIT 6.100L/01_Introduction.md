---
aliases:
  - MIT 6.100L Lecture 01
  - 6.100L L01
  - Introduction
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 01
---

# Lecture 01: Introduction

> [!tip] Hint
> - 我能区分 declarative knowledge 和 imperative knowledge，并说明编程为什么更像写 recipe。
> - 我能说出一个可执行算法至少要有哪三个组成部分。
> - 我能解释对象、类型、表达式、赋值之间的关系，而不是只背语法。
> - 我能从最简单的数值例子里看出“计算 = 不断改进 guess”。
> - 我能围绕本讲的主轴 “Computation、knowledge 与 algorithm” / “Python 的对象、类型与表达式” / “变量、binding 与简单调试”，不翻 slides 也把整节课重新讲一遍。
> - 我能不用看笔记，口头讲出 declarative knowledge 与 imperative knowledge 的区别。
> - 我能解释为什么“有步骤”还不够，算法还必须有 flow of control 和 stopping condition。
> - 我能说明 `type(5)` 与 `type(3.0)` 的差异为什么会影响程序行为。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 1, Ch 2.1-2.2
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Computation、knowledge 与 algorithm / Python 的对象、类型与表达式 / 变量、binding 与简单调试
> - 本讲是整门课的入口：先建立“计算是什么”的心智模型，再落到 Python 的对象、类型、变量和基本运算。
> - 如果这一讲掌握得稳，后面所有控制流、函数、数据结构都会变成“在对象上做更复杂的 recipe”。
> - 最重要的不是记住几个运算符，而是理解程序如何把模糊想法变成机器可执行的步骤。

## Core ideas
### Computation、knowledge 与 algorithm
这门课把程序看成一种把事实生产出来的 recipe。你不是直接告诉计算机答案，而是告诉它怎样一步步得到答案。
- ==Declarative knowledge== 是事实陈述，例如“平方根是满足 y*y = x 的数”；==imperative knowledge== 是操作步骤，例如“先猜一个值，再不断修正它”。
- 编程本质上是在写 imperative knowledge：你给机器的是一套可重复执行的过程，而不是一堆最后的结论。
- 一个算法至少要有三件事：一串简单步骤、步骤执行的控制流程、以及清楚的停止条件。
- 如果一个方法没有说明什么时候停，或者每一步都不够机械化，那它就不是机器友好的算法。

> [!note] What to internalize
> - One-sentence takeaway: 这门课把程序看成一种把事实生产出来的 recipe。你不是直接告诉计算机答案，而是告诉它怎样一步步得到答案。
> - Review anchor: ==Declarative knowledge== 是事实陈述，例如“平方根是满足 y*y = x 的数”；==imperative knowledge== 是操作步骤，例如“先猜一个值，再不断修正它”。
> - Review anchor: 编程本质上是在写 imperative knowledge：你给机器的是一套可重复执行的过程，而不是一堆最后的结论。

从做题角度看，只要题目在考“Computation、knowledge 与 algorithm”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：这门课把程序看成一种把事实生产出来的 recipe。你不是直接告诉计算机答案，而是告诉它怎样一步步得到答案。

### Python 的对象、类型与表达式
Python 里几乎所有值都是对象。类型决定这个对象支持哪些操作，也决定不同操作组合后会得到什么。
- `5` 是整数对象，`3.0` 是浮点对象；`type(...)` 让你先确认“我手上的值到底是什么”。
- 表达式会先被求值，再产生一个结果对象；例如 `(4+2)*6-1` 先算括号，再算乘法，再算减法。
- `int(...)`、`float(...)`、`round(...)` 等转换函数很重要，因为同一个数学量在不同类型下行为并不完全一样。
- 当你觉得代码“看起来没错”，第一件事通常不是盲猜，而是先检查值和类型。

> [!note] What to internalize
> - One-sentence takeaway: Python 里几乎所有值都是对象。类型决定这个对象支持哪些操作，也决定不同操作组合后会得到什么。
> - Review anchor: `5` 是整数对象，`3.0` 是浮点对象；`type(...)` 让你先确认“我手上的值到底是什么”。
> - Review anchor: 表达式会先被求值，再产生一个结果对象；例如 `(4+2)*6-1` 先算括号，再算乘法，再算减法。

从做题角度看，只要题目在考“Python 的对象、类型与表达式”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：Python 里几乎所有值都是对象。类型决定这个对象支持哪些操作，也决定不同操作组合后会得到什么。

### 变量、binding 与简单调试
变量不是一个永远装着固定内容的小盒子，更准确的说法是：名字被绑定到某个对象上；重新赋值就是把名字重新指向另一个值。
- `pi = 355/113` 与 `radius = 2.2` 建立了两个 binding，后面 `area = pi*(radius**2)` 只是把已有 binding 组合起来。
- 变量名要表达含义。`radius` 比 `r` 更利于阅读，而 `area`、`circumference` 让结果的角色一眼就懂。
- 代码风格也是调试工具：命名清楚、每一步含义明确，错误更容易被发现。
- 当变量被重新赋值时，例如 `radius = radius + 1`，你要同时跟踪“名字没变”与“对象的值变了”这两件事。

> [!note] What to internalize
> - One-sentence takeaway: 变量不是一个永远装着固定内容的小盒子，更准确的说法是：名字被绑定到某个对象上；重新赋值就是把名字重新指向另一个值。
> - Review anchor: `pi = 355/113` 与 `radius = 2.2` 建立了两个 binding，后面 `area = pi*(radius**2)` 只是把已有 binding 组合起来。
> - Review anchor: 变量名要表达含义。`radius` 比 `r` 更利于阅读，而 `area`、`circumference` 让结果的角色一眼就懂。

从做题角度看，只要题目在考“变量、binding 与简单调试”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：变量不是一个永远装着固定内容的小盒子，更准确的说法是：名字被绑定到某个对象上；重新赋值就是把名字重新指向另一个值。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - TYPE THIS IN THE CONSOLE - CHECK THE TYPE OF OBJECTS ##
> - TYPE THIS IN THE CONSOLE - CONVERT TO ANOTHER TYPE ##
> - TYPE THIS IN THE CONSOLE - EXPRESSIONS ##
> - TYPE THIS IN THE CONSOLE - VARIABLES ##
> - Compute approximate value for pi
> - CODE STYLE ##
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 用近似法理解“算法是逐步改进 guess”
> ```python
> x = 16
> guess = 3
> new_guess = (guess + x/guess) / 2
> print(new_guess)
> ```
> 这段代码不在于一行算出平方根，而在于展示一种通用套路：从一个可接受的初值出发，反复根据误差更新 guess。

> [!example] 把物理量写成有语义的 binding
> ```python
> pi = 355 / 113
> radius = 2.2
> area = pi * (radius ** 2)
> circumference = pi * (radius * 2)
> print(area, circumference)
> ```
> 这里的重点不是公式本身，而是把问题拆成可读的中间量。只要中间量清楚，代码就更像推导而不是谜语。

## Exercise log
> [!note] Finger exercise snapshot
> - Official prompt: Assume 3 variables are already defined for you: a , b , and c . Create a variable called total that adds a and b then multiplies the result by c . Include a last line in your code to print the value: print(total)
> - Official solution sketch:
> ```python
> total = (a+b)*c
> print(total)
> ```
> - What this is really testing: whether you can compress the lecture into one small, high-frequency coding move without needing the slides beside you.
> - Where to revisit if this feels shaky: go back to the first two Core ideas sections in this note, then rerun the official lecture code once with your own input.
> - Follow-on practice path: after this finger exercise, the most natural next stop is Recitation 01.
> - Homework bridge: this lecture is directly connected to the following calendar milestones: PS 0 out (not graded).

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: calendar shows this lecture touching the following milestones: PS 0 out (not graded)。读完本讲后，不应只会解释概念，还应能把它们搬到更长的程序里。
> - Recitation connection: Recitation 01 is the best place to turn the lecture ideas into shorter solved exercises.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec01.pdf|Lecture 01 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec01_code.py|Lecture 01 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex01_sol.pdf|Lecture 01 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec01_transcript.pdf|Lecture 01 transcript]]
- Recitation 1: [[MIT 6.100L-recitations/mit6_100l_rec01.pdf|Recitation 01 materials]]
- PS 0 out (not graded): [[MIT 6.100L-problem-sets/mit6_100l_ps0.pdf|PS0 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps0_code.zip|PS0 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 1, Ch 2.1-2.2)

## Review checklist
- [ ] 我能不用看笔记，口头讲出 declarative knowledge 与 imperative knowledge 的区别。
- [ ] 我能解释为什么“有步骤”还不够，算法还必须有 flow of control 和 stopping condition。
- [ ] 我能说明 `type(5)` 与 `type(3.0)` 的差异为什么会影响程序行为。
- [ ] 我能解释为什么 `radius = radius + 1` 在编程里合法，但在数学等式里不合法。
- [ ] 我能把一个生活中的 recipe 描述成算法。
- [ ] 我能围绕“Computation、knowledge 与 algorithm”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Python 的对象、类型与表达式”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：把 = 当成数学里的“左右永远相等”，而不是“把名字绑定到右边的结果”。
- [ ] 我能说出并避免这个高频误区：不检查类型，直接假设整数和浮点的行为完全一样。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 把 `=` 当成数学里的“左右永远相等”，而不是“把名字绑定到右边的结果”。
> - 不检查类型，直接假设整数和浮点的行为完全一样。
> - 用没有语义的变量名，让后续代码读起来像在猜谜。

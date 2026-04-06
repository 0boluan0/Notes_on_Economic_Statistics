---
aliases:
  - MIT 6.100L Lecture 26
  - 6.100L L26
  - List Access, Hashing, Simulations, and Wrap-Up
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 26
---

# Lecture 26: List Access, Hashing, Simulations, and Wrap-Up

> [!tip] Hint
> - 我能解释按位置访问 list 为什么通常便宜，而哈希表又为什么能支持按 key 的快速访问。
> - 我能说明 hashing 的目标是把查找从线性扫描转成更直接的定位。
> - 我能解释 simulation 为什么是程序的一种重要用途。
> - 我能把整门课的主线串起来：数据表示、控制流、抽象、复杂度。
> - 我能围绕本讲的主轴 “List access 与 hashing 是两种不同的定位思路” / “Simulation：当直接推公式很难时，用程序做实验” / “Wrap-up：整门课的主线应当串成一个系统”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释 list access 与 hashing 的思路差异。
> - 我能说明为什么 dict 的查找方式与 list 不同。
> - 我能设计一个最简单的 simulation 实验。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 12.3, Ch 17
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: List access 与 hashing 是两种不同的定位思路 / Simulation：当直接推公式很难时，用程序做实验 / Wrap-up：整门课的主线应当串成一个系统
> - 最后一讲回收多个主题：底层访问方式、哈希思想、模拟方法，以及对全课主线的整理。
> - 它提醒你：课程不是一堆零散专题，而是围绕‘怎样表示问题、怎样组织计算、怎样评估代价’展开。
> - list access 与 hashing 给出不同的数据访问路径，simulation 则展示程序如何用于研究难以直接解析的问题。

## Core ideas
### List access 与 hashing 是两种不同的定位思路
顺序容器和映射容器的差别，最终会落实到‘我要怎样更快找到目标数据’这个问题上。
- list 的优势在于顺序和按位置访问；当你知道 index 时，拿元素通常很直接。
- 但如果你只知道内容或 key，而不知道位置，线性搜索的成本就会上升。
- hashing 的目标是把 key 通过哈希函数映射到更容易定位的位置，从而避免从头扫到尾。
- 这正是 dict 在大量查找场景里常常很强的原因。

> [!note] What to internalize
> - One-sentence takeaway: 顺序容器和映射容器的差别，最终会落实到‘我要怎样更快找到目标数据’这个问题上。
> - Review anchor: list 的优势在于顺序和按位置访问；当你知道 index 时，拿元素通常很直接。
> - Review anchor: 但如果你只知道内容或 key，而不知道位置，线性搜索的成本就会上升。

从做题角度看，只要题目在考“List access 与 hashing 是两种不同的定位思路”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：顺序容器和映射容器的差别，最终会落实到‘我要怎样更快找到目标数据’这个问题上。

### Simulation：当直接推公式很难时，用程序做实验
模拟的本质是：明确规则、反复试验、统计结果。很多复杂系统不容易手算，但可以通过程序近似观察其行为。
- simulation 需要先定义状态、随机过程或更新规则，再运行足够多次观察总体行为。
- 它不一定给出解析闭式答案，但能帮助你估计概率、趋势或平均表现。
- 程序在这里不只是计算器，而是实验平台。
- 这让课程回到最初的主题：计算机可以帮助我们探索那些手工难以穷尽的过程。

> [!note] What to internalize
> - One-sentence takeaway: 模拟的本质是：明确规则、反复试验、统计结果。很多复杂系统不容易手算，但可以通过程序近似观察其行为。
> - Review anchor: simulation 需要先定义状态、随机过程或更新规则，再运行足够多次观察总体行为。
> - Review anchor: 它不一定给出解析闭式答案，但能帮助你估计概率、趋势或平均表现。

从做题角度看，只要题目在考“Simulation：当直接推公式很难时，用程序做实验”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：模拟的本质是：明确规则、反复试验、统计结果。很多复杂系统不容易手算，但可以通过程序近似观察其行为。

### Wrap-up：整门课的主线应当串成一个系统
学完一门入门课，最重要的不是记住每个语法点，而是能把核心思想连起来形成自己的编程地图。
- 数据表示：数字、字符串、序列、字典、对象，决定你如何组织世界。
- 控制流：分支、循环、递归，决定你如何组织计算过程。
- 抽象：函数、类、模块化，决定你如何控制复杂度。
- 分析：测试、调试、复杂度、可视化、模拟，决定你如何判断程序是否可靠、是否高效、是否解释得通。

> [!note] What to internalize
> - One-sentence takeaway: 学完一门入门课，最重要的不是记住每个语法点，而是能把核心思想连起来形成自己的编程地图。
> - Review anchor: 数据表示：数字、字符串、序列、字典、对象，决定你如何组织世界。
> - Review anchor: 控制流：分支、循环、递归，决定你如何组织计算过程。

从做题角度看，只要题目在考“Wrap-up：整门课的主线应当串成一个系统”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：学完一门入门课，最重要的不是记住每个语法点，而是能把核心思想连起来形成自己的编程地图。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - set line width
> - set font size for titles
> - set font size for labels on axes
> - set size of numbers on x-axis
> - set size of numbers on y-axis
> - set size of ticks on x-axis
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 用 dict 体现 hashing 思维
> ```python
> phone_book = {"Ana": "617-0000", "MIT": "617-1111"}
> print(phone_book["Ana"])
> ```
> 这里你不需要顺序扫描整个表，而是通过 key 直接定位对应值，这正是 hashing 思维在 Python 中的日常体现。

> [!example] 最小随机模拟
> ```python
> import random
>
> heads = 0
> trials = 1000
> for _ in range(trials):
>     if random.choice(["H", "T"]) == "H":
>         heads += 1
> print(heads / trials)
> ```
> 模拟的套路是：定义随机规则，多次重复，然后用统计结果近似观察整体行为。

## Exercise log
> [!warning] No official finger exercise
> - Calendar explicitly marks this lecture as having no official finger exercise.
> - Use the review checklist, the lecture code, and the linked recitation / problem set materials as the primary self-test for this lecture.
> - For this lecture, a good replacement for the missing finger exercise is: hand-trace one representative example from the code, then write a fresh one from memory.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: this lecture does not have a direct calendar milestone attached, so use it as a consolidation lecture rather than a sprint lecture.
> - Recitation connection: there is no recitation attached to this lecture week in the official calendar.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec26.pdf|Lecture 26 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec26_code.py|Lecture 26 code (py)]]
- Finger exercise: no official file for this lecture
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec26_transcript.pdf|Lecture 26 transcript]]
- Recitation: none attached to this lecture week
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 12.3, Ch 17)

## Review checklist
- [ ] 我能解释 list access 与 hashing 的思路差异。
- [ ] 我能说明为什么 dict 的查找方式与 list 不同。
- [ ] 我能设计一个最简单的 simulation 实验。
- [ ] 我能用自己的话总结整门课的主线。
- [ ] 我能说出以后继续学习 Python / CS 时，这门课留下些什么。
- [ ] 我能围绕“List access 与 hashing 是两种不同的定位思路”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Simulation：当直接推公式很难时，用程序做实验”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：只记住某个数据结构快，却不理解它快在什么访问模式上。
- [ ] 我能说出并避免这个高频误区：把 simulation 当成随便跑随机代码，而没有先定义清楚规则与统计目标。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 只记住某个数据结构快，却不理解它快在什么访问模式上。
> - 把 simulation 当成随便跑随机代码，而没有先定义清楚规则与统计目标。
> - 学完整门课后仍然把概念当碎片，没有形成统一地图。

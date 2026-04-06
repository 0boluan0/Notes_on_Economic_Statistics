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
> - 我能解释为什么‘程序快不快’不该只靠感觉。
> - 我能区分 wall-clock timing 与 operation counting 的角色。
> - 我能说明常数因子、输入规模和增长趋势分别影响什么。
> - 我能把性能讨论从‘这个程序现在很快’提升到‘规模变大时会怎样’。
> - 我能围绕本讲的主轴 “Timing 给你经验事实，counting 给你结构解释” / “先识别程序里的主导操作” / “常数因子与增长趋势要分开看”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释 timing 与 operation counting 分别解决什么问题。
> - 我能从一段代码里识别主导操作。
> - 我能说明为什么性能讨论必须带上输入规模。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 11
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Timing 给你经验事实，counting 给你结构解释 / 先识别程序里的主导操作 / 常数因子与增长趋势要分开看
> - 这一讲是复杂度分析的入口：先从可观测的运行时间和可推理的操作计数开始。
> - 实际时间测量很重要，但它会受机器、实现细节、随机因素影响；因此还需要更抽象的计数模型。
> - 后面的 Big-O / Theta 都是在这里的基础上把‘增长趋势’说得更形式化。

## Core ideas
### Timing 给你经验事实，counting 给你结构解释
如果只计时不分析，你只能知道‘它现在快不快’；如果只分析不计时，你又可能忽略实现细节与常数因素。两者结合才完整。
- wall-clock timing 直接测程序运行耗时，适合比较真实实现表现。
- operation counting 关注核心操作发生了多少次，适合抽离机器差异理解算法结构。
- 当输入规模变化时，真正关键的是操作次数如何增长，而不只是某一次测试耗时。
- 性能分析的目标不是神秘数学，而是预测程序扩展后的行为。

> [!note] What to internalize
> - One-sentence takeaway: 如果只计时不分析，你只能知道‘它现在快不快’；如果只分析不计时，你又可能忽略实现细节与常数因素。两者结合才完整。
> - Review anchor: wall-clock timing 直接测程序运行耗时，适合比较真实实现表现。
> - Review anchor: operation counting 关注核心操作发生了多少次，适合抽离机器差异理解算法结构。

从做题角度看，只要题目在考“Timing 给你经验事实，counting 给你结构解释”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：如果只计时不分析，你只能知道‘它现在快不快’；如果只分析不计时，你又可能忽略实现细节与常数因素。两者结合才完整。

### 先识别程序里的主导操作
不是所有语句都同等重要。分析复杂度时，你通常只盯住那个随着输入增长最有代表性的核心操作。
- 例如搜索算法里常关注比较次数，排序里常关注比较或交换次数。
- 如果一个循环嵌套另一个循环，主导操作往往来自最内层重复部分。
- 把复杂程序拆成‘有哪些重复结构’之后，计数就会容易很多。
- 这一步是在为后面的 asymptotic notation 做铺垫。

> [!note] What to internalize
> - One-sentence takeaway: 不是所有语句都同等重要。分析复杂度时，你通常只盯住那个随着输入增长最有代表性的核心操作。
> - Review anchor: 例如搜索算法里常关注比较次数，排序里常关注比较或交换次数。
> - Review anchor: 如果一个循环嵌套另一个循环，主导操作往往来自最内层重复部分。

从做题角度看，只要题目在考“先识别程序里的主导操作”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：不是所有语句都同等重要。分析复杂度时，你通常只盯住那个随着输入增长最有代表性的核心操作。

### 常数因子与增长趋势要分开看
小输入时常数和实现细节很显眼，大输入时增长阶往往决定一切。要学会同时看这两个层次。
- 一个常数更小的二次算法，在很小输入上可能暂时比线性对数算法还快。
- 但随着规模变大，增长更慢的算法通常会反超。
- 因此性能讨论不能脱离输入规模范围和应用上下文。
- 复杂度分析的真正价值是帮助你预测‘未来会不会炸’，而不是只评价当下。

> [!note] What to internalize
> - One-sentence takeaway: 小输入时常数和实现细节很显眼，大输入时增长阶往往决定一切。要学会同时看这两个层次。
> - Review anchor: 一个常数更小的二次算法，在很小输入上可能暂时比线性对数算法还快。
> - Review anchor: 但随着规模变大，增长更慢的算法通常会反超。

从做题角度看，只要题目在考“常数因子与增长趋势要分开看”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：小输入时常数和实现细节很显眼，大输入时增长阶往往决定一切。要学会同时看这两个层次。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - Example: timing a program
> - constant fcn
> - linear fcn -- finds 0+1+2+...+x
> - quadratic fcn -- finds n*n inefficiently
> - helper function to show timing
> - creates a list [1, 10, 100, ...] to test different input sizes
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 简单计时实验
> ```python
> import time
>
> start = time.time()
> total = 0
> for i in range(1000000):
>     total += i
> print(time.time() - start)
> ```
> 计时能给你经验感觉，但如果不结合结构分析，你很难预测输入规模再翻十倍会发生什么。

> [!example] 显式计数核心操作
> ```python
> count = 0
> for i in range(5):
>     for j in range(3):
>         count += 1
> print(count)
> ```
> 这里的 `count` 可以帮助你把‘嵌套循环执行了多少次核心操作’具象化。

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
- Slides: [[MIT 6.100L-slides/mit6_100l_lec21.pdf|Lecture 21 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec21_code.py|Lecture 21 code (py)]]
- Finger exercise: no official file for this lecture
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec21_transcript.pdf|Lecture 21 transcript]]
- Recitation: none attached to this lecture week
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 11)

## Review checklist
- [ ] 我能解释 timing 与 operation counting 分别解决什么问题。
- [ ] 我能从一段代码里识别主导操作。
- [ ] 我能说明为什么性能讨论必须带上输入规模。
- [ ] 我能比较常数因子和增长趋势的影响。
- [ ] 我能对一个简单循环程序做粗略计数分析。
- [ ] 我能围绕“Timing 给你经验事实，counting 给你结构解释”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“先识别程序里的主导操作”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：只看一次计时结果就下结论，不分析输入规模增长。
- [ ] 我能说出并避免这个高频误区：把所有语句都平均看待，没有抓住主导操作。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 只看一次计时结果就下结论，不分析输入规模增长。
> - 把所有语句都平均看待，没有抓住主导操作。
> - 把常数因子与增长阶混成一回事。

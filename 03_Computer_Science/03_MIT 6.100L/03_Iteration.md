---
aliases:
  - MIT 6.100L Lecture 03
  - 6.100L L03
  - Iteration
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 03
---

# Lecture 03: Iteration

> [!tip] Hint
> - 我能解释 iteration 为什么本质上是在‘重复执行 + 持续更新状态’。
> - 我能区分 loop guard、loop body、loop variable 各自的角色。
> - 我能通过 trace 的方式检查循环什么时候停、停下时变量是什么。
> - 我能识别 infinite loop 和 off-by-one 错误分别来自哪里。
> - 我能围绕本讲的主轴 “Iteration 的核心：重复执行直到满足条件” / “Counter、accumulator 与 range” / “Loop invariant 与常见错误”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释为什么 iteration 一定伴随状态变化。
> - 我能区分 `while` 与 `for` 更适合的问题类型。
> - 我能手动 trace 一个循环并写出每一轮变量变化。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 2.5-2.8
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Iteration 的核心：重复执行直到满足条件 / Counter、accumulator 与 range / Loop invariant 与常见错误
> - 本讲把程序从“单次判断”升级到“重复执行直到条件满足”。
> - 循环真正难的不是写语法，而是同时管理好状态更新、停止条件和中间量。
> - 掌握 iteration 以后，后面的搜索、近似、枚举、递归都更容易理解。

## Core ideas
### Iteration 的核心：重复执行直到满足条件
循环不是简单的复制粘贴。它要求你在每次迭代后都能回答两个问题：状态怎么变了？离停止条件更近了吗？
- `while` 适合“做多久取决于运行中状态”的问题；guard 为真时继续，为假时结束。
- 循环体的每一步都应该推动程序朝终止状态前进，否则就可能无限循环。
- 写循环时最值得跟踪的是 loop variable、累加器以及任何会影响 guard 的变量。
- 如果你能用表格 trace 每轮的值，通常就能 debug 大多数基础循环。

> [!note] What to internalize
> - One-sentence takeaway: 循环不是简单的复制粘贴。它要求你在每次迭代后都能回答两个问题：状态怎么变了？离停止条件更近了吗？
> - Review anchor: `while` 适合“做多久取决于运行中状态”的问题；guard 为真时继续，为假时结束。
> - Review anchor: 循环体的每一步都应该推动程序朝终止状态前进，否则就可能无限循环。

从做题角度看，只要题目在考“Iteration 的核心：重复执行直到满足条件”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：循环不是简单的复制粘贴。它要求你在每次迭代后都能回答两个问题：状态怎么变了？离停止条件更近了吗？

### Counter、accumulator 与 range
很多循环都可以抽象成‘一个变量负责走步数，一个变量负责累计结果’。这两个角色搞清楚，代码就会清晰很多。
- counter 控制你已经走了多少轮，accumulator 记录迭代过程中逐步形成的结果。
- `for i in range(...)` 把“轮数控制”交给 Python 处理，适合固定次数或已知区间的重复任务。
- 不论是 `while` 还是 `for`，都要明确循环结束后的状态是否正好就是你需要的答案。
- 循环中的 print 不只是输出结果，也可以是定位 bug 的临时观察窗口。

> [!note] What to internalize
> - One-sentence takeaway: 很多循环都可以抽象成‘一个变量负责走步数，一个变量负责累计结果’。这两个角色搞清楚，代码就会清晰很多。
> - Review anchor: counter 控制你已经走了多少轮，accumulator 记录迭代过程中逐步形成的结果。
> - Review anchor: `for i in range(...)` 把“轮数控制”交给 Python 处理，适合固定次数或已知区间的重复任务。

从做题角度看，只要题目在考“Counter、accumulator 与 range”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：很多循环都可以抽象成‘一个变量负责走步数，一个变量负责累计结果’。这两个角色搞清楚，代码就会清晰很多。

### Loop invariant 与常见错误
循环最稳的理解方式是给自己一个 invariant：在每轮开始或结束时，哪些事实始终成立？
- 例如做累加时，invariant 可以是‘当前的 sum 永远等于已经处理过元素的和’。
- off-by-one 常见于 guard 写错边界，或在错误时机更新 counter。
- 无限循环常见于忘记更新状态，或更新方向与停止条件相反。
- 如果循环跑完但答案不对，先检查 invariant 是否一开始就没建立好。

> [!note] What to internalize
> - One-sentence takeaway: 循环最稳的理解方式是给自己一个 invariant：在每轮开始或结束时，哪些事实始终成立？
> - Review anchor: 例如做累加时，invariant 可以是‘当前的 sum 永远等于已经处理过元素的和’。
> - Review anchor: off-by-one 常见于 guard 写错边界，或在错误时机更新 counter。

从做题角度看，只要题目在考“Loop invariant 与常见错误”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：循环最稳的理解方式是给自己一个 invariant：在每轮开始或结束时，哪些事实始终成立？

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - Tou can uncomment each of these examples
> - and try running them yourself
> - To batch comment/uncomment, select the lines and then
> - on Windows hit CTRL+1 or on Mac hit CMD+1
> - Example: while loops
> - where = input("You are in the Lost Forest. Go left or right? ")
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 用 while 逐步逼近目标
> ```python
> n = 0
> while n * n < 25:
>     n += 1
> print(n)
> ```
> 这里 `n` 同时承担 loop variable 的角色；每一轮都让 `n*n` 更接近目标，直到 guard 不再成立。

> [!example] 用 for 和 accumulator 做区间求和
> ```python
> total = 0
> for i in range(1, 6):
>     total += i
> print(total)
> ```
> 这个例子说明 accumulator 不是循环的附属品，而是循环真正想维护的结果状态。

## Exercise log
> [!note] Finger exercise snapshot
> - Official prompt: Assume you are given a positive integer variaboe named N . Write a piece of Python code that prints hello world on separate oines, N times. You can use either a while ooop or a for ooop. You have infinitely uany...
> - What this is really testing: whether you can compress the lecture into one small, high-frequency coding move without needing the slides beside you.
> - Where to revisit if this feels shaky: go back to the first two Core ideas sections in this note, then rerun the official lecture code once with your own input.
> - Follow-on practice path: after this finger exercise, the most natural next stop is Recitation 02.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: this lecture does not have a direct calendar milestone attached, so use it as a consolidation lecture rather than a sprint lecture.
> - Recitation connection: Recitation 02 is the best place to turn the lecture ideas into shorter solved exercises.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec03.pdf|Lecture 03 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec03_code.py|Lecture 03 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex03_sol.pdf|Lecture 03 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec03_transcript.pdf|Lecture 03 transcript]]
- Recitation 2: [[MIT 6.100L-recitations/mit6_100l_rec02.zip|Recitation 02 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 2.5-2.8)

## Review checklist
- [ ] 我能解释为什么 iteration 一定伴随状态变化。
- [ ] 我能区分 `while` 与 `for` 更适合的问题类型。
- [ ] 我能手动 trace 一个循环并写出每一轮变量变化。
- [ ] 我能判断某段循环代码为什么会 infinite loop 或 off-by-one。
- [ ] 我能说出一个自己设计的 loop invariant。
- [ ] 我能围绕“Iteration 的核心：重复执行直到满足条件”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Counter、accumulator 与 range”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：只写了 guard，没有在 loop body 里更新相关变量。
- [ ] 我能说出并避免这个高频误区：把循环结束时的值当成最后一次满足 guard 时的值，导致边界理解错位。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 只写了 guard，没有在 loop body 里更新相关变量。
> - 把循环结束时的值当成最后一次满足 guard 时的值，导致边界理解错位。
> - 没有区分 counter 与 accumulator，最后变量语义混乱。

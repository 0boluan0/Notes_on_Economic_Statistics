---
aliases:
  - MIT 6.100L Lecture 10
  - 6.100L L10
  - Lists and Mutability
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 10
---

# Lecture 10: Lists and Mutability

> [!tip] Hint
> - 我能区分 mutation 与 rebinding。
> - 我能解释为什么 list 上的操作有些会原地改对象，有些会返回新对象。
> - 我能说出遍历 list 时原地修改它为什么危险。
> - 我能看懂一个 bug 到底是值变了，还是名字换绑了。
> - 我能围绕本讲的主轴 “Mutability 改变了你推理程序的方式” / “常见 list 操作要分清副作用” / “写 list 代码时要显式管理状态变化”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释 mutation 与 rebinding 的区别。
> - 我能判断某段 list 代码有没有副作用。
> - 我能说明为什么共享同一个 list 会带来意外联动。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 5.3-5.5
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Mutability 改变了你推理程序的方式 / 常见 list 操作要分清副作用 / 写 list 代码时要显式管理状态变化
> - 这讲把 list 真正当作可变对象来使用，重点不再是‘怎么读’，而是‘怎么改’。
> - 一旦允许 mutation，程序就会出现时间维度：对象之前是什么、之后又变成了什么。
> - 理解 mutability 是后续 aliasing、cloning、class object state 的前提。

## Core ideas
### Mutability 改变了你推理程序的方式
对不可变对象，你只需关心当前值；对可变对象，你还必须关心‘这个对象会不会被后续代码在原地改掉’。
- list 支持原地更新，例如 `append`、`remove`、`sort`、通过 index 赋值等。
- mutation 会保留对象身份，只改变内容；rebind 则是让名字指向另一个对象。
- 调试可变对象时，只看变量名经常不够，还要关注对象是否被共享引用。
- 可变性让程序更高效、更自然，但也让状态推理更难。

> [!note] What to internalize
> - One-sentence takeaway: 对不可变对象，你只需关心当前值；对可变对象，你还必须关心‘这个对象会不会被后续代码在原地改掉’。
> - Review anchor: list 支持原地更新，例如 `append`、`remove`、`sort`、通过 index 赋值等。
> - Review anchor: mutation 会保留对象身份，只改变内容；rebind 则是让名字指向另一个对象。

从做题角度看，只要题目在考“Mutability 改变了你推理程序的方式”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：对不可变对象，你只需关心当前值；对可变对象，你还必须关心‘这个对象会不会被后续代码在原地改掉’。

### 常见 list 操作要分清副作用
学习 list API 时，不要只记‘它能做什么’，更要记‘它是原地改，还是返回新结果’。
- 有些操作像 `append` 直接修改原 list；有些表达式像切片会产生新 list。
- 如果不清楚副作用，代码表面上看起来像是在‘得到新答案’，实际上可能已经把旧数据改坏了。
- 对 list 做循环时，尤其要警惕边遍历边修改，因为这会干扰遍历顺序与长度。
- 可读代码通常会在 mutation 前后明确区分‘原数据’和‘要产生的新数据’。

> [!note] What to internalize
> - One-sentence takeaway: 学习 list API 时，不要只记‘它能做什么’，更要记‘它是原地改，还是返回新结果’。
> - Review anchor: 有些操作像 `append` 直接修改原 list；有些表达式像切片会产生新 list。
> - Review anchor: 如果不清楚副作用，代码表面上看起来像是在‘得到新答案’，实际上可能已经把旧数据改坏了。

从做题角度看，只要题目在考“常见 list 操作要分清副作用”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：学习 list API 时，不要只记‘它能做什么’，更要记‘它是原地改，还是返回新结果’。

### 写 list 代码时要显式管理状态变化
在 list 问题里，程序正确性很大程度上取决于你能否画出一条清晰的状态变化链。
- 如果某个 list 之后还要复用，先问自己：这里应该 mutate，还是应该复制后再操作？
- 边界情况常常出现在空 list、单元素 list、重复元素和删除操作上。
- 一段 list 代码越依赖隐式副作用，后面就越难 debug。
- ‘这一步改的是原对象还是新对象’应当成为你阅读 list 代码时的默认自问。

> [!note] What to internalize
> - One-sentence takeaway: 在 list 问题里，程序正确性很大程度上取决于你能否画出一条清晰的状态变化链。
> - Review anchor: 如果某个 list 之后还要复用，先问自己：这里应该 mutate，还是应该复制后再操作？
> - Review anchor: 边界情况常常出现在空 list、单元素 list、重复元素和删除操作上。

从做题角度看，只要题目在考“写 list 代码时要显式管理状态变化”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：在 list 问题里，程序正确性很大程度上取决于你能否画出一条清晰的状态变化链。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - Example: change value in a list and appending a value to a list
> - L = [2, 4, 3]
> - print(L)
> - L[1] = 5
> - L = L.append(5)
> - You try it #####################
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 区分 mutation 与 rebinding
> ```python
> L = [1, 2, 3]
> M = L
> L.append(4)
> print(L)
> print(M)
> ```
> 这里 `append` 改的是同一个 list 对象，所以 `M` 也会看到变化。这是 mutability 带来的核心现象。

> [!example] 原地修改 list 元素
> ```python
> numbers = [1, 2, 3]
> numbers[1] = 20
> print(numbers)
> ```
> 通过 index 赋值是最直接的 list mutation；对象身份不变，内容发生局部更新。

## Exercise log
> [!note] Finger exercise snapshot
> - Official prompt: Impoement the function that meets the specification beoow.: def all_true(n, Lf): """ n is an int Lf is a list of functions that take in an int and return a Boolean Returns True if each and every function in Lf returns...
> - What this is really testing: whether you can compress the lecture into one small, high-frequency coding move without needing the slides beside you.
> - Where to revisit if this feels shaky: go back to the first two Core ideas sections in this note, then rerun the official lecture code once with your own input.
> - Follow-on practice path: after this finger exercise, the most natural next stop is Recitation 05.
> - Homework bridge: this lecture is directly connected to the following calendar milestones: PS 2 halfway hand-in due.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: calendar shows this lecture touching the following milestones: PS 2 halfway hand-in due。读完本讲后，不应只会解释概念，还应能把它们搬到更长的程序里。
> - Recitation connection: Recitation 05 is the best place to turn the lecture ideas into shorter solved exercises.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec10.pdf|Lecture 10 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec10_code.py|Lecture 10 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex10_sol.pdf|Lecture 10 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec10_transcript.pdf|Lecture 10 transcript]]
- Recitation 5: [[MIT 6.100L-recitations/mit6_100l_rec05.zip|Recitation 05 materials]]
- PS 2 halfway hand-in due: [[MIT 6.100L-problem-sets/mit6_100l_ps2.pdf|PS2 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps2_code.zip|PS2 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 5.3-5.5)

## Review checklist
- [ ] 我能解释 mutation 与 rebinding 的区别。
- [ ] 我能判断某段 list 代码有没有副作用。
- [ ] 我能说明为什么共享同一个 list 会带来意外联动。
- [ ] 我能在需要保留旧数据时选择复制而不是原地修改。
- [ ] 我能为 list 操作写出清楚的状态变化说明。
- [ ] 我能围绕“Mutability 改变了你推理程序的方式”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“常见 list 操作要分清副作用”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：把 mutation 与 rebinding 混成一回事，导致共享对象时完全没意识到副作用。
- [ ] 我能说出并避免这个高频误区：遍历 list 的同时删除或插入元素，结果逻辑乱掉。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 把 mutation 与 rebinding 混成一回事，导致共享对象时完全没意识到副作用。
> - 遍历 list 的同时删除或插入元素，结果逻辑乱掉。
> - 只记方法名，不记它是否原地修改对象。

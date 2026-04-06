---
aliases:
  - MIT 6.100L Lecture 09
  - 6.100L L09
  - Lambda Functions, Tuples, and Lists
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 09
---

# Lecture 09: Lambda Functions, Tuples, and Lists

> [!tip] Hint
> - 我能解释 lambda 适合解决什么问题，以及何时不该滥用它。
> - 我能区分 tuple 的 immutability 与 list 的 mutability。
> - 我能说明 sequence 数据结构为何重要：它让同一种循环骨架可以处理很多对象。
> - 我能说出 list 和 tuple 在建模上的不同侧重点。
> - 我能围绕本讲的主轴 “Lambda：轻量级匿名函数” / “Tuple：固定、轻量、不可变的 sequence” / “List：可变 sequence 的入口”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释 lambda 的适用场景与局限。
> - 我能比较 tuple 与 list 在 mutability 和用途上的差异。
> - 我能用 tuple packing / unpacking 组织多值返回。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 5.1-5.3
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Lambda：轻量级匿名函数 / Tuple：固定、轻量、不可变的 sequence / List：可变 sequence 的入口
> - 这讲把‘函数作为对象’和‘一组值作为一个对象’这两条线接起来。
> - Tuple 常用来表达固定结构的数据，list 常用来表达可变集合；这会直接影响后续算法与 bug 类型。
> - 后面的 mutability、aliasing、sorting、classes 都会不断回到这里的 sequence 直觉。

## Core ideas
### Lambda：轻量级匿名函数
lambda 不是用来替代所有普通函数的，而是用来表达那些非常短、只在局部使用一次的小行为。
- 当某个函数逻辑非常短，而且只是为了当参数传给别的函数时，lambda 很方便。
- lambda 的价值在于减少样板代码，而不是让逻辑变得神秘。
- 一旦逻辑开始复杂、需要注释或多步推导，就该回到 `def`。
- 阅读 lambda 时，先问清楚它接收什么输入、返回什么输出。

> [!note] What to internalize
> - One-sentence takeaway: lambda 不是用来替代所有普通函数的，而是用来表达那些非常短、只在局部使用一次的小行为。
> - Review anchor: 当某个函数逻辑非常短，而且只是为了当参数传给别的函数时，lambda 很方便。
> - Review anchor: lambda 的价值在于减少样板代码，而不是让逻辑变得神秘。

从做题角度看，只要题目在考“Lambda：轻量级匿名函数”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：lambda 不是用来替代所有普通函数的，而是用来表达那些非常短、只在局部使用一次的小行为。

### Tuple：固定、轻量、不可变的 sequence
tuple 最适合表示‘这一组东西天然应该绑在一起，但不打算在原地修改’的结构。
- tuple 的 immutability 意味着创建后不能通过 index 改元素，这让它更稳定、更适合做返回值或坐标记录。
- tuple packing / unpacking 让多值返回、位置绑定变得自然。
- 你可以把 tuple 理解成‘带顺序的记录’，尤其适合固定字段数量的小结构。
- 不可变不是限制，而是在设计上强调‘这份结构代表的是一个事实，而不是可编辑容器’。

> [!note] What to internalize
> - One-sentence takeaway: tuple 最适合表示‘这一组东西天然应该绑在一起，但不打算在原地修改’的结构。
> - Review anchor: tuple 的 immutability 意味着创建后不能通过 index 改元素，这让它更稳定、更适合做返回值或坐标记录。
> - Review anchor: tuple packing / unpacking 让多值返回、位置绑定变得自然。

从做题角度看，只要题目在考“Tuple：固定、轻量、不可变的 sequence”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：tuple 最适合表示‘这一组东西天然应该绑在一起，但不打算在原地修改’的结构。

### List：可变 sequence 的入口
list 比 tuple 更灵活，因为它支持添加、删除、替换元素；但这种灵活也带来了更多需要管理的状态变化。
- list 适合表示一批数量可变、顺序重要、以后还会被修改的数据。
- indexing、slicing、membership test、遍历方式都和字符串、tuple 一脉相承。
- 一旦对象可变，程序正确性就不只取决于‘当前值是什么’，还取决于‘谁可能在后面改它’。
- list 的 mutability 是下一讲 aliasing 与 cloning 的真正背景。

> [!note] What to internalize
> - One-sentence takeaway: list 比 tuple 更灵活，因为它支持添加、删除、替换元素；但这种灵活也带来了更多需要管理的状态变化。
> - Review anchor: list 适合表示一批数量可变、顺序重要、以后还会被修改的数据。
> - Review anchor: indexing、slicing、membership test、遍历方式都和字符串、tuple 一脉相承。

从做题角度看，只要题目在考“List：可变 sequence 的入口”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：list 比 tuple 更灵活，因为它支持添加、删除、替换元素；但这种灵活也带来了更多需要管理的状态变化。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - print('apply with is_5:',apply(is_5,10))
> - print('apply with anon fcn:', apply(lambda x: x==5, 100))
> - Shown another way, the following are equivalent
> - is_even(8) # returns True
> - (lambda x: x%2==0)(8) # returns True
> - 1. What does this print?
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 用 lambda 表示一次性的小行为
> ```python
> double = lambda x: x * 2
> print(double(5))
> ```
> 这个例子只想说明 lambda 的位置：它适合极短逻辑。只要逻辑稍微复杂，就应该换回 `def`。

> [!example] 用 tuple 返回多个结果
> ```python
> def quotient_and_remainder(a, b):
>     return a // b, a % b
>
> q, r = quotient_and_remainder(17, 5)
> print(q, r)
> ```
> tuple 让多值返回自然可读，同时保留顺序结构。

## Exercise log
> [!note] Finger exercise snapshot
> - Official prompt: Impoement the function that meets the specification beoow.: def dot_product(tA, tB): """ tA: a tuple of numbers tB: a tuple of numbers of the same length as tA Assumes tA and tB are the same length. Returns a tuple...
> - What this is really testing: whether you can compress the lecture into one small, high-frequency coding move without needing the slides beside you.
> - Where to revisit if this feels shaky: go back to the first two Core ideas sections in this note, then rerun the official lecture code once with your own input.
> - Follow-on practice path: after this finger exercise, the most natural next stop is Recitation 04.
> - Homework bridge: this lecture is directly connected to the following calendar milestones: PS 2 out, PS 1 due.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: calendar shows this lecture touching the following milestones: PS 2 out；PS 1 due。读完本讲后，不应只会解释概念，还应能把它们搬到更长的程序里。
> - Recitation connection: Recitation 04 is the best place to turn the lecture ideas into shorter solved exercises.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec09.pdf|Lecture 09 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec09_code.py|Lecture 09 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex09_sol.pdf|Lecture 09 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec09_transcript.pdf|Lecture 09 transcript]]
- Recitation 4: [[MIT 6.100L-recitations/mit6_100l_rec04.zip|Recitation 04 materials]]
- PS 2 out: [[MIT 6.100L-problem-sets/mit6_100l_ps2.pdf|PS2 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps2_code.zip|PS2 starter code]]
- PS 1 due: [[MIT 6.100L-problem-sets/mit6_100l_ps1.pdf|PS1 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps1_code.zip|PS1 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 5.1-5.3)

## Review checklist
- [ ] 我能解释 lambda 的适用场景与局限。
- [ ] 我能比较 tuple 与 list 在 mutability 和用途上的差异。
- [ ] 我能用 tuple packing / unpacking 组织多值返回。
- [ ] 我能说明为什么 sequence 思维能复用前面学过的循环模式。
- [ ] 我能为一个具体问题选择更适合的 sequence 类型。
- [ ] 我能围绕“Lambda：轻量级匿名函数”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Tuple：固定、轻量、不可变的 sequence”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：为了炫技把复杂逻辑都塞进 lambda。
- [ ] 我能说出并避免这个高频误区：没有意识到 tuple 与 list 的 mutability 不同，导致建模选错数据结构。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 为了炫技把复杂逻辑都塞进 lambda。
> - 没有意识到 tuple 与 list 的 mutability 不同，导致建模选错数据结构。
> - 把 sequence 当成一堆散乱变量，没有利用统一的遍历与索引思维。

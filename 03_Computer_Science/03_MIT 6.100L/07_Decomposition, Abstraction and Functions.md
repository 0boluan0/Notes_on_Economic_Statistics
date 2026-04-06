---
aliases:
  - MIT 6.100L Lecture 07
  - 6.100L L07
  - Decomposition, Abstraction, and Functions
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 07
---

# Lecture 07: Decomposition, Abstraction, and Functions

> [!tip] Hint
> - 我能解释为什么函数是 decomposition 的基本单位。
> - 我能区分 specification、implementation、return value 三者的角色。
> - 我能说明 abstraction 的价值：调用者不需要知道内部细节也能正确使用函数。
> - 我能自己设计一个函数接口，而不是只会把代码挪进 def 里。
> - 我能围绕本讲的主轴 “Decomposition：把大问题拆成可命名的小动作” / “Specification 与 abstraction barrier” / “Return value、print 与程序组织”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释 decomposition 为什么能降低程序复杂度。
> - 我能为一个函数写出清楚的 specification。
> - 我能区分 print 与 return 在程序中的不同作用。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 4.1-4.2
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Decomposition：把大问题拆成可命名的小动作 / Specification 与 abstraction barrier / Return value、print 与程序组织
> - 前面几讲主要在写一段从上到下的脚本；这一讲开始学习如何把程序切成可复用、可理解的小模块。
> - 函数不仅是语法结构，更是‘给一个动作命名’并隐藏内部细节的抽象工具。
> - 后面所有高阶函数、递归、类方法，本质上都建立在这里的函数思维之上。

## Core ideas
### Decomposition：把大问题拆成可命名的小动作
当一段代码同时承担太多责任时，它就难以读、难以测、难以复用。函数的第一价值就是把职责切开。
- 如果某段逻辑在多个地方都会出现，或者它本身值得一个清晰名字，就应该考虑提成函数。
- 好 decomposition 的目标不是让函数数量变多，而是让每个函数有单一、清楚的责任。
- 函数名应该描述‘它做什么’，而不是它内部刚好用了什么技巧。
- 程序越复杂，越需要通过 decomposition 控制认知负担。

> [!note] What to internalize
> - One-sentence takeaway: 当一段代码同时承担太多责任时，它就难以读、难以测、难以复用。函数的第一价值就是把职责切开。
> - Review anchor: 如果某段逻辑在多个地方都会出现，或者它本身值得一个清晰名字，就应该考虑提成函数。
> - Review anchor: 好 decomposition 的目标不是让函数数量变多，而是让每个函数有单一、清楚的责任。

从做题角度看，只要题目在考“Decomposition：把大问题拆成可命名的小动作”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：当一段代码同时承担太多责任时，它就难以读、难以测、难以复用。函数的第一价值就是把职责切开。

### Specification 与 abstraction barrier
函数设计时要先回答‘外部世界需要知道什么’，再回答‘内部具体怎么做’。这条边界就是 abstraction barrier。
- specification 通常包括输入、输出、假设条件以及重要副作用。
- 调用者只需要信任 spec，不需要知道函数内部的循环、分支、临时变量怎么写。
- 一旦接口稳定，内部实现可以重写、优化、debug，而不必牵连所有调用点。
- 写函数时，先想清楚返回什么值，再想内部怎么算到它。

> [!note] What to internalize
> - One-sentence takeaway: 函数设计时要先回答‘外部世界需要知道什么’，再回答‘内部具体怎么做’。这条边界就是 abstraction barrier。
> - Review anchor: specification 通常包括输入、输出、假设条件以及重要副作用。
> - Review anchor: 调用者只需要信任 spec，不需要知道函数内部的循环、分支、临时变量怎么写。

从做题角度看，只要题目在考“Specification 与 abstraction barrier”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：函数设计时要先回答‘外部世界需要知道什么’，再回答‘内部具体怎么做’。这条边界就是 abstraction barrier。

### Return value、print 与程序组织
初学时最常见的混淆之一，是把‘把信息打印出来’误当成‘把结果交给其他代码使用’。
- `print(...)` 面向人类读者；`return ...` 面向程序的后续计算。
- 如果一个函数的结果还要被别的函数继续处理，就应该 `return`，而不是只 print。
- 函数内部可以包含局部中间量，但最终接口应该清楚而简洁。
- 设计函数时，优先思考数据流：输入怎样进入，结果怎样流出。

> [!note] What to internalize
> - One-sentence takeaway: 初学时最常见的混淆之一，是把‘把信息打印出来’误当成‘把结果交给其他代码使用’。
> - Review anchor: `print(...)` 面向人类读者；`return ...` 面向程序的后续计算。
> - Review anchor: 如果一个函数的结果还要被别的函数继续处理，就应该 `return`，而不是只 print。

从做题角度看，只要题目在考“Return value、print 与程序组织”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：初学时最常见的混淆之一，是把‘把信息打印出来’误当成‘把结果交给其他代码使用’。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - Example: applying functions to repeat same task many times
> - A very simple example of a function that has one
> - argument and returns one value
> - is_even(3) # <- returns False
> - is_even(8) # <- returns True
> - print(is_even(3)) # <- prints False
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 把重复逻辑提成函数
> ```python
> def area_of_circle(radius):
>     pi = 355 / 113
>     return pi * (radius ** 2)
>
> print(area_of_circle(2.2))
> ```
> 函数把‘求圆面积’这个动作命名了出来。调用者只关心半径与返回值，不需要再看内部公式细节。

> [!example] 区分 print 与 return
> ```python
> def double_bad(x):
>     print(x * 2)
>
> def double_good(x):
>     return x * 2
>
> result = double_good(4) + 1
> print(result)
> ```
> `double_bad` 只能把结果展示出来，不能参与下一步计算；`double_good` 才真正把结果交给程序继续使用。

## Exercise log
> [!warning] No official finger exercise
> - Calendar explicitly marks this lecture as having no official finger exercise.
> - Use the review checklist, the lecture code, and the linked recitation / problem set materials as the primary self-test for this lecture.
> - For this lecture, a good replacement for the missing finger exercise is: hand-trace one representative example from the code, then write a fresh one from memory.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: this lecture does not have a direct calendar milestone attached, so use it as a consolidation lecture rather than a sprint lecture.
> - Recitation connection: Recitation 03 is the best place to turn the lecture ideas into shorter solved exercises.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec07.pdf|Lecture 07 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec07_code.py|Lecture 07 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex07_sol.pdf|Lecture 07 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec07_transcript.pdf|Lecture 07 transcript]]
- Recitation 3: [[MIT 6.100L-recitations/mit6_100l_rec03.zip|Recitation 03 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 4.1-4.2)

## Review checklist
- [ ] 我能解释 decomposition 为什么能降低程序复杂度。
- [ ] 我能为一个函数写出清楚的 specification。
- [ ] 我能区分 print 与 return 在程序中的不同作用。
- [ ] 我能判断什么时候应该把一段逻辑提成函数。
- [ ] 我能自己设计一个简单但接口清楚的函数。
- [ ] 我能围绕“Decomposition：把大问题拆成可命名的小动作”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Specification 与 abstraction barrier”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：把所有代码都塞进一个大脚本，不愿意做 decomposition。
- [ ] 我能说出并避免这个高频误区：写函数时只顾内部实现，没有先定义清楚输入/输出规范。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 把所有代码都塞进一个大脚本，不愿意做 decomposition。
> - 写函数时只顾内部实现，没有先定义清楚输入/输出规范。
> - 用 print 代替 return，导致结果无法被复用。

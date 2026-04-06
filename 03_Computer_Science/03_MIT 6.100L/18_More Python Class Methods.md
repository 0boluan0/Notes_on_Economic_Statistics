---
aliases:
  - MIT 6.100L Lecture 18
  - 6.100L L18
  - More Python Class Methods
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 18
---

# Lecture 18: More Python Class Methods

> [!tip] Hint
> - 我能解释 dunder method 为什么能让自定义对象更像内置对象。
> - 我能说明对象表示、比较、运算行为是如何被方法协议接管的。
> - 我能区分‘对象能做什么’与‘对象如何展示自己’。
> - 我能把 class 设计看成接口协议设计，而不仅是存几个 attribute。
> - 我能围绕本讲的主轴 “Dunder methods 让对象接入 Python 协议” / “对象表示与对象语义是两个层面” / “方法设计要围绕对象不变量”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释 dunder method 的本质是协议对接。
> - 我能区分对象表示与对象语义。
> - 我能写出一个最简单的 `__str__` 或 `__len__`。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 10.1
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Dunder methods 让对象接入 Python 协议 / 对象表示与对象语义是两个层面 / 方法设计要围绕对象不变量
> - 上一讲建立了 class 的基本结构，这一讲继续把对象打磨成‘像 Python 原生对象一样自然使用’的抽象。
> - dunder methods 的重点不是记名字，而是理解 Python 通过协议来决定对象支持哪些行为。
> - 对象设计在这里开始真正接近工程实践：不仅要能存状态，还要能友好地参与比较、打印、组合。

## Core ideas
### Dunder methods 让对象接入 Python 协议
像 `__str__`、`__repr__`、`__len__`、比较运算相关方法，本质上是在告诉 Python：遇到某种语法时，应该如何解释这个对象。
- 如果没有这些方法，自定义对象往往只能‘存在’，却很难被自然打印、比较或组合。
- dunder method 让对象行为与语言语法接轨，所以它们本质上是接口协议的一部分。
- 学习它们时不要只背名称，更要问：这个协议对应语言里的哪个操作？
- 当对象接口更贴近语言习惯，调用者理解成本会显著降低。

> [!note] What to internalize
> - One-sentence takeaway: 像 `__str__`、`__repr__`、`__len__`、比较运算相关方法，本质上是在告诉 Python：遇到某种语法时，应该如何解释这个对象。
> - Review anchor: 如果没有这些方法，自定义对象往往只能‘存在’，却很难被自然打印、比较或组合。
> - Review anchor: dunder method 让对象行为与语言语法接轨，所以它们本质上是接口协议的一部分。

从做题角度看，只要题目在考“Dunder methods 让对象接入 Python 协议”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：像 `__str__`、`__repr__`、`__len__`、比较运算相关方法，本质上是在告诉 Python：遇到某种语法时，应该如何解释这个对象。

### 对象表示与对象语义是两个层面
一个对象既有内部真实状态，也有对外展示方式。把两者分清，才能写出对人和对程序都友好的类。
- `__str__` 更偏向给人读，强调简洁可读；`__repr__` 更偏向给开发者看，强调信息完整和可调试性。
- 显示方式不会改变对象语义，但会极大影响 debug 与使用体验。
- 当类越来越复杂，好的对象表示常常会成为定位问题的第一入口。
- 所以对象设计里，‘怎么被看见’和‘怎么被使用’都值得显式考虑。

> [!note] What to internalize
> - One-sentence takeaway: 一个对象既有内部真实状态，也有对外展示方式。把两者分清，才能写出对人和对程序都友好的类。
> - Review anchor: `__str__` 更偏向给人读，强调简洁可读；`__repr__` 更偏向给开发者看，强调信息完整和可调试性。
> - Review anchor: 显示方式不会改变对象语义，但会极大影响 debug 与使用体验。

从做题角度看，只要题目在考“对象表示与对象语义是两个层面”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：一个对象既有内部真实状态，也有对外展示方式。把两者分清，才能写出对人和对程序都友好的类。

### 方法设计要围绕对象不变量
类越复杂，越要注意所有方法都应维护对象状态的一致性，而不是各写各的。
- 如果两个方法对同一状态有不同隐含假设，很快就会出现对象内部自相矛盾。
- dunder methods 也不例外：它们应当反映对象真实语义，而不是随手拼一个形式上的实现。
- 面向对象设计真正难的地方，在于协调多个方法对同一状态的共同约束。
- 这也是为什么类设计常常比单个函数设计更需要前置思考。

> [!note] What to internalize
> - One-sentence takeaway: 类越复杂，越要注意所有方法都应维护对象状态的一致性，而不是各写各的。
> - Review anchor: 如果两个方法对同一状态有不同隐含假设，很快就会出现对象内部自相矛盾。
> - Review anchor: dunder methods 也不例外：它们应当反映对象真实语义，而不是随手拼一个形式上的实现。

从做题角度看，只要题目在考“方法设计要围绕对象不变量”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：类越复杂，越要注意所有方法都应维护对象状态的一致性，而不是各写各的。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - Example: simple Coordinate class
> - #Print a coordinate object's data attributes
> - print(f"c's x is {c.x} and origin's x is {origin.x}")
> - #These are equivalent calls
> - print(c.distance(origin))
> - print(Coordinate.distance(c, origin))
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 自定义对象的可读表示
> ```python
> class Point:
>     def __init__(self, x, y):
>         self.x = x
>         self.y = y
>
>     def __str__(self):
>         return f"Point({self.x}, {self.y})"
>
> print(Point(3, 4))
> ```
> 实现 `__str__` 后，对象在 print 时会更可读，这对调试和交互非常有帮助。

> [!example] 让对象支持长度协议
> ```python
> class Deck:
>     def __init__(self, cards):
>         self.cards = cards
>
>     def __len__(self):
>         return len(self.cards)
>
> print(len(Deck([1, 2, 3])))
> ```
> 这里 `len(obj)` 能工作，不是因为 `len` 特别认识这个类，而是因为类实现了相应协议方法。

## Exercise log
> [!note] Finger exercise snapshot
> - Official prompt: Write the class according to the specifications below: class Circle(): def __init__(self, radius): """ Initializes self with radius """ # your code here def get_radius(self): """ Returns the radius of self """ # your...
> - Official solution sketch:
> ```python
> class Circle():
> def __init__(self, radius):
> self.r = radius
> def get_radius(self):
> return self.r
> def __add__(self, c):
> ```
> - What this is really testing: whether you can compress the lecture into one small, high-frequency coding move without needing the slides beside you.
> - Where to revisit if this feels shaky: go back to the first two Core ideas sections in this note, then rerun the official lecture code once with your own input.
> - Homework bridge: this lecture is directly connected to the following calendar milestones: PS 4 halfway hand-in due.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: calendar shows this lecture touching the following milestones: PS 4 halfway hand-in due。读完本讲后，不应只会解释概念，还应能把它们搬到更长的程序里。
> - Recitation connection: there is no recitation attached to this lecture week in the official calendar.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec18.pdf|Lecture 18 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec18_code.py|Lecture 18 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex18_sol.pdf|Lecture 18 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec18_transcript.pdf|Lecture 18 transcript]]
- Recitation: none attached to this lecture week
- PS 4 halfway hand-in due: [[MIT 6.100L-problem-sets/mit6_100l_ps4.pdf|PS4 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps4_code.zip|PS4 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 10.1)

## Review checklist
- [ ] 我能解释 dunder method 的本质是协议对接。
- [ ] 我能区分对象表示与对象语义。
- [ ] 我能写出一个最简单的 `__str__` 或 `__len__`。
- [ ] 我能说明为什么方法设计要共同维护对象不变量。
- [ ] 我能看懂一个自定义类为什么能参与内置语法。
- [ ] 我能围绕“Dunder methods 让对象接入 Python 协议”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“对象表示与对象语义是两个层面”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：把 dunder method 当成死记硬背名单，而不理解它对应的语言协议。
- [ ] 我能说出并避免这个高频误区：对象打印出来毫无信息，导致调试困难。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 把 dunder method 当成死记硬背名单，而不理解它对应的语言协议。
> - 对象打印出来毫无信息，导致调试困难。
> - 多个方法对对象状态的理解不一致，破坏不变量。

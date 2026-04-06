---
aliases:
  - MIT 6.100L Lecture 08
  - 6.100L L08
  - Functions as Objects
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 08
---

# Lecture 08: Functions as Objects

> [!tip] Hint
> - 我能说明函数为什么也是对象，因此可以被赋值、传参、返回。
> - 我能解释 scope 与 environment 如何影响变量查找。
> - 我能区分调用函数、把函数本身当值传递、以及函数返回值三件事。
> - 我能看懂一个 higher-order function 在做什么，而不是被括号迷惑。
> - 我能围绕本讲的主轴 “Functions are first-class objects” / “Scope 与 environment：名字到底去哪里找” / “Higher-order thinking：把行为作为参数”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释为什么函数在 Python 中也是对象。
> - 我能说清楚一次函数调用会创建什么新的 environment。
> - 我能区分‘传函数’与‘传函数调用结果’。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 4.3-4.6
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Functions are first-class objects / Scope 与 environment：名字到底去哪里找 / Higher-order thinking：把行为作为参数
> - 这一讲把函数从‘程序里的一个段落’升级成‘可以像其他值一样被操作的对象’。
> - 理解 environment 和 scope，是理解函数调用、局部变量、后续递归与类方法的关键。
> - 函数一旦是一等公民，程序就可以把行为本身当成数据来组织。

## Core ideas
### Functions are first-class objects
在 Python 里，函数不是语法附属品，而是可以绑定到名字、放进数据结构、作为参数传来传去的对象。
- 写 `f = abs` 是把函数对象绑定给新名字；写 `f = abs()` 则是先调用再绑定返回值，这两者完全不同。
- 因为函数是对象，你可以把‘选哪种行为’也变成程序运行时的决定。
- 这让代码更通用：同一个框架可以接受不同函数作为策略输入。
- 一旦把函数当对象看，很多“奇怪的括号”其实只是对象与调用的区别。

> [!note] What to internalize
> - One-sentence takeaway: 在 Python 里，函数不是语法附属品，而是可以绑定到名字、放进数据结构、作为参数传来传去的对象。
> - Review anchor: 写 `f = abs` 是把函数对象绑定给新名字；写 `f = abs()` 则是先调用再绑定返回值，这两者完全不同。
> - Review anchor: 因为函数是对象，你可以把‘选哪种行为’也变成程序运行时的决定。

从做题角度看，只要题目在考“Functions are first-class objects”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：在 Python 里，函数不是语法附属品，而是可以绑定到名字、放进数据结构、作为参数传来传去的对象。

### Scope 与 environment：名字到底去哪里找
函数调用会创建新的 local environment。名字查找不是随便发生的，而是按作用域规则逐层寻找。
- 函数内部定义的局部变量默认只在本次调用有效，调用结束后环境就消失。
- 如果局部作用域里没有某个名字，Python 才会继续往外层找。
- 理解 scope 能帮助你解释‘为什么这个变量在这里能用、在那里报错’。
- 写函数时要尽量减少对外部可变状态的隐式依赖，否则程序会难以推理。

> [!note] What to internalize
> - One-sentence takeaway: 函数调用会创建新的 local environment。名字查找不是随便发生的，而是按作用域规则逐层寻找。
> - Review anchor: 函数内部定义的局部变量默认只在本次调用有效，调用结束后环境就消失。
> - Review anchor: 如果局部作用域里没有某个名字，Python 才会继续往外层找。

从做题角度看，只要题目在考“Scope 与 environment：名字到底去哪里找”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：函数调用会创建新的 local environment。名字查找不是随便发生的，而是按作用域规则逐层寻找。

### Higher-order thinking：把行为作为参数
函数对象最有价值的地方，是允许你把‘要做什么操作’抽象成参数，而不是写死在函数体里。
- 如果一个框架只是‘对数据做某种处理’，那么那种处理方式就很适合抽象成函数参数。
- 这会让代码从‘只会做一种事’变成‘在同一框架下做一类事’。
- 学习 higher-order function 的关键不是炫技，而是理解抽象层次提高后，重复代码会变少。
- 后面的 `lambda`、排序 key、map/filter 风格都属于这一类思想。

> [!note] What to internalize
> - One-sentence takeaway: 函数对象最有价值的地方，是允许你把‘要做什么操作’抽象成参数，而不是写死在函数体里。
> - Review anchor: 如果一个框架只是‘对数据做某种处理’，那么那种处理方式就很适合抽象成函数参数。
> - Review anchor: 这会让代码从‘只会做一种事’变成‘在同一框架下做一类事’。

从做题角度看，只要题目在考“Higher-order thinking：把行为作为参数”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：函数对象最有价值的地方，是允许你把‘要做什么操作’抽象成参数，而不是写死在函数体里。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - Example: combinations of print and return
> - is_even_with_return(3) # -> False
> - print(is_even_with_return(3)) # -> print(False)
> - return None
> - is_even_without_return(3) # -> None
> - print(is_even_without_return(3)) # -> print(None)
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 把函数绑定给变量
> ```python
> def square(x):
>     return x * x
>
> f = square
> print(f(5))
> ```
> 这里 `f` 绑定的是函数对象本身，所以后面 `f(5)` 与 `square(5)` 完全等价。

> [!example] 把函数当参数传递
> ```python
> def apply_twice(fn, x):
>     return fn(fn(x))
>
> def add_one(y):
>     return y + 1
>
> print(apply_twice(add_one, 3))
> ```
> 这个例子体现 higher-order function 的核心：框架是 `apply_twice`，具体行为由传入的函数决定。

## Exercise log
> [!note] Finger exercise snapshot
> - Official prompt: Impoement the function that meets the specification beoow.: def same_chars(s1, s2): """ s1 and s2 are strings Returns boolean True is a character in s1 is also in s2, and vice versa. If a character only exists in one of...
> - What this is really testing: whether you can compress the lecture into one small, high-frequency coding move without needing the slides beside you.
> - Where to revisit if this feels shaky: go back to the first two Core ideas sections in this note, then rerun the official lecture code once with your own input.
> - Follow-on practice path: after this finger exercise, the most natural next stop is Recitation 04.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: this lecture does not have a direct calendar milestone attached, so use it as a consolidation lecture rather than a sprint lecture.
> - Recitation connection: Recitation 04 is the best place to turn the lecture ideas into shorter solved exercises.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec08.pdf|Lecture 08 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec08_code.py|Lecture 08 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex08_sol.pdf|Lecture 08 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec08_transcript.pdf|Lecture 08 transcript]]
- Recitation 4: [[MIT 6.100L-recitations/mit6_100l_rec04.zip|Recitation 04 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 4.3-4.6)

## Review checklist
- [ ] 我能解释为什么函数在 Python 中也是对象。
- [ ] 我能说清楚一次函数调用会创建什么新的 environment。
- [ ] 我能区分‘传函数’与‘传函数调用结果’。
- [ ] 我能读懂一个 higher-order function 的数据流。
- [ ] 我能判断一个重复逻辑是否适合抽象成函数参数。
- [ ] 我能围绕“Functions are first-class objects”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Scope 与 environment：名字到底去哪里找”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：把 `f` 与 `f()` 混为一谈，不知道自己传的是函数还是返回值。
- [ ] 我能说出并避免这个高频误区：对局部变量与外层变量的查找规则没有概念，导致 scope bug。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 把 `f` 与 `f()` 混为一谈，不知道自己传的是函数还是返回值。
> - 对局部变量与外层变量的查找规则没有概念，导致 scope bug。
> - 为了复用逻辑仍然硬复制代码，而不是把行为抽象成函数参数。

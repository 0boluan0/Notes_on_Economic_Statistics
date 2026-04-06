---
aliases:
  - MIT 6.100L Lecture 02
  - 6.100L L02
  - Strings, Input/Output, and Branching
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 02
---

# Lecture 02: Strings, Input/Output, and Branching

> [!tip] Hint
> - 我能解释为什么 `input()` 默认返回 string，以及这会给分支判断带来什么坑。
> - 我能描述 string 的 indexing / slicing 规则，并知道 string 是 immutable。
> - 我能写出一个 if/elif/else 结构并说明缩进为什么决定语义。
> - 我能区分“显示给用户看的格式化输出”和“程序内部真正用来计算的值”。
> - 我能围绕本讲的主轴 “Strings 是 sequence，不是“特殊的数字”” / “Input/Output 的关键是类型转换” / “Branching 让程序根据条件走不同路径”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释为什么字符串是 sequence，以及这对 indexing / slicing 有什么影响。
> - 我能准确说出 `input()` 的返回类型，并解释什么时候必须转换。
> - 我能写出一个覆盖完整、没有遗漏的 if/elif/else 结构。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 2.3-2.4
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Strings 是 sequence，不是“特殊的数字” / Input/Output 的关键是类型转换 / Branching 让程序根据条件走不同路径
> - 本讲把程序从“只会算”推进到“能和用户交互、能根据条件选择路径”。
> - 字符串、输入输出与分支构成几乎所有交互式程序的骨架：读数据、判断情况、给反馈。
> - 后面循环和函数都会在这里的基础上继续堆：先决定一轮做什么，再讨论做多少轮。

## Core ideas
### Strings 是 sequence，不是“特殊的数字”
字符串最重要的身份是 sequence：它有顺序、能取子串、能逐字符处理，但它不是拿来直接做数值运算的。
- 字符串用引号包住；单引号和双引号都行，关键是配对一致。
- indexing 让你取单个字符，slicing 让你取一段子串；写程序时要非常清楚起点是否包含、终点是否包含。
- 字符串是 immutable，所以“改字符串里的第 3 个字符”这种想法本身就错了；正确做法通常是拼出一个新字符串。
- 连接、重复、membership test（`in`）是字符串编程里的常用操作。

> [!note] What to internalize
> - One-sentence takeaway: 字符串最重要的身份是 sequence：它有顺序、能取子串、能逐字符处理，但它不是拿来直接做数值运算的。
> - Review anchor: 字符串用引号包住；单引号和双引号都行，关键是配对一致。
> - Review anchor: indexing 让你取单个字符，slicing 让你取一段子串；写程序时要非常清楚起点是否包含、终点是否包含。

从做题角度看，只要题目在考“Strings 是 sequence，不是“特殊的数字””相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：字符串最重要的身份是 sequence：它有顺序、能取子串、能逐字符处理，但它不是拿来直接做数值运算的。

### Input/Output 的关键是类型转换
用户和程序沟通时，屏幕上看到的内容与程序内部的数据表示不是一回事。输入输出一旦牵涉到数字，就要主动做类型转换。
- `input()` 永远先给你 string，所以 `'5' < '20'` 与 `5 < 20` 的语义完全不同。
- 格式化输出的目标是让信息更可读，例如 `f"Hello, {name}"`，它不改变内部数据的类型与结构。
- 把输入转成 `int` 或 `float` 是逻辑判断前的必要清洗步骤。
- 如果程序输出不对，先 print 中间值，再确认这些中间值的 type。

> [!note] What to internalize
> - One-sentence takeaway: 用户和程序沟通时，屏幕上看到的内容与程序内部的数据表示不是一回事。输入输出一旦牵涉到数字，就要主动做类型转换。
> - Review anchor: `input()` 永远先给你 string，所以 `'5' < '20'` 与 `5 < 20` 的语义完全不同。
> - Review anchor: 格式化输出的目标是让信息更可读，例如 `f"Hello, {name}"`，它不改变内部数据的类型与结构。

从做题角度看，只要题目在考“Input/Output 的关键是类型转换”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：用户和程序沟通时，屏幕上看到的内容与程序内部的数据表示不是一回事。输入输出一旦牵涉到数字，就要主动做类型转换。

### Branching 让程序根据条件走不同路径
分支不是“语法体操”，它的作用是让同一个程序在不同状态下选择不同策略。
- `if / elif / else` 把条件组织成互斥路径；顺序很重要，因为程序从上往下依次检查。
- 布尔表达式可以来自比较、逻辑连接词、membership test 等；写分支前先明确“我要区分的是哪几类情况”。
- Python 的 block 用缩进表示，所以缩进错误通常不是样式问题，而是语义错误。
- 好分支代码往往有清楚的覆盖关系：所有情况都被考虑到，而且没有重复判断。

> [!note] What to internalize
> - One-sentence takeaway: 分支不是“语法体操”，它的作用是让同一个程序在不同状态下选择不同策略。
> - Review anchor: `if / elif / else` 把条件组织成互斥路径；顺序很重要，因为程序从上往下依次检查。
> - Review anchor: 布尔表达式可以来自比较、逻辑连接词、membership test 等；写分支前先明确“我要区分的是哪几类情况”。

从做题角度看，只要题目在考“Branching 让程序根据条件走不同路径”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：分支不是“语法体操”，它的作用是让同一个程序在不同状态下选择不同策略。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - TYPE THIS IN THE CONSOLE -- STRINGS ##
> - TYPE THIS IN THE CONSOLE -- INDEXING ##
> - s[3] # this is an error
> - TYPE THIS IN THE CONSOLE -- SLICING ##
> - TYPE THIS IN THE CONSOLE - MANIPULATION ##
> - s[0] = 'b' # this is an error
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 把字符串输入变成可比较的数字
> ```python
> raw = input("Enter your age: ")
> age = int(raw)
> if age >= 18:
>     print("adult")
> else:
>     print("minor")
> ```
> 如果漏掉 `int(raw)`，程序比较的就不是数字大小，而是字符串的字典序，这会让逻辑悄悄跑偏。

> [!example] 用 slicing 拆解结构化文本
> ```python
> course = "6.100L"
> prefix = course[:1]
> number = course[2:5]
> print(prefix, number)
> ```
> 字符串处理的常见套路不是“逐个字符硬写”，而是先明确哪些位置对应什么含义，再用 indexing / slicing 拆出来。

## Exercise log
> [!note] Finger exercise snapshot
> - Official prompt: Assume you are given a variaboe named number (has a numericao vaoue). Write a piece of Python code that prints out one of the foooowing strings: positive if the variaboe number is positive negative if the variaboe...
> - What this is really testing: whether you can compress the lecture into one small, high-frequency coding move without needing the slides beside you.
> - Where to revisit if this feels shaky: go back to the first two Core ideas sections in this note, then rerun the official lecture code once with your own input.
> - Follow-on practice path: after this finger exercise, the most natural next stop is Recitation 02.
> - Homework bridge: this lecture is directly connected to the following calendar milestones: PS 1 out.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: calendar shows this lecture touching the following milestones: PS 1 out。读完本讲后，不应只会解释概念，还应能把它们搬到更长的程序里。
> - Recitation connection: Recitation 02 is the best place to turn the lecture ideas into shorter solved exercises.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec02.pdf|Lecture 02 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec02_code.py|Lecture 02 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex02_sol.pdf|Lecture 02 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec02_transcript.pdf|Lecture 02 transcript]]
- Recitation 2: [[MIT 6.100L-recitations/mit6_100l_rec02.zip|Recitation 02 materials]]
- PS 1 out: [[MIT 6.100L-problem-sets/mit6_100l_ps1.pdf|PS1 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps1_code.zip|PS1 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 2.3-2.4)

## Review checklist
- [ ] 我能解释为什么字符串是 sequence，以及这对 indexing / slicing 有什么影响。
- [ ] 我能准确说出 `input()` 的返回类型，并解释什么时候必须转换。
- [ ] 我能写出一个覆盖完整、没有遗漏的 if/elif/else 结构。
- [ ] 我能判断一个 bug 是由缩进错误、类型错误还是条件顺序错误导致的。
- [ ] 我能自己写一个根据输入给出不同反馈的小程序。
- [ ] 我能围绕“Strings 是 sequence，不是“特殊的数字””自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Input/Output 的关键是类型转换”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：忘记 `input()` 返回的是 string，直接拿去做数值比较。
- [ ] 我能说出并避免这个高频误区：把 `elif` 写成多个独立的 `if`，导致本应互斥的逻辑重复触发。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 忘记 `input()` 返回的是 string，直接拿去做数值比较。
> - 把 `elif` 写成多个独立的 `if`，导致本应互斥的逻辑重复触发。
> - 对 slicing 的边界没有概念，写出 off-by-one 错误。

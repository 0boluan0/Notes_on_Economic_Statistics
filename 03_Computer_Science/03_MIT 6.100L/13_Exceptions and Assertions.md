---
aliases:
  - MIT 6.100L Lecture 13
  - 6.100L L13
  - Exceptions and Assertions
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 13
---

# Lecture 13: Exceptions and Assertions

> [!tip] Hint
> - 我能解释 exception 是什么，以及它和普通返回值为什么不是一回事。
> - 我能说出 try/except 的作用：隔离错误处理，而不是吞掉一切。
> - 我能说明 assertion 在程序里表达的是什么 contract。
> - 我能判断某个错误应该用条件分支处理，还是用异常机制处理。
> - 我能围绕本讲的主轴 “Exceptions 是程序对异常状态的显式信号” / “try / except 应该让错误处理更清楚，而不是更模糊” / “Assertions 把隐含假设写成可执行的检查”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释 exception 为什么不是普通返回值。
> - 我能写出只捕获特定异常的 try/except。
> - 我能说明 assertion 最适合表达哪类假设。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 9
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Exceptions 是程序对异常状态的显式信号 / try / except 应该让错误处理更清楚，而不是更模糊 / Assertions 把隐含假设写成可执行的检查
> - 前面的 testing/debugging 主要在定位问题，这一讲开始讨论如何在程序运行时显式处理异常路径。
> - 异常机制的重点不是‘不让程序报错’，而是让错误被更合理地暴露、传播或恢复。
> - assertion 则是一种更主动的自检：把本来只存在于脑子里的假设写进代码。

## Core ideas
### Exceptions 是程序对异常状态的显式信号
异常不是普通数据，它代表程序遇到了当前流程无法正常继续的情况。
- 如果把错误也塞进普通返回值，调用者很容易忘记检查；异常机制强迫你正视错误路径。
- 常见异常如 `ZeroDivisionError`、`TypeError`、`ValueError` 都在提醒‘当前输入或状态违背了预期’。
- 异常会沿调用栈向上传播，直到被匹配的 `except` 处理或最终让程序终止。
- 所以异常本质上也是一种控制流。

> [!note] What to internalize
> - One-sentence takeaway: 异常不是普通数据，它代表程序遇到了当前流程无法正常继续的情况。
> - Review anchor: 如果把错误也塞进普通返回值，调用者很容易忘记检查；异常机制强迫你正视错误路径。
> - Review anchor: 常见异常如 `ZeroDivisionError`、`TypeError`、`ValueError` 都在提醒‘当前输入或状态违背了预期’。

从做题角度看，只要题目在考“Exceptions 是程序对异常状态的显式信号”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：异常不是普通数据，它代表程序遇到了当前流程无法正常继续的情况。

### try / except 应该让错误处理更清楚，而不是更模糊
好的异常处理代码能把正常路径和错误恢复路径分开写，从而保持主逻辑清晰。
- 只捕获你真正知道如何处理的异常类型，避免一个裸 `except` 把所有问题都吃掉。
- 异常处理的目的是恢复、补救、记录或明确失败，而不是掩盖错误。
- 如果 except 块太大，往往说明你没有定位清楚风险点在哪里。
- 写异常处理时要问：如果这里出错，我到底打算让程序怎么继续？

> [!note] What to internalize
> - One-sentence takeaway: 好的异常处理代码能把正常路径和错误恢复路径分开写，从而保持主逻辑清晰。
> - Review anchor: 只捕获你真正知道如何处理的异常类型，避免一个裸 `except` 把所有问题都吃掉。
> - Review anchor: 异常处理的目的是恢复、补救、记录或明确失败，而不是掩盖错误。

从做题角度看，只要题目在考“try / except 应该让错误处理更清楚，而不是更模糊”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：好的异常处理代码能把正常路径和错误恢复路径分开写，从而保持主逻辑清晰。

### Assertions 把隐含假设写成可执行的检查
assert 不是给用户做输入校验，而是给程序员自己做内部一致性检查：如果这里不成立，说明程序逻辑已经偏了。
- assertion 常用来表达 precondition、postcondition 或重要 invariant。
- 它的价值在于把‘我默认这里应该成立’变成机器能立刻验证的事实。
- 当 assert 失败时，你得到的是更早、更靠近 bug 根源的反馈。
- 会写 assert，往往说明你已经能清楚说出程序依赖哪些逻辑假设。

> [!note] What to internalize
> - One-sentence takeaway: assert 不是给用户做输入校验，而是给程序员自己做内部一致性检查：如果这里不成立，说明程序逻辑已经偏了。
> - Review anchor: assertion 常用来表达 precondition、postcondition 或重要 invariant。
> - Review anchor: 它的价值在于把‘我默认这里应该成立’变成机器能立刻验证的事实。

从做题角度看，只要题目在考“Assertions 把隐含假设写成可执行的检查”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：assert 不是给用户做输入校验，而是给程序员自己做内部一致性检查：如果这里不成立，说明程序逻辑已经偏了。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - Example: Exceptions with summing digits in a string
> - Not using exceptions
> - print(sum_digits("123"))
> - print(sum_digits("123abc"))
> - Using exceptions around potentially problematic code
> - Print that an error happened and let the program keep going
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 只捕获自己知道如何处理的异常
> ```python
> def safe_divide(a, b):
>     try:
>         return a / b
>     except ZeroDivisionError:
>         return None
>
> print(safe_divide(10, 0))
> ```
> 这里的 except 很具体：只处理除零这一个已知问题，而不是把所有错误都吞掉。

> [!example] 用 assertion 写出函数依赖的假设
> ```python
> def average(nums):
>     assert len(nums) > 0, "nums must not be empty"
>     return sum(nums) / len(nums)
>
> print(average([1, 2, 3]))
> ```
> 这个 assert 把‘不能为空’从脑中假设变成了代码里的 contract。

## Exercise log
> [!note] Finger exercise snapshot
> - Official prompt: Impoement the function that meets the specification beoow.: def sum_str_lengths(L): """ L is a non-empty list containing either: * string elements or * a non-empty sublist of string elements Returns the sum of the...
> - What this is really testing: whether you can compress the lecture into one small, high-frequency coding move without needing the slides beside you.
> - Where to revisit if this feels shaky: go back to the first two Core ideas sections in this note, then rerun the official lecture code once with your own input.
> - Follow-on practice path: after this finger exercise, the most natural next stop is Recitation 07.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: this lecture does not have a direct calendar milestone attached, so use it as a consolidation lecture rather than a sprint lecture.
> - Recitation connection: Recitation 07 is the best place to turn the lecture ideas into shorter solved exercises.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec13.pdf|Lecture 13 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec13_code.py|Lecture 13 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex13_sol.pdf|Lecture 13 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec13_transcript.pdf|Lecture 13 transcript]]
- Recitation 7: [[MIT 6.100L-recitations/mit6_100l_rec07.zip|Recitation 07 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 9)

## Review checklist
- [ ] 我能解释 exception 为什么不是普通返回值。
- [ ] 我能写出只捕获特定异常的 try/except。
- [ ] 我能说明 assertion 最适合表达哪类假设。
- [ ] 我能判断一个错误应该由分支处理还是由异常传播。
- [ ] 我能说明‘吞掉异常’为什么危险。
- [ ] 我能围绕“Exceptions 是程序对异常状态的显式信号”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“try / except 应该让错误处理更清楚，而不是更模糊”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：用一个大而全的裸 `except` 掩盖真正的 bug。
- [ ] 我能说出并避免这个高频误区：把 assertion 当成所有输入校验的万能工具，而不是内部逻辑检查。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 用一个大而全的裸 `except` 掩盖真正的 bug。
> - 把 assertion 当成所有输入校验的万能工具，而不是内部逻辑检查。
> - 异常发生后没有明确策略，只是让程序继续跑下去。

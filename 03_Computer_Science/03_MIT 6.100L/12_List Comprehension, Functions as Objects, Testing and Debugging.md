---
aliases:
  - MIT 6.100L Lecture 12
  - 6.100L L12
  - List Comprehension, Functions as Objects, Testing, and Debugging
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 12
---

# Lecture 12: List Comprehension, Functions as Objects, Testing, and Debugging

> [!tip] Hint
> - 我能解释 keyword argument / default argument 如何改变函数接口的可用性。
> - 我能区分 glass-box testing 与 black-box testing。
> - 我能把 debugging 看成系统定位原因，而不是随机改代码。
> - 我能说明为什么测试覆盖的是‘思路空间’，而不是样例数量。
> - 我能围绕本讲的主轴 “函数接口设计：位置参数、keyword argument、default argument” / “Testing：从‘样例通过’升级到‘结构覆盖’” / “Debugging：观察、缩小范围、验证假设”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释 keyword argument 与 default argument 的设计价值。
> - 我能比较 black-box testing 与 glass-box testing 的关注点。
> - 我能为一个简单函数设计边界测试用例。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 4.4, Ch 8
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: 函数接口设计：位置参数、keyword argument、default argument / Testing：从‘样例通过’升级到‘结构覆盖’ / Debugging：观察、缩小范围、验证假设
> - 这讲把前面学过的函数、list comprehension 聚到一起，进一步讨论如何把代码写得更可测试、更易 debug。
> - 课程在这里开始强调工程习惯：正确程序不是只会跑通一个例子，而是要经得起系统测试。
> - 后面异常、assertion、classes 都会建立在这里的 testing/debugging 直觉上。

## Core ideas
### 函数接口设计：位置参数、keyword argument、default argument
当函数开始承担真正的抽象职责时，接口设计的重要性会迅速上升。调用方式越清楚，函数越容易被正确使用。
- keyword argument 让调用点更自解释，尤其当参数含义相近时更能减少误用。
- default argument 提供了合理默认值，让简单调用更轻量，但也要求你明确默认行为是否安全。
- 接口设计的目标不是让函数更花哨，而是让调用者更难犯错。
- 当参数多到难以记忆时，通常说明函数职责太重或接口需要重构。

> [!note] What to internalize
> - One-sentence takeaway: 当函数开始承担真正的抽象职责时，接口设计的重要性会迅速上升。调用方式越清楚，函数越容易被正确使用。
> - Review anchor: keyword argument 让调用点更自解释，尤其当参数含义相近时更能减少误用。
> - Review anchor: default argument 提供了合理默认值，让简单调用更轻量，但也要求你明确默认行为是否安全。

从做题角度看，只要题目在考“函数接口设计：位置参数、keyword argument、default argument”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：当函数开始承担真正的抽象职责时，接口设计的重要性会迅速上升。调用方式越清楚，函数越容易被正确使用。

### Testing：从‘样例通过’升级到‘结构覆盖’
测试不是表演程序能跑，而是主动寻找程序会坏在哪里。不同测试思想关心的是不同维度的覆盖。
- black-box testing 站在 specification 角度设计输入，不依赖你内部具体怎么实现。
- glass-box testing 站在实现细节角度补充测试，关注边界分支、循环路径、难走到的状态。
- 好测试不会只测‘典型值’，还会测边界值、空输入、非法输入、重复值等容易出错的区域。
- 测试案例的价值不在数量，而在是否覆盖了不同类别的行为。

> [!note] What to internalize
> - One-sentence takeaway: 测试不是表演程序能跑，而是主动寻找程序会坏在哪里。不同测试思想关心的是不同维度的覆盖。
> - Review anchor: black-box testing 站在 specification 角度设计输入，不依赖你内部具体怎么实现。
> - Review anchor: glass-box testing 站在实现细节角度补充测试，关注边界分支、循环路径、难走到的状态。

从做题角度看，只要题目在考“Testing：从‘样例通过’升级到‘结构覆盖’”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：测试不是表演程序能跑，而是主动寻找程序会坏在哪里。不同测试思想关心的是不同维度的覆盖。

### Debugging：观察、缩小范围、验证假设
debugging 最低效的方式是到处乱改。更有效的方法是先收集证据，再缩小 bug 所在的位置。
- 先复现 bug，再记录期望输出与实际输出的差异，避免在模糊印象上 debug。
- print 中间值、单步 trace、缩小输入规模，都是为了更快定位问题出现在哪一步。
- 每次只改一个假设相关点，然后重新运行验证；否则你根本不知道哪次修改真的修好了问题。
- 能被清楚解释的 bug，往往也更容易被根治。

> [!note] What to internalize
> - One-sentence takeaway: debugging 最低效的方式是到处乱改。更有效的方法是先收集证据，再缩小 bug 所在的位置。
> - Review anchor: 先复现 bug，再记录期望输出与实际输出的差异，避免在模糊印象上 debug。
> - Review anchor: print 中间值、单步 trace、缩小输入规模，都是为了更快定位问题出现在哪一步。

从做题角度看，只要题目在考“Debugging：观察、缩小范围、验证假设”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：debugging 最低效的方式是到处乱改。更有效的方法是先收集证据，再缩小 bug 所在的位置。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - WORDLE ###########
> - ASSUME YOU ARE GIVEN CODE FROM HERE.... ############
> - ....ASSUME YOU ARE GIVEN CODE UP TO HERE ############
> - THE CODE BELOW IS BUGGY #############
> - if is_a_real_word(guess, word_list) and is_correct_len(guess, wordle_len)
> - result = make_wordle(guess, secret)
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 用 keyword 与 default argument 提升接口可读性
> ```python
> def greet(name, punctuation="!"):
>     return f"Hello, {name}{punctuation}"
>
> print(greet("Ana"))
> print(greet("Ana", punctuation="?"))
> ```
> 默认值让常见调用更短，keyword argument 让特殊调用的意图一眼可读。

> [!example] 先 print 中间状态再 debug
> ```python
> def average(nums):
>     total = sum(nums)
>     print("debug total:", total)
>     return total / len(nums)
>
> print(average([1, 2, 3]))
> ```
> 临时打印中间值的目标不是永久留在代码里，而是帮助你验证对程序状态的假设是否正确。

## Exercise log
> [!note] Finger exercise snapshot
> - Official prompt: Impoement the function that meets the specification beoow.: def count_sqrts(nums_list): """ nums_list: a list Assumes that nums_list only contains positive numbers and that there are no duplicates. Returns how many...
> - What this is really testing: whether you can compress the lecture into one small, high-frequency coding move without needing the slides beside you.
> - Where to revisit if this feels shaky: go back to the first two Core ideas sections in this note, then rerun the official lecture code once with your own input.
> - Follow-on practice path: after this finger exercise, the most natural next stop is Recitation 06.
> - Homework bridge: this lecture is directly connected to the following calendar milestones: PS 3 out, PS 2 due.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: calendar shows this lecture touching the following milestones: PS 3 out；PS 2 due。读完本讲后，不应只会解释概念，还应能把它们搬到更长的程序里。
> - Recitation connection: Recitation 06 is the best place to turn the lecture ideas into shorter solved exercises.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec12.pdf|Lecture 12 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec12_code.zip|Lecture 12 code (zip)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex12_sol.pdf|Lecture 12 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec12_transcript.pdf|Lecture 12 transcript]]
- Recitation 6: [[MIT 6.100L-recitations/mit6_100l_rec06.zip|Recitation 06 materials]]
- PS 3 out: [[MIT 6.100L-problem-sets/mit6_100l_ps3.pdf|PS3 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps3_code.zip|PS3 starter code]]
- PS 2 due: [[MIT 6.100L-problem-sets/mit6_100l_ps2.pdf|PS2 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps2_code.zip|PS2 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 4.4, Ch 8)

## Review checklist
- [ ] 我能解释 keyword argument 与 default argument 的设计价值。
- [ ] 我能比较 black-box testing 与 glass-box testing 的关注点。
- [ ] 我能为一个简单函数设计边界测试用例。
- [ ] 我能描述一个系统 debugging 流程，而不是‘试试看改这里’。
- [ ] 我能判断某个函数接口是否已经不易使用。
- [ ] 我能围绕“函数接口设计：位置参数、keyword argument、default argument”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Testing：从‘样例通过’升级到‘结构覆盖’”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：把测试理解成随便挑几个例子试一下。
- [ ] 我能说出并避免这个高频误区：程序一出错就重写，而不是先定位哪一步违背了预期。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 把测试理解成随便挑几个例子试一下。
> - 程序一出错就重写，而不是先定位哪一步违背了预期。
> - default argument 设得随意，结果把‘默认行为’也变成了 bug 来源。

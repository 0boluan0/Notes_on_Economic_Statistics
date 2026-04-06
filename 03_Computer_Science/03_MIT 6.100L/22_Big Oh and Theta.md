---
aliases:
  - MIT 6.100L Lecture 22
  - 6.100L L22
  - Big Oh and Theta
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 22
---

# Lecture 22: Big Oh and Theta

> [!tip] Hint
> - 我能解释 Big-O 与 Theta 分别表达什么。
> - 我能说明为什么复杂度关注的是规模趋大时的增长行为。
> - 我能在简单表达式里找出主导项并忽略低阶项。
> - 我能把复杂度记号和前一讲的 operation counting 连接起来。
> - 我能围绕本讲的主轴 “Asymptotic notation 的目标是只保留增长本质” / “Big-O 与 Theta 的语义差别” / “从计数表达式到复杂度结论”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释 asymptotic notation 为什么要忽略常数和低阶项。
> - 我能比较 Big-O 与 Theta 的信息含量。
> - 我能从循环结构看出线性、二次、对数增长。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 11
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Asymptotic notation 的目标是只保留增长本质 / Big-O 与 Theta 的语义差别 / 从计数表达式到复杂度结论
> - Lecture 21 讲了怎么计数，这一讲给计数结果加上正式的增长记号。
> - Big-O 强调上界视角，Theta 强调紧致阶；它们都服务于比较算法增长速度。
> - 这部分是后续搜索、排序、哈希与复杂度分类的统一语言。

## Core ideas
### Asymptotic notation 的目标是只保留增长本质
复杂度记号不是忽略细节的偷懒，而是把注意力集中在输入足够大时真正决定趋势的部分。
- 常数倍和低阶项在规模足够大时影响相对减弱，因此会被省略。
- 分析重点变成：主导项是什么，它增长得有多快。
- 这让你能跨实现、跨机器、跨语言比较算法结构。
- 所以复杂度并不是运行时间本身，而是运行时间如何随输入规模增长。

> [!note] What to internalize
> - One-sentence takeaway: 复杂度记号不是忽略细节的偷懒，而是把注意力集中在输入足够大时真正决定趋势的部分。
> - Review anchor: 常数倍和低阶项在规模足够大时影响相对减弱，因此会被省略。
> - Review anchor: 分析重点变成：主导项是什么，它增长得有多快。

从做题角度看，只要题目在考“Asymptotic notation 的目标是只保留增长本质”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：复杂度记号不是忽略细节的偷懒，而是把注意力集中在输入足够大时真正决定趋势的部分。

### Big-O 与 Theta 的语义差别
Big-O 常被初学者误当成‘复杂度的唯一答案’，但它本质上只是一个上界表达；Theta 更强调紧致匹配。
- 如果一个算法是 `Theta(n)`，它当然也是 `O(n^2)`，但后者信息更松、更不精确。
- 讨论算法时，若能给出 Theta，通常比只给 Big-O 更有信息量。
- Big-O 仍然很有用，因为它能快速表达‘最多增长到多快’。
- 关键不是背定义，而是知道不同记号保留了多少信息。

> [!note] What to internalize
> - One-sentence takeaway: Big-O 常被初学者误当成‘复杂度的唯一答案’，但它本质上只是一个上界表达；Theta 更强调紧致匹配。
> - Review anchor: 如果一个算法是 `Theta(n)`，它当然也是 `O(n^2)`，但后者信息更松、更不精确。
> - Review anchor: 讨论算法时，若能给出 Theta，通常比只给 Big-O 更有信息量。

从做题角度看，只要题目在考“Big-O 与 Theta 的语义差别”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：Big-O 常被初学者误当成‘复杂度的唯一答案’，但它本质上只是一个上界表达；Theta 更强调紧致匹配。

### 从计数表达式到复杂度结论
真正的能力不是记住几类常见阶，而是能从循环结构或递推关系出发，把计数转成复杂度。
- 单层循环常对应线性级别，双重完整嵌套常对应二次级别，但要看边界与范围是否依赖输入规模。
- 如果某段代码每次都把问题规模减半，你就要想到对数级别。
- 分析时应先找主导操作，再看它执行次数的增长规律。
- 这一步把前一讲的直觉正式化了。

> [!note] What to internalize
> - One-sentence takeaway: 真正的能力不是记住几类常见阶，而是能从循环结构或递推关系出发，把计数转成复杂度。
> - Review anchor: 单层循环常对应线性级别，双重完整嵌套常对应二次级别，但要看边界与范围是否依赖输入规模。
> - Review anchor: 如果某段代码每次都把问题规模减半，你就要想到对数级别。

从做题角度看，只要题目在考“从计数表达式到复杂度结论”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：真正的能力不是记住几类常见阶，而是能从循环结构或递推关系出发，把计数转成复杂度。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - Example: timing a program
> - define two functions
> - creates a list [1, 10, 100, ...] to test different input sizes
> - Example: Report timing and ops/sec for two functions
> - print time and ops/sec for constant fcn
> - for N in L_N
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 找出多项式里的主导项
> ```python
> # 3*n**2 + 10*n + 7
> # asymptotically behaves like n**2
> ```
> 当 `n` 很大时，`n**2` 会压过线性项和常数项，因此整体增长阶由它主导。

> [!example] 看到问题规模减半就联想到对数
> ```python
> count = 0
> n = 1024
> while n > 1:
>     n = n // 2
>     count += 1
> print(count)
> ```
> 每轮都把规模砍半，因此迭代次数与 `log n` 同阶。

## Exercise log
> [!warning] No official finger exercise
> - Calendar explicitly marks this lecture as having no official finger exercise.
> - Use the review checklist, the lecture code, and the linked recitation / problem set materials as the primary self-test for this lecture.
> - For this lecture, a good replacement for the missing finger exercise is: hand-trace one representative example from the code, then write a fresh one from memory.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: this lecture does not have a direct calendar milestone attached, so use it as a consolidation lecture rather than a sprint lecture.
> - Recitation connection: Recitation 10 is the best place to turn the lecture ideas into shorter solved exercises.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec22.pdf|Lecture 22 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec22_code.py|Lecture 22 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex22_sol.pdf|Lecture 22 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec22_transcript.pdf|Lecture 22 transcript]]
- Recitation 10: [[MIT 6.100L-recitations/mit6_100l_rec10.zip|Recitation 10 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 11)

## Review checklist
- [ ] 我能解释 asymptotic notation 为什么要忽略常数和低阶项。
- [ ] 我能比较 Big-O 与 Theta 的信息含量。
- [ ] 我能从循环结构看出线性、二次、对数增长。
- [ ] 我能把 operation counting 结果转成复杂度记号。
- [ ] 我能判断一个更松的上界和一个紧致阶之间的差别。
- [ ] 我能围绕“Asymptotic notation 的目标是只保留增长本质”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Big-O 与 Theta 的语义差别”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：把 Big-O 当成‘唯一正确答案’，忽略它只是上界。
- [ ] 我能说出并避免这个高频误区：看复杂度只盯常数和小输入表现，不看增长趋势。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 把 Big-O 当成‘唯一正确答案’，忽略它只是上界。
> - 看复杂度只盯常数和小输入表现，不看增长趋势。
> - 不会从代码结构回推操作次数。

---
aliases:
  - MIT 6.100L Lecture 15
  - 6.100L L15
  - Recursion
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 15
---

# Lecture 15: Recursion

> [!tip] Hint
> - 我能解释 recursion 不只是‘函数调用自己’，而是把问题定义在更小规模的同类问题上。
> - 我能区分 base case 与 recursive case。
> - 我能说明 recursion 的正确性为什么常与 inductive reasoning 对应。
> - 我能比较 recursion 与 iteration 在表达力和执行代价上的差异。
> - 我能围绕本讲的主轴 “Recursion 的真正定义：同类问题的规模缩小” / “递归与归纳证明天然呼应” / “Recursion 与 iteration 的取舍”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释递归定义里 base case 和 recursive case 的作用。
> - 我能用归纳思维说明一个递归程序为什么正确。
> - 我能判断某个问题是否自然适合递归。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 6.1
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Recursion 的真正定义：同类问题的规模缩小 / 递归与归纳证明天然呼应 / Recursion 与 iteration 的取舍
> - 递归是本课程最重要的思维转折之一：程序不再只靠显式循环推进，也可以通过自我缩小的问题定义前进。
> - 理解递归的关键不是背模板，而是学会看出‘同类更小问题’在哪里。
> - 后面的 merge sort、树结构、某些搜索问题都以递归视角更自然。

## Core ideas
### Recursion 的真正定义：同类问题的规模缩小
一个递归定义必须同时给出两个东西：最小可直接解决的 base case，以及如何把大问题化成更小的同类问题。
- 如果没有 base case，递归就会像没有停止条件的循环一样失控。
- 如果 recursive case 没有真正减小规模，程序也不会收敛到 base case。
- 写递归前要先问：‘更小的同类问题’是什么？‘小到什么程度我就能直接回答？’
- 递归的难点不在语法，而在问题重述能力。

> [!note] What to internalize
> - One-sentence takeaway: 一个递归定义必须同时给出两个东西：最小可直接解决的 base case，以及如何把大问题化成更小的同类问题。
> - Review anchor: 如果没有 base case，递归就会像没有停止条件的循环一样失控。
> - Review anchor: 如果 recursive case 没有真正减小规模，程序也不会收敛到 base case。

从做题角度看，只要题目在考“Recursion 的真正定义：同类问题的规模缩小”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：一个递归定义必须同时给出两个东西：最小可直接解决的 base case，以及如何把大问题化成更小的同类问题。

### 递归与归纳证明天然呼应
递归程序的正确性通常可以用与数学归纳法相同的结构来理解：base case 成立，且假设小问题能正确解决，则大问题也能正确解决。
- 这种思维方式让你不必模拟所有调用细节，也能相信递归会得到对的结果。
- 在阅读递归时，先信任递归调用会返回‘小问题的正确答案’，再看当前层怎样用它组成大答案。
- 如果你总想一步不漏地手动追完整棵调用树，往往会被递归绕晕。
- 递归思维强调的是结构，而不是每一步机器细节。

> [!note] What to internalize
> - One-sentence takeaway: 递归程序的正确性通常可以用与数学归纳法相同的结构来理解：base case 成立，且假设小问题能正确解决，则大问题也能正确解决。
> - Review anchor: 这种思维方式让你不必模拟所有调用细节，也能相信递归会得到对的结果。
> - Review anchor: 在阅读递归时，先信任递归调用会返回‘小问题的正确答案’，再看当前层怎样用它组成大答案。

从做题角度看，只要题目在考“递归与归纳证明天然呼应”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：递归程序的正确性通常可以用与数学归纳法相同的结构来理解：base case 成立，且假设小问题能正确解决，则大问题也能正确解决。

### Recursion 与 iteration 的取舍
有些问题用循环更直接，有些问题用递归更接近问题定义；正确的选择来自数据结构与思维结构的匹配。
- 当问题天然按层次、分治或嵌套结构组织时，递归常常更清楚。
- 递归表达力强，但会带来函数调用开销，也更依赖栈深度。
- 如果只是简单重复并显式维护状态，循环通常更高效。
- 判断标准不是偏好，而是谁更能把问题结构写明白。

> [!note] What to internalize
> - One-sentence takeaway: 有些问题用循环更直接，有些问题用递归更接近问题定义；正确的选择来自数据结构与思维结构的匹配。
> - Review anchor: 当问题天然按层次、分治或嵌套结构组织时，递归常常更清楚。
> - Review anchor: 递归表达力强，但会带来函数调用开销，也更依赖栈深度。

从做题角度看，只要题目在考“Recursion 与 iteration 的取舍”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：有些问题用循环更直接，有些问题用递归更接近问题定义；正确的选择来自数据结构与思维结构的匹配。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - multiplying a*b using a for loop
> - print(mult(5,4))
> - a*b using a while loop
> - print(mult_iter(5,4))
> - a*b recursive
> - print(mult_recur(5,4))
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 递归 factorial
> ```python
> def factorial(n):
>     if n == 0:
>         return 1
>     return n * factorial(n - 1)
>
> print(factorial(5))
> ```
> base case 是 `n == 0`，recursive case 通过 `n-1` 缩小规模。这个例子是递归结构最干净的展示。

> [!example] 递归求字符串长度
> ```python
> def my_len(s):
>     if s == "":
>         return 0
>     return 1 + my_len(s[1:])
>
> print(my_len("python"))
> ```
> 这个例子说明递归不只适用于数字；关键是每轮都把问题变成更短的同类字符串问题。

## Exercise log
> [!note] Finger exercise snapshot
> - Official prompt: Impoement the function that meets the specification beoow.: def recur_power(base, exp): """ base: int or float. exp: int >= 0 Returns base to the power of exp using recursion. Hint: Base case is when exp = 0. Otherwise,...
> - What this is really testing: whether you can compress the lecture into one small, high-frequency coding move without needing the slides beside you.
> - Where to revisit if this feels shaky: go back to the first two Core ideas sections in this note, then rerun the official lecture code once with your own input.
> - Follow-on practice path: after this finger exercise, the most natural next stop is Recitation 08.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: this lecture does not have a direct calendar milestone attached, so use it as a consolidation lecture rather than a sprint lecture.
> - Recitation connection: Recitation 08 is the best place to turn the lecture ideas into shorter solved exercises.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec15.pdf|Lecture 15 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec15_code.py|Lecture 15 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex15_sol.pdf|Lecture 15 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec15_transcript.pdf|Lecture 15 transcript]]
- Recitation 8: [[MIT 6.100L-recitations/mit6_100l_rec08.zip|Recitation 08 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 6.1)

## Review checklist
- [ ] 我能解释递归定义里 base case 和 recursive case 的作用。
- [ ] 我能用归纳思维说明一个递归程序为什么正确。
- [ ] 我能判断某个问题是否自然适合递归。
- [ ] 我能写出一个规模明确递减的递归函数。
- [ ] 我能比较递归与循环的优缺点。
- [ ] 我能围绕“Recursion 的真正定义：同类问题的规模缩小”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“递归与归纳证明天然呼应”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：把‘调用自己’当成递归的全部，没检查问题规模是否真的变小。
- [ ] 我能说出并避免这个高频误区：base case 写得过弱或过晚，导致无限递归。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 把‘调用自己’当成递归的全部，没检查问题规模是否真的变小。
> - base case 写得过弱或过晚，导致无限递归。
> - 看递归时试图死跟所有调用细节，而不是抓结构。

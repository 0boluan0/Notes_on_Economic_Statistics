---
aliases:
  - MIT 6.100L Lecture 23
  - 6.100L L23
  - Complexity Classes Examples
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 23
---

# Lecture 23: Complexity Classes Examples

> [!tip] Hint
> - 我能用具体字符串/列表/搜索例子解释复杂度，而不只会背抽象记号。
> - 我能区分 access、scan、search 等操作在列表上的代价。
> - 我能说明为什么二分搜索需要有序结构。
> - 我能从具体代码里识别哪些操作支配总复杂度。
> - 我能围绕本讲的主轴 “列表与字符串上的基础操作成本” / “Linear search 与 bisection search 的结构差异” / “复杂度分析要回到具体代码路径”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释为什么线性搜索和二分搜索的复杂度不同。
> - 我能说明二分搜索对有序性的依赖。
> - 我能从 list/string 代码里识别主导操作。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 12.1
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: 列表与字符串上的基础操作成本 / Linear search 与 bisection search 的结构差异 / 复杂度分析要回到具体代码路径
> - 这一讲把复杂度记号重新落回具体程序：字符串、列表、搜索例子是理解复杂度最好的训练场。
> - 抽象记号只有放回代码结构里才真的有意义，否则容易流于背诵。
> - 线性搜索与二分搜索的对比，也为下一讲排序算法的分析建立直觉。

## Core ideas
### 列表与字符串上的基础操作成本
想分析程序复杂度，先要知道底层操作本身大概花多少代价，否则整体分析会失去落脚点。
- 按 index 访问 list 某个位置通常比从头扫描查找某个值更便宜，因为它不需要逐个比对。
- 而 membership test、线性搜索这类操作若没有额外结构，通常需要按元素一路看过去。
- 因此同样是‘拿到一个元素’，通过位置拿和通过内容找在复杂度上可能完全不同。
- 理解这点后，你会更清楚为什么数据组织方式会影响算法选择。

> [!note] What to internalize
> - One-sentence takeaway: 想分析程序复杂度，先要知道底层操作本身大概花多少代价，否则整体分析会失去落脚点。
> - Review anchor: 按 index 访问 list 某个位置通常比从头扫描查找某个值更便宜，因为它不需要逐个比对。
> - Review anchor: 而 membership test、线性搜索这类操作若没有额外结构，通常需要按元素一路看过去。

从做题角度看，只要题目在考“列表与字符串上的基础操作成本”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：想分析程序复杂度，先要知道底层操作本身大概花多少代价，否则整体分析会失去落脚点。

### Linear search 与 bisection search 的结构差异
两种搜索方法的差别不只是快慢，而是利用了不同程度的结构信息。
- linear search 几乎不要求前提，只要序列能被逐个看就能用。
- bisection search 要求序列有序，因为它依赖比较结果来直接砍掉一半候选区域。
- 因此复杂度差异来自信息量差异：有序结构让你每次排除的信息更多。
- 当问题不满足有序前提时，强套二分搜索往往得不到正确结果。

> [!note] What to internalize
> - One-sentence takeaway: 两种搜索方法的差别不只是快慢，而是利用了不同程度的结构信息。
> - Review anchor: linear search 几乎不要求前提，只要序列能被逐个看就能用。
> - Review anchor: bisection search 要求序列有序，因为它依赖比较结果来直接砍掉一半候选区域。

从做题角度看，只要题目在考“Linear search 与 bisection search 的结构差异”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：两种搜索方法的差别不只是快慢，而是利用了不同程度的结构信息。

### 复杂度分析要回到具体代码路径
真正分析程序时，不要先贴标签说‘这是 O(n)’，而要先问：循环怎么跑？比较做了几次？哪些操作在重复？
- 字符串与列表例子特别适合练这一点，因为代码结构清楚、输入规模也容易定义。
- 当一个操作在外层循环里反复调用时，其内部复杂度会被整体放大。
- 复杂度不是给函数贴个装饰标签，而是对执行路径的概括。
- 把每个例子都拆回主导操作，会让你的分析更稳。

> [!note] What to internalize
> - One-sentence takeaway: 真正分析程序时，不要先贴标签说‘这是 O(n)’，而要先问：循环怎么跑？比较做了几次？哪些操作在重复？
> - Review anchor: 字符串与列表例子特别适合练这一点，因为代码结构清楚、输入规模也容易定义。
> - Review anchor: 当一个操作在外层循环里反复调用时，其内部复杂度会被整体放大。

从做题角度看，只要题目在考“复杂度分析要回到具体代码路径”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：真正分析程序时，不要先贴标签说‘这是 O(n)’，而要先问：循环怎么跑？比较做了几次？哪些操作在重复？

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - CONSTANT Theta(1)
> - Theta(1)
> - LINEAR Theta(n)
> - Specify what n is in terms of input
> - constant in x: Theta(1)
> - linear in y: Theta(y)
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 线性搜索
> ```python
> def linear_search(L, target):
>     for item in L:
>         if item == target:
>             return True
>     return False
> ```
> 最坏情况下要把整个列表看完，因此主导成本与列表长度线性相关。

> [!example] 二分搜索
> ```python
> def bisection_search(L, target):
>     low, high = 0, len(L) - 1
>     while low <= high:
>         mid = (low + high) // 2
>         if L[mid] == target:
>             return True
>         if L[mid] < target:
>             low = mid + 1
>         else:
>             high = mid - 1
>     return False
> ```
> 每次都丢掉一半候选区间，所以它的搜索轮数是对数级别，但前提是列表必须有序。

## Exercise log
> [!warning] No official finger exercise
> - Calendar explicitly marks this lecture as having no official finger exercise.
> - Use the review checklist, the lecture code, and the linked recitation / problem set materials as the primary self-test for this lecture.
> - For this lecture, a good replacement for the missing finger exercise is: hand-trace one representative example from the code, then write a fresh one from memory.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: calendar shows this lecture touching the following milestones: PS 5 halfway hand-in due。读完本讲后，不应只会解释概念，还应能把它们搬到更长的程序里。
> - Recitation connection: Recitation 10 is the best place to turn the lecture ideas into shorter solved exercises.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec23.pdf|Lecture 23 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec23_code.py|Lecture 23 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex23_sol.pdf|Lecture 23 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec23_transcript.pdf|Lecture 23 transcript]]
- Recitation 10: [[MIT 6.100L-recitations/mit6_100l_rec10.zip|Recitation 10 materials]]
- PS 5 halfway hand-in due: [[MIT 6.100L-problem-sets/mit6_100l_ps5.pdf|PS5 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps5_code.zip|PS5 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 12.1)

## Review checklist
- [ ] 我能解释为什么线性搜索和二分搜索的复杂度不同。
- [ ] 我能说明二分搜索对有序性的依赖。
- [ ] 我能从 list/string 代码里识别主导操作。
- [ ] 我能比较 access、scan、search 的成本。
- [ ] 我能把抽象复杂度记号重新落回具体程序结构。
- [ ] 我能围绕“列表与字符串上的基础操作成本”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Linear search 与 bisection search 的结构差异”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：不看数据前提，机械地把二分搜索套到无序数据上。
- [ ] 我能说出并避免这个高频误区：分析复杂度时只背结论，不回到代码路径。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 不看数据前提，机械地把二分搜索套到无序数据上。
> - 分析复杂度时只背结论，不回到代码路径。
> - 混淆‘按位置访问’和‘按内容查找’的成本差异。

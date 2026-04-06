---
aliases:
  - MIT 6.100L Lecture 24
  - 6.100L L24
  - Sorting Algorithms
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 24
---

# Lecture 24: Sorting Algorithms

> [!tip] Hint
> - 我能比较 bogo、bubble、selection、merge sort 的核心策略差异。
> - 我能解释为什么 merge sort 体现了分治递归的力量。
> - 我能说明同样是排序，‘交换策略’不同会带来怎样的性能差异。
> - 我能把排序算法的正确性和复杂度放在一起理解。
> - 我能围绕本讲的主轴 “不同排序算法在‘如何消除无序’上有不同策略” / “Merge sort：divide, conquer, merge” / “排序算法比较时要同时看正确性与成本”，不翻 slides 也把整节课重新讲一遍。
> - 我能比较 bogo、bubble、selection、merge 的核心策略。
> - 我能解释 merge sort 为什么体现分治思想。
> - 我能从循环或递归结构说明排序正确性的来源。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 12.2
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: 不同排序算法在‘如何消除无序’上有不同策略 / Merge sort：divide, conquer, merge / 排序算法比较时要同时看正确性与成本
> - 排序是算法课的经典主题，因为它能同时训练你比较策略、循环结构、递归分治与复杂度分析。
> - 本讲的重点不是背所有排序名称，而是看清不同算法怎样逐步消除无序性。
> - merge sort 也把前面的 recursion 与复杂度分析连接成了一个完整范例。

## Core ideas
### 不同排序算法在‘如何消除无序’上有不同策略
排序问题看似统一，但每种算法对‘无序’的理解和处理方式都不一样。
- bogo sort 几乎不利用结构，是反例式教学：你会立刻感受到‘随机尝试’有多糟。
- bubble sort 通过相邻比较与交换，把较大元素一轮轮往后推。
- selection sort 每轮从未排序部分挑出最小值放到前面，强调‘选谁放好’。
- 不同策略带来的比较次数、交换次数和实现复杂度都不同。

> [!note] What to internalize
> - One-sentence takeaway: 排序问题看似统一，但每种算法对‘无序’的理解和处理方式都不一样。
> - Review anchor: bogo sort 几乎不利用结构，是反例式教学：你会立刻感受到‘随机尝试’有多糟。
> - Review anchor: bubble sort 通过相邻比较与交换，把较大元素一轮轮往后推。

从做题角度看，只要题目在考“不同排序算法在‘如何消除无序’上有不同策略”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：排序问题看似统一，但每种算法对‘无序’的理解和处理方式都不一样。

### Merge sort：divide, conquer, merge
merge sort 的力量在于：它先把大问题拆成更小的排序问题，再利用‘两个子问题已排序’这一结构高效合并。
- 递归部分负责把列表不断对半拆分，直到子列表足够小直接有序。
- merge 过程负责线性地把两个有序列表合成一个更大的有序列表。
- 这让整体排序避免了许多重复比较，体现了分治的结构优势。
- 它也是递归算法设计与复杂度分析最经典的示例之一。

> [!note] What to internalize
> - One-sentence takeaway: merge sort 的力量在于：它先把大问题拆成更小的排序问题，再利用‘两个子问题已排序’这一结构高效合并。
> - Review anchor: 递归部分负责把列表不断对半拆分，直到子列表足够小直接有序。
> - Review anchor: merge 过程负责线性地把两个有序列表合成一个更大的有序列表。

从做题角度看，只要题目在考“Merge sort：divide, conquer, merge”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：merge sort 的力量在于：它先把大问题拆成更小的排序问题，再利用‘两个子问题已排序’这一结构高效合并。

### 排序算法比较时要同时看正确性与成本
一个排序算法至少要回答两件事：它为什么一定能排对？它在规模增长时代价怎样变化？
- bubble / selection 更容易从循环不变量角度理解正确性。
- merge sort 更容易从递归结构与 merge 正确性证明入手理解。
- 分析排序时，比较次数、交换次数、额外空间都会成为成本指标。
- 所以排序不是单纯比谁更快，而是不同场景下策略取舍。

> [!note] What to internalize
> - One-sentence takeaway: 一个排序算法至少要回答两件事：它为什么一定能排对？它在规模增长时代价怎样变化？
> - Review anchor: bubble / selection 更容易从循环不变量角度理解正确性。
> - Review anchor: merge sort 更容易从递归结构与 merge 正确性证明入手理解。

从做题角度看，只要题目在考“排序算法比较时要同时看正确性与成本”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：一个排序算法至少要回答两件事：它为什么一定能排对？它在规模增长时代价怎样变化？

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - Bogo/Random/Monkey Sort Example
> - print("--- BOGO SORT ---")
> - # L = []
> - # for i in range(0, 9)
> - # L.append(random.randint(0, 100))
> - L = [8, 4, 1, 6, 5, 11, 2, 0]
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] bubble sort 的局部交换思路
> ```python
> def bubble_sort(L):
>     did_swap = True
>     while did_swap:
>         did_swap = False
>         for j in range(1, len(L)):
>             if L[j - 1] > L[j]:
>                 L[j - 1], L[j] = L[j], L[j - 1]
>                 did_swap = True
> ```
> 相邻交换的意义是每轮都把更大的元素往右推，直到整轮不再发生交换时停止。

> [!example] merge sort 的递归骨架
> ```python
> def merge_sort(L):
>     if len(L) < 2:
>         return L[:]
>     mid = len(L) // 2
>     left = merge_sort(L[:mid])
>     right = merge_sort(L[mid:])
>     return merge(left, right)
> ```
> 这个骨架把递归排序与线性合并分开，正是分治策略的核心。

## Exercise log
> [!warning] No official finger exercise
> - Calendar explicitly marks this lecture as having no official finger exercise.
> - Use the review checklist, the lecture code, and the linked recitation / problem set materials as the primary self-test for this lecture.
> - For this lecture, a good replacement for the missing finger exercise is: hand-trace one representative example from the code, then write a fresh one from memory.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: this lecture does not have a direct calendar milestone attached, so use it as a consolidation lecture rather than a sprint lecture.
> - Recitation connection: there is no recitation attached to this lecture week in the official calendar.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec24.pdf|Lecture 24 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec24_code.py|Lecture 24 code (py)]]
- Finger exercise: no official file for this lecture
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec24_transcript.pdf|Lecture 24 transcript]]
- Recitation: none attached to this lecture week
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 12.2)

## Review checklist
- [ ] 我能比较 bogo、bubble、selection、merge 的核心策略。
- [ ] 我能解释 merge sort 为什么体现分治思想。
- [ ] 我能从循环或递归结构说明排序正确性的来源。
- [ ] 我能比较不同排序的复杂度与代价类型。
- [ ] 我能看懂 lecture code 里排序实现的主要控制流。
- [ ] 我能围绕“不同排序算法在‘如何消除无序’上有不同策略”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Merge sort：divide, conquer, merge”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：只背算法名字，不看它真正通过什么步骤减少无序。
- [ ] 我能说出并避免这个高频误区：把 merge sort 当成‘神奇更快’，却不理解 divide + merge 的结构。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 只背算法名字，不看它真正通过什么步骤减少无序。
> - 把 merge sort 当成‘神奇更快’，却不理解 divide + merge 的结构。
> - 比较排序算法时只说快慢，不说前提和代价组成。

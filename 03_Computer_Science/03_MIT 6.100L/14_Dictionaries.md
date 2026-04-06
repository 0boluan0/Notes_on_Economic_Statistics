---
aliases:
  - MIT 6.100L Lecture 14
  - 6.100L L14
  - Dictionaries
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 14
---

# Lecture 14: Dictionaries

> [!tip] Hint
> - 我能解释 dict 为什么是 key -> value 的映射，而不是顺序容器。
> - 我能说出哪些对象适合做 key，以及为什么 mutability 会影响 key 的可用性。
> - 我能用 dict 做计数、查表、分组等常见模式。
> - 我能区分访问不存在 key 的风险与相应防护方式。
> - 我能围绕本讲的主轴 “Dictionary 的核心是 mapping，不是 sequence” / “常见模式：计数、查表、累积” / “访问与更新都要考虑不存在 key 的情况”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释 dict 与 list 的核心差别。
> - 我能说出为什么 key 通常应当是不可变对象。
> - 我能写一个基于 dict 的频率统计程序。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 5.7
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Dictionary 的核心是 mapping，不是 sequence / 常见模式：计数、查表、累积 / 访问与更新都要考虑不存在 key 的情况
> - dict 是 Python 中最重要的非顺序数据结构之一，它把‘通过位置访问’变成‘通过名字或键访问’。
> - 从这一讲开始，程序不再只依赖顺序扫描；你可以直接通过 key 找值，这会改变算法设计方式。
> - 后面 hashing、复杂度、classes 中的 attribute 查找，都和这里的映射直觉密切相关。

## Core ideas
### Dictionary 的核心是 mapping，不是 sequence
list 强调顺序与位置，dict 强调‘给定 key，迅速找到对应 value’。这是完全不同的组织信息方式。
- key 必须是可哈希、稳定的对象；通常会用字符串、数字、tuple 等不可变值。
- value 可以是任意对象，因此 dict 很适合把标签与复杂数据结构绑定起来。
- 当问题天然是‘名称 -> 信息’、‘对象 -> 属性’、‘类别 -> 统计量’时，dict 往往比 list 更自然。
- 思考 dict 时，先问‘我真正想通过什么东西查找答案’。

> [!note] What to internalize
> - One-sentence takeaway: list 强调顺序与位置，dict 强调‘给定 key，迅速找到对应 value’。这是完全不同的组织信息方式。
> - Review anchor: key 必须是可哈希、稳定的对象；通常会用字符串、数字、tuple 等不可变值。
> - Review anchor: value 可以是任意对象，因此 dict 很适合把标签与复杂数据结构绑定起来。

从做题角度看，只要题目在考“Dictionary 的核心是 mapping，不是 sequence”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：list 强调顺序与位置，dict 强调‘给定 key，迅速找到对应 value’。这是完全不同的组织信息方式。

### 常见模式：计数、查表、累积
dict 的高频用法不是背 API，而是识别模式：如果某个累积结果是按类别分别维护的，dict 往往就是答案。
- 频率统计可以把元素当 key，把出现次数当 value。
- 查表问题可以把已有规则预先编码进 dict，再用 key 直接查询。
- 多类别累积时，dict 可以避免大量 if/elif/else 分支，把逻辑统一到同一结构里。
- 这类写法不仅更短，也更容易扩展到新类别。

> [!note] What to internalize
> - One-sentence takeaway: dict 的高频用法不是背 API，而是识别模式：如果某个累积结果是按类别分别维护的，dict 往往就是答案。
> - Review anchor: 频率统计可以把元素当 key，把出现次数当 value。
> - Review anchor: 查表问题可以把已有规则预先编码进 dict，再用 key 直接查询。

从做题角度看，只要题目在考“常见模式：计数、查表、累积”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：dict 的高频用法不是背 API，而是识别模式：如果某个累积结果是按类别分别维护的，dict 往往就是答案。

### 访问与更新都要考虑不存在 key 的情况
dict 的灵活来自于 key 的开放性，但这也意味着‘某个 key 不存在’是你必须主动处理的正常情况。
- 访问前可以先测试 membership，也可以用 `get` 这类更安全的读取方式。
- 更新累积值时，要先考虑 key 首次出现的初始化逻辑。
- 遍历 dict 时，通常更关心 keys、values 或 items 哪一组信息真正是当前任务需要的。
- 很多 dict bug 不是算法错了，而是默认假设某个 key 一定存在。

> [!note] What to internalize
> - One-sentence takeaway: dict 的灵活来自于 key 的开放性，但这也意味着‘某个 key 不存在’是你必须主动处理的正常情况。
> - Review anchor: 访问前可以先测试 membership，也可以用 `get` 这类更安全的读取方式。
> - Review anchor: 更新累积值时，要先考虑 key 首次出现的初始化逻辑。

从做题角度看，只要题目在考“访问与更新都要考虑不存在 key 的情况”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：dict 的灵活来自于 key 的开放性，但这也意味着‘某个 key 不存在’是你必须主动处理的正常情况。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - Example: getting grades using lists, do NOT do it this way...
> - can add more lists for course number, etc.
> - print(get_grade_list('John', names, grade))
> - Example: getting grades using list of lists, do NOT do it this way...
> - print(get_grades('eric', 'mq', grades))
> - print(get_grades('ana', 'ps', grades))
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 用 dict 做词频统计
> ```python
> counts = {}
> for ch in "abca":
>     counts[ch] = counts.get(ch, 0) + 1
> print(counts)
> ```
> 这段代码展示了 dict 最常见的思维：key 表示类别，value 表示该类别当前累计状态。

> [!example] 把映射当查表工具
> ```python
> grade_points = {"A": 4.0, "B": 3.0, "C": 2.0}
> print(grade_points["A"])
> ```
> 当问题是‘给定名称查值’时，dict 比列表搜索更直接，也更贴近问题表达。

## Exercise log
> [!warning] No official finger exercise
> - Calendar explicitly marks this lecture as having no official finger exercise.
> - Use the review checklist, the lecture code, and the linked recitation / problem set materials as the primary self-test for this lecture.
> - For this lecture, a good replacement for the missing finger exercise is: hand-trace one representative example from the code, then write a fresh one from memory.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: calendar shows this lecture touching the following milestones: PS 3 halfway hand-in due。读完本讲后，不应只会解释概念，还应能把它们搬到更长的程序里。
> - Recitation connection: Recitation 07 is the best place to turn the lecture ideas into shorter solved exercises.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec14.pdf|Lecture 14 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec14_code.py|Lecture 14 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex14_sol.pdf|Lecture 14 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec14_transcript.pdf|Lecture 14 transcript]]
- Recitation 7: [[MIT 6.100L-recitations/mit6_100l_rec07.zip|Recitation 07 materials]]
- PS 3 halfway hand-in due: [[MIT 6.100L-problem-sets/mit6_100l_ps3.pdf|PS3 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps3_code.zip|PS3 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 5.7)

## Review checklist
- [ ] 我能解释 dict 与 list 的核心差别。
- [ ] 我能说出为什么 key 通常应当是不可变对象。
- [ ] 我能写一个基于 dict 的频率统计程序。
- [ ] 我能处理首次见到某个 key 的初始化问题。
- [ ] 我能判断一个问题是否更适合用 dict 表达。
- [ ] 我能围绕“Dictionary 的核心是 mapping，不是 sequence”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“常见模式：计数、查表、累积”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：把 dict 当作有固定位置顺序的容器来思考。
- [ ] 我能说出并避免这个高频误区：没有处理 key 缺失，直接访问导致异常。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 把 dict 当作有固定位置顺序的容器来思考。
> - 没有处理 key 缺失，直接访问导致异常。
> - 明明是 mapping 问题，却硬用 list + if/elif/else 堆逻辑。

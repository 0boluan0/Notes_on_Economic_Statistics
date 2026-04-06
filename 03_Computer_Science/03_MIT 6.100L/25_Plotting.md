---
aliases:
  - MIT 6.100L Lecture 25
  - 6.100L L25
  - Plotting
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 25
---

# Lecture 25: Plotting

> [!tip] Hint
> - 我能解释为什么可视化不是‘画图装饰’，而是分析数据与模型的工具。
> - 我能区分 figure、axes、data series 这些层级概念。
> - 我能用 matplotlib 画出最基本的折线图并加上标签。
> - 我能说明图像设计不当会怎样误导结论。
> - 我能围绕本讲的主轴 “Plotting 是分析的一部分，不是收尾装饰” / “Figure、axes、data series 的基本层级” / “图像设计要服务解释，不要误导读者”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释为什么可视化是分析流程的一部分。
> - 我能区分 figure、axes 与 data series。
> - 我能写出一个带标题和坐标轴标签的基本折线图。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 13
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Plotting 是分析的一部分，不是收尾装饰 / Figure、axes、data series 的基本层级 / 图像设计要服务解释，不要误导读者
> - 这一讲把课程从纯粹算法和数据结构拉到数据观察层：程序不仅能算，也要能把结果呈现出来。
> - 可视化的重点不是 API 数量，而是把数据、坐标轴、标签、趋势解释组织清楚。
> - 后面的模拟与数据例子会更依赖这种‘先把行为画出来再分析’的工作流。

## Core ideas
### Plotting 是分析的一部分，不是收尾装饰
好的图像能让模式、异常点、趋势和比较关系一眼出现；坏图像则会掩盖事实甚至制造错觉。
- 程序生成图像的价值在于可重复：当数据更新、参数变化时，你能快速重画并比较。
- 图像让你把‘数字列表’转成肉眼更容易判断的结构，例如趋势、斜率、波动。
- 因此 plotting 是科学计算和数据分析流程中的核心环节，而不是附加美化。
- 写图像代码时要始终问：这张图想支持什么判断？

> [!note] What to internalize
> - One-sentence takeaway: 好的图像能让模式、异常点、趋势和比较关系一眼出现；坏图像则会掩盖事实甚至制造错觉。
> - Review anchor: 程序生成图像的价值在于可重复：当数据更新、参数变化时，你能快速重画并比较。
> - Review anchor: 图像让你把‘数字列表’转成肉眼更容易判断的结构，例如趋势、斜率、波动。

从做题角度看，只要题目在考“Plotting 是分析的一部分，不是收尾装饰”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：好的图像能让模式、异常点、趋势和比较关系一眼出现；坏图像则会掩盖事实甚至制造错觉。

### Figure、axes、data series 的基本层级
理解 matplotlib 最稳的方法不是背命令，而是先理解它在组织什么对象：画布、坐标轴、以及画在轴上的数据系列。
- figure 是整体画布，axes 是具体坐标区域，plot 则是在某个 axes 上添加数据系列。
- 标签、标题、图例帮助读者知道每条线和每个坐标轴代表什么。
- 同一张图上多条曲线时，清晰命名和图例尤其重要。
- 图像代码越结构化，越容易扩展到更多变量和多子图布局。

> [!note] What to internalize
> - One-sentence takeaway: 理解 matplotlib 最稳的方法不是背命令，而是先理解它在组织什么对象：画布、坐标轴、以及画在轴上的数据系列。
> - Review anchor: figure 是整体画布，axes 是具体坐标区域，plot 则是在某个 axes 上添加数据系列。
> - Review anchor: 标签、标题、图例帮助读者知道每条线和每个坐标轴代表什么。

从做题角度看，只要题目在考“Figure、axes、data series 的基本层级”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：理解 matplotlib 最稳的方法不是背命令，而是先理解它在组织什么对象：画布、坐标轴、以及画在轴上的数据系列。

### 图像设计要服务解释，不要误导读者
可视化不只是把数据扔到屏幕上。坐标范围、刻度、颜色、标签选择都会影响读者理解。
- 缺少坐标轴标签和标题，会让图像失去语义，别人根本不知道图在说什么。
- 不合理的坐标范围可能夸大或压扁变化幅度。
- 如果图像要支持比较，就要保证比较维度一致且标注清楚。
- 写图之前先想清楚你打算让读者看到哪种关系。

> [!note] What to internalize
> - One-sentence takeaway: 可视化不只是把数据扔到屏幕上。坐标范围、刻度、颜色、标签选择都会影响读者理解。
> - Review anchor: 缺少坐标轴标签和标题，会让图像失去语义，别人根本不知道图在说什么。
> - Review anchor: 不合理的坐标范围可能夸大或压扁变化幅度。

从做题角度看，只要题目在考“图像设计要服务解释，不要误导读者”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：可视化不只是把数据扔到屏幕上。坐标范围、刻度、颜色、标签选择都会影响读者理解。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - set line width
> - set font size for titles
> - set font size for labels on axes
> - set size of numbers on x-axis
> - set size of numbers on y-axis
> - set size of ticks on x-axis
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 最小 matplotlib 折线图
> ```python
> import matplotlib.pyplot as plt
>
> x = [1, 2, 3, 4]
> y = [1, 4, 2, 3]
> plt.plot(x, y)
> plt.xlabel("x")
> plt.ylabel("y")
> plt.title("Simple line plot")
> plt.show()
> ```
> 这段代码已经包含可视化最基本的语义元素：数据、轴标签、标题。

> [!example] 比较两条数据系列
> ```python
> import matplotlib.pyplot as plt
>
> x = [1, 2, 3]
> plt.plot(x, [1, 2, 3], label="A")
> plt.plot(x, [1, 4, 9], label="B")
> plt.legend()
> plt.show()
> ```
> 一旦图里出现多条线，图例就从可选项变成必要信息。

## Exercise log
> [!warning] No official finger exercise
> - Calendar explicitly marks this lecture as having no official finger exercise.
> - Use the review checklist, the lecture code, and the linked recitation / problem set materials as the primary self-test for this lecture.
> - For this lecture, a good replacement for the missing finger exercise is: hand-trace one representative example from the code, then write a fresh one from memory.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: calendar shows this lecture touching the following milestones: PS 5 due。读完本讲后，不应只会解释概念，还应能把它们搬到更长的程序里。
> - Recitation connection: there is no recitation attached to this lecture week in the official calendar.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec25.pdf|Lecture 25 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec25_code.zip|Lecture 25 code (zip)]]
- Finger exercise: no official file for this lecture
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec25_transcript.pdf|Lecture 25 transcript]]
- Recitation: none attached to this lecture week
- PS 5 due: [[MIT 6.100L-problem-sets/mit6_100l_ps5.pdf|PS5 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps5_code.zip|PS5 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 13)

## Review checklist
- [ ] 我能解释为什么可视化是分析流程的一部分。
- [ ] 我能区分 figure、axes 与 data series。
- [ ] 我能写出一个带标题和坐标轴标签的基本折线图。
- [ ] 我能判断一张图是否缺少关键信息。
- [ ] 我能说明图像设计如何影响结论。
- [ ] 我能围绕“Plotting 是分析的一部分，不是收尾装饰”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Figure、axes、data series 的基本层级”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：把 plotting 当成附加装饰，不思考图像要支持什么分析。
- [ ] 我能说出并避免这个高频误区：画图却不写标题、轴标签、图例。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 把 plotting 当成附加装饰，不思考图像要支持什么分析。
> - 画图却不写标题、轴标签、图例。
> - 坐标设置不合理，误导对趋势和差异的判断。

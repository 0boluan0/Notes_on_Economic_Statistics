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
> - 这节课开头很轻松，但不是可有可无；它是在教你如何把数据真正画出来。
> - 第一件事是 `import matplotlib.pyplot as plt`，`plt` 只是一个更短的别名。
> - 老师先从最简单的 line plot 开始，然后立刻提醒“plot 会连线，scatter 不会”。
> - 这节课很强调视觉表达的基本元素：title、xlabel、ylabel、xticks、grid、legend。
> - 多条曲线放在同一图里时，label 和 line style 变得很重要。
> - subplots 不是新图库，而是同一张 figure 里安排多个坐标区。
> - semilogy 的例子在说明：换坐标尺度，有时比换数据更能看清结构。
> - US population、country population、temperature 数据集都是为了把 plotting 从 toy example 推到真实文件数据。
> - Benford's law 直方图例子在提醒你，不同任务适合不同图形：line plot 不一定总是合适。
> - 听完这节课，你应该能自己读一个简单文件、整理出 x/y 数据并画出有标签的图。

## Lecture flow

### 1. 先说明为什么课程最后还要讲 plotting
Lecture 25 开场老师先解释：  
虽然课程交付项基本结束了，但 plotting 仍然值得学。

原因很现实：

- 以后做 UROP、实验、数据分析时，几乎总会需要可视化
- 这是“写程序处理数据”之后自然的下一步

所以这节课不是额外彩蛋，而是把课程前面学过的数据处理能力接到最基本的 visualization 上。

### 2. 导入库：`matplotlib.pyplot as plt`
第一步是：

```python
import matplotlib.pyplot as plt
```

老师特地解释了 `as plt`：

- 不是强制
- 只是给长模块名取一个简短别名

以后所有调用都可以写成：

- `plt.plot(...)`
- `plt.scatter(...)`
- `plt.title(...)`

### 3. 先设置 `rcParams`：图的风格也能统一控制
课程代码开头用了很多 `plt.rcParams[...] = ...`。

这一步的目的不是让你背每个参数，而是让你知道：

- 图形样式可以全局配置
- 比如线宽、标题字号、轴标签字号、figure 大小等

这在现实里很有用，因为做图不只是“画出来”，还要可读。

### 4. 从最简单的单线图开始
老师第一批例子是一些简单数列：

- linear
- quadratic
- cubic
- exponential

然后用：

```python
plt.plot(nVals, linear)
```

画出一条线。

这一步的目标非常朴素：

- 先接受 plotting 的最小接口就是 x-values + y-values

### 5. `plot` vs `scatter`：是否连线非常关键
老师接着拿一组打乱顺序的点专门比较：

- `plt.plot(...)`
- `plt.scatter(...)`

其中最重要的课堂提醒是：

- `plot` 会按给定顺序把点连起来
- `scatter` 只画点，不连线

这说明：

- 数据点的顺序本身会影响 line plot 的视觉含义

如果顺序没有物理意义，用 `plot` 可能会误导。

### 6. 多条曲线放一张图：开始需要 label 和区分样式
老师随后把多条函数曲线放到同一张图上。

一旦这样做，图就不再只是“能看”，而必须回答：

- 哪条线代表什么

于是就自然引出：

- `label=...`
- `plt.legend(...)`
- 不同颜色、线型、marker

### 7. 基本图形元信息：title、axes labels、ticks、grid
中段老师系统地加上图的说明性元素：

- `plt.title(...)`
- `plt.xlabel(...)`
- `plt.ylabel(...)`
- `plt.xlim(...)`
- `plt.xticks(...)`
- `plt.grid()`

这是课堂里非常重要的一部分，因为：

- 没有标题和坐标标签，图几乎很难复用
- ticks 决定读者怎么理解横纵轴

比如月份数据如果保留 `1,2,3...12`，可读性远不如直接写成 `Jan ... Dec`。

### 8. line style / marker / width：同一数据可以有不同表达风格
老师接着演示了几种写法：

- `'b-'`
- `'r--'`
- `'*g-.'`

以及关键词方式：

- `color='b'`
- `linestyle='--'`
- `linewidth=...`

这部分的重点是让你意识到：

- 图形风格本身是信息编码的一部分

当多条数据同时出现时，风格差异能显著提升可读性。

### 9. subplots：一张 figure 里放多个坐标区
随后老师引入 `plt.subplot(...)`。

用它可以把多张小图排在同一个 figure 中，例如：

- 2 行 1 列
- 2 行 2 列

这样做的好处是：

- 既能比较多个序列
- 又不至于把所有线都挤在同一坐标区里互相干扰

### 10. 从 toy data 切到真实文件：美国人口
课堂后半段开始把 plotting 用到文件数据。

第一个例子是 `lec25_USPopulation.txt`：

- 读取年份
- 读取人口
- 画成 line plot

然后老师又演示：

- `plt.semilogy()`

让 y 轴变成对数尺度。

这一步很关键，因为它说明可视化不只是换数据，也可以换刻度来让趋势更清楚。

### 11. Country population 与 Benford's law：图类型要匹配任务
接下来老师读入很多国家人口数据。

然后做两件事：

- 直接按国家排名画人口规模
- 提取首位数字并画 histogram

这里真正值得记的是：

- line plot 适合展示顺序趋势
- histogram 更适合展示分布

Benford's law 例子就是在训练“选择合适图形类型”。

### 12. 温度数据：从单城市到多城市，再到更复杂对比
课堂最后一大块是 `temperatures.csv`。

老师展示了：

- 读取某城市温度
- 多个城市同图比较
- 年平均温度
- 某年或某区间的温度数据

这一段虽然代码更多，但主线很清楚：

- 先把文件读成结构化数据
- 再挑出想比较的维度
- 最后决定用 line / subplot / 其他方式画出来

### 13. 这节课真正教的是“把结果表达出来”
Lecture 25 并不追求高级可视化，而是在补最关键的一环：

- 你前面已经会计算、会整理数据
- 现在要学会把结果变成一张别人能读懂的图

这在实际工作里常常和算法本身一样重要。

## Exercise log

> [!warning] No official finger exercise
> 这讲官方没有单独的 finger exercise 文件。

最像课堂练习的自测是自己完成这三步：

- 画一条单线图并补齐 title、xlabel、ylabel。
- 把两组城市月均温画到同一图里，并加 legend。
- 把一个真实文件读成两列数据，再画出 line plot 或 histogram。

这三步正好覆盖了本讲最核心的能力：

- 基础绘图
- 多序列对比
- 文件数据可视化

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec25.pdf|Lecture 25 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec25_code.zip|Lecture 25 code (zip)]]
- Finger exercise: no official file for this lecture
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec25_transcript.pdf|Lecture 25 transcript]]
- Recitation: none attached to this lecture week
- PS 5 due: [[MIT 6.100L-problem-sets/mit6_100l_ps5.pdf|PS5 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps5_code.zip|PS5 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 13)

## Review checklist
- [ ] 我能解释为什么 plotting 是数据处理之后自然的一步。
- [ ] 我能写出 `import matplotlib.pyplot as plt` 并说明 `plt` 的角色。
- [ ] 我能区分 `plot` 和 `scatter`。
- [ ] 我能给图补齐 title、axes labels、ticks、grid、legend。
- [ ] 我能为多条曲线设置不同的颜色、线型或 marker。
- [ ] 我能用 subplot 在一个 figure 中放多张小图。
- [ ] 我能说明 semilog 坐标有什么用。
- [ ] 我能把文件数据读成 x/y 再画图。
- [ ] 我能判断 line plot 和 histogram 哪个更适合某类任务。
- [ ] 我能按课堂顺序复述：basic plot -> plot vs scatter -> labeling -> styling -> subplots -> real datasets。

> [!warning] Common mistakes
> - 把点的顺序不当回事，却用 `plot` 连线误导自己。
> - 图里没有标题和坐标标签，导致别人根本不知道在看什么。
> - 所有序列都画成同一种样式，图一复杂就读不出来。
> - 不根据任务选图形类型，什么都先画 line plot。

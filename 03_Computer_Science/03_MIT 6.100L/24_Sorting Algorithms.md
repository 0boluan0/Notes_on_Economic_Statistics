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
> - 这节课不是凭空讲排序，而是先从上节的搜索问题继续问：既然 binary search 快，值不值得先排序？
> - 老师先论证“sort once, search many times”才是排序真正有意义的场景。
> - 课堂先故意讲一个极差算法 bogo sort，是为了给后面正常排序方法立对比。
> - bubble sort 的核心是“每一轮把最大元素往右边冒”。
> - selection sort 的核心是“每一轮选出剩余部分最小元素放到前面”。
> - merge sort 的核心不是交换，而是 divide recursively + merge sorted halves。
> - merge 这一步为什么高效，靠的是两个子列表本身已经有序。
> - 课堂里最重要的复杂度结论不是背答案，而是看算法为什么会产生 `n^2` 或 `n log n`。
> - merge sort 之所以比 bubble/selection 更强，不是魔法，而是因为它把“大排序”拆成“排序子问题 + 线性合并”。
> - 听完这节课，你应该能解释三种排序的工作机制，而不是只记它们的名字。

## Lecture flow

### 1. 先从搜索问题继续推进：为什么需要排序
Lecture 24 开头先回顾上节的搜索：

- linear search：`Theta(n)`
- binary search：`Theta(log n)`，但前提是列表有序

于是老师提出新的问题：

- 如果 binary search 这么快，那我是不是应该总是先排序，再二分搜索？

这一步把排序自然地引了进来。

### 2. “先排序再二分”为什么不总是划算
老师很快指出，如果：

- 你只搜索一次

那么：

- 排序成本 + 二分成本

未必比直接 linear search 划算。

因为排序本身至少得看一遍所有元素，所以不可能便宜到忽略不计。

所以排序真正有价值的场景是：

- sort once
- search many times

这时前面的排序成本才可能被后续很多次更快的搜索摊薄。

### 3. 先故意看一个很烂的排序：bogo sort
在进入正常排序前，老师先拿了一个刻意离谱的例子：bogo sort。

它的思路是：

- 不断随机打乱
- 每次检查是不是已经排好序
- 直到好运气碰上为止

这当然不是实用算法，但它的教学价值很高：

- 它让你先看到“排序”最原始的目标是什么
- 也让后面真正的算法显得更有结构

### 4. `is_sorted(L)`：排序算法通常都围绕“局部顺序”展开
无论是 bogo sort 里的检查，还是后面 bubble / selection / merge 的思路，  
老师都在不断回到一个最基础的问题：

- 怎样判断列表已经有序

这也提醒你，排序算法并不是和“比较相邻元素”毫无关系的神秘过程，而是不断利用局部有序去逼近整体有序。

### 5. bubble sort：把大元素一轮轮冒到右边
接下来课堂正式进入第一个正常算法：bubble sort。

老师的讲解方式很形象：

- 从左到右比较相邻元素
- 如果前一个比后一个大，就交换
- 这样一轮下来，最大元素会被“冒泡”到最右边

然后对剩余未排序部分重复。

所以 bubble sort 的主线不是“我在排序整张表”，而是：

- 每一轮固定一个最大的元素

### 6. bubble sort 为什么会是平方级
从复杂度上看，bubble sort 的直觉来源也很直接：

- 一轮要扫过接近整个列表
- 而这样的轮数又要接近列表长度

于是总工作量大致是：

- `n * n`

也就是 `Theta(n^2)` 级别。

课堂里老师还会强调一些局部优化，比如如果一轮没有 swap 就提前停，但整体阶数不会因此改变。

### 7. selection sort：每一轮选最小值放到前面
第二个正常算法是 selection sort。

它和 bubble sort 的区别是：

- bubble 是通过一连串相邻交换把最大值推到后面
- selection 是在剩余部分里扫描出最小值，再一次性放到前面

所以 selection sort 的思维更像：

1. 先把第 0 个位置该放什么找出来
2. 再把第 1 个位置该放什么找出来
3. 依此类推

### 8. selection sort 和 bubble sort 的差别
老师把这两个算法放在一起讲，是为了让你看到：

- 它们都属于平方级排序
- 但“局部动作”不一样

bubble 强调相邻交换和元素上浮；  
selection 强调在剩余区间里选极值。

这很适合理解算法设计中的一个事实：

- 相同复杂度类内部，算法工作方式仍然可以差很多

### 9. selection sort 的变体：减少不必要交换
老师还给了一个 selection sort variation：

- 先记住当前最小值位置
- 扫完一轮后再交换一次

这说明即使同一个大算法思路里，也可以继续优化常数开销。  
但大方向仍然不变：

- 还是每轮扫描剩余部分
- 还是总计平方级

### 10. merge sort 登场：开始换思路，不再局部交换
讲完两个平方级算法后，老师转向 merge sort。

这里课堂明显在换思路：

- 不再一遍遍在原列表上做局部比较交换
- 而是先把问题拆小

merge sort 的骨架是：

1. divide：把列表分成两半
2. recursively sort 两半
3. merge：把两个已排序子列表线性合并

这一步非常重要，因为它把排序问题重新写成了递归问题。

### 11. merge step：为什么合并两个有序列表这么聪明
老师在讲 merge sort 时，用了大量时间单讲 merge。

假设：

- left 已排序
- right 已排序

那么合并时你只需要不断比较：

- left 当前最小未取元素
- right 当前最小未取元素

每次拿更小的那个放入结果。

一旦某边用尽，直接把另一边剩余元素接上即可。

> [!note]
> merge 之所以能线性完成，关键前提是两个输入子列表已经各自有序。

### 12. merge sort 的递归骨架
在代码里，merge sort 的典型形式是：

```python
if len(L) < 2:
    return L[:]
else:
    middle = len(L) // 2
    left = merge_sort(L[:middle])
    right = merge_sort(L[middle:])
    return merge(left, right)
```

这里很清楚地体现出：

- base case：长度 0 或 1 的列表天然已排序
- recursive step：排序左右两半
- combine step：merge

这就是典型的 divide and conquer。

### 13. merge sort 为什么会到 `n log n`
老师后面开始解释 merge sort 的复杂度直觉。

可以从两层看：

- 拆分层数大约是 `log n`
- 每一层所有 merge 工作加起来大约是 `n`

于是整体复杂度就是：

- `Theta(n log n)`

这和前面平方级排序的差距会在大输入上越来越明显。

### 14. 这节课真正让你比较的是“算法设计范式”
Lecture 24 的价值不只是背：

- bubble
- selection
- merge

而是让你看到三种很不同的设计范式：

- 笨拙随机尝试：bogo sort
- 局部交换 / 局部选择：bubble / selection
- divide and conquer：merge sort

从这节课开始，你应该能把排序算法看成算法思想的展示窗口，而不是孤立 API。

## Exercise log

> [!warning] No official finger exercise
> 这讲官方没有单独的 finger exercise 文件。

最适合的课堂后自测是：

- 不看代码，手动模拟一轮 bubble sort 和一轮 selection sort。
- 自己把 `[8, 4, 1, 6, 5, 11, 2, 0]` 画出 merge sort 的拆分树和合并顺序。

这两步正好对应本讲两种最核心的排序思路：

- 原地局部交换
- 递归拆分再合并

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec24.pdf|Lecture 24 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec24_code.py|Lecture 24 code (py)]]
- Finger exercise: no official file for this lecture
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec24_transcript.pdf|Lecture 24 transcript]]
- Recitation: none attached to this lecture week
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 12.2)

## Review checklist
- [ ] 我能解释为什么排序问题是从“binary search 需要有序列表”自然引出来的。
- [ ] 我能说明什么时候“先排序再二分”才值得。
- [ ] 我能复述 bogo sort 的思路以及它为什么只是反面教材。
- [ ] 我能手动模拟 bubble sort 的一轮。
- [ ] 我能手动模拟 selection sort 的一轮。
- [ ] 我能比较 bubble sort 和 selection sort 在工作方式上的差异。
- [ ] 我能解释 merge step 为什么是线性的。
- [ ] 我能画出 merge sort 的递归拆分与回合并过程。
- [ ] 我能说明为什么 merge sort 是 `Theta(n log n)` 而前两者是 `Theta(n^2)`。
- [ ] 我能按课堂顺序复述：search motivation -> bad sort -> bubble -> selection -> merge sort。

> [!warning] Common mistakes
> - 只背排序名字，不理解每轮在“固定”什么信息。
> - 把 merge sort 看成只是“又一个排序”，没看到 divide-and-conquer 的结构。
> - 看到 selection sort 交换次数更少，就误以为它复杂度阶数也更低。
> - 忘记 merge step 之所以快，前提是两边已经排序好。

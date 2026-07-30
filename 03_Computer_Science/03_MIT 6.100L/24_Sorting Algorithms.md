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
> <!-- bilingual-en:start -->
> - Sorting is not introduced in isolation: the lecture builds on the previous search problem by asking whether fast binary search justifies sorting first.
> - The instructor shows that sorting is most useful in a “sort once, search many times” setting.
> - The lecture deliberately begins with the terrible bogo sort algorithm to contrast it with the practical methods that follow.
> - Bubble sort repeatedly moves the largest unsorted element toward the right.
> - Selection sort repeatedly chooses the smallest remaining element and places it at the front of the unsorted region.
> - Merge sort is based not on swaps, but on recursively dividing a list and merging sorted halves.
> - The merge step is efficient because both input sublists are already sorted.
> - The central complexity lesson is not to memorize results, but to understand why an algorithm produces `n^2` or `n log n` work.
> - Merge sort outperforms bubble sort and selection sort because it turns one large sorting problem into smaller sorting problems followed by linear-time merges.
> - After this lecture, you should be able to explain how all three sorting algorithms work, not merely recall their names.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 先从搜索问题继续推进：为什么需要排序
<!-- bilingual-en:start -->
*1. Building on the previous lecture's search problem: why sort?*
<!-- bilingual-en:end -->
Lecture 24 开头先回顾上节的搜索：
<!-- bilingual-en:start -->
Lecture 24 begins by reviewing search from the previous lecture:
<!-- bilingual-en:end -->

- linear search：`Theta(n)`
- binary search：`Theta(log n)`，但前提是列表有序
<!-- bilingual-en:start -->
- Linear search: `Theta(n)`
- Binary search: `Theta(log n)`, but only if the list is sorted
<!-- bilingual-en:end -->

于是老师提出新的问题：
<!-- bilingual-en:start -->
The instructor then poses a new question:
<!-- bilingual-en:end -->

- 如果 binary search 这么快，那我是不是应该总是先排序，再二分搜索？
<!-- bilingual-en:start -->
- If binary search is so fast, shouldn't I always sort first and then binary search?
<!-- bilingual-en:end -->

这一步把排序自然地引了进来。
<!-- bilingual-en:start -->
This naturally introduces sorting.
<!-- bilingual-en:end -->

### 2. “先排序再二分”为什么不总是划算
<!-- bilingual-en:start -->
*2. Why sorting before binary search is not always worthwhile*
<!-- bilingual-en:end -->
老师很快指出，如果：
<!-- bilingual-en:start -->
The instructor quickly points out that if:
<!-- bilingual-en:end -->

- 你只搜索一次
<!-- bilingual-en:start -->
- You only search once
<!-- bilingual-en:end -->

那么：
<!-- bilingual-en:start -->
Then:
<!-- bilingual-en:end -->

- 排序成本 + 二分成本
<!-- bilingual-en:start -->
- Sorting cost + binary search cost
<!-- bilingual-en:end -->

未必比直接 linear search 划算。
<!-- bilingual-en:start -->
The combined cost may not be lower than the cost of a direct linear search.
<!-- bilingual-en:end -->

因为排序本身至少得看一遍所有元素，所以不可能便宜到忽略不计。
<!-- bilingual-en:start -->
Sorting itself must inspect every element at least once, so its cost can never be ignored.
<!-- bilingual-en:end -->

所以排序真正有价值的场景是：
<!-- bilingual-en:start -->
Therefore, the truly valuable scenario for sorting is when:
<!-- bilingual-en:end -->

- sort once
- search many times

这时前面的排序成本才可能被后续很多次更快的搜索摊薄。
<!-- bilingual-en:start -->
The initial sorting cost can be amortized by multiple faster searches afterward.
<!-- bilingual-en:end -->

### 3. 先故意看一个很烂的排序：bogo sort
<!-- bilingual-en:start -->
*3. First, a deliberately terrible sort: bogo sort*
<!-- bilingual-en:end -->
在进入正常排序前，老师先拿了一个刻意离谱的例子：bogo sort。
<!-- bilingual-en:start -->
Before introducing practical sorting algorithms, the instructor starts with an intentionally absurd example: bogo sort.
<!-- bilingual-en:end -->

它的思路是：
<!-- bilingual-en:start -->
Its approach is:
<!-- bilingual-en:end -->

- 不断随机打乱
- 每次检查是不是已经排好序
- 直到好运气碰上为止
<!-- bilingual-en:start -->
- Keep randomly shuffling
- Check if it's sorted after each shuffle
- Repeat until luck strikes
<!-- bilingual-en:end -->

这当然不是实用算法，但它的教学价值很高：
<!-- bilingual-en:start -->
This is not a practical algorithm, but it has considerable teaching value:
<!-- bilingual-en:end -->

- 它让你先看到“排序”最原始的目标是什么
- 也让后面真正的算法显得更有结构
<!-- bilingual-en:start -->
- It shows you the most basic goal of sorting
- It also makes the structure of the practical algorithms that follow easier to see
<!-- bilingual-en:end -->

### 4. `is_sorted(L)`：排序算法通常都围绕“局部顺序”展开
<!-- bilingual-en:start -->
*4. `is_sorted(L)`: sorting algorithms often build global order from local order*
<!-- bilingual-en:end -->
无论是 bogo sort 里的检查，还是后面 bubble / selection / merge 的思路，  
老师都在不断回到一个最基础的问题：
<!-- bilingual-en:start -->
Whether checking bogo sort or explaining bubble, selection, and merge sort, the instructor repeatedly returns to one basic question:
<!-- bilingual-en:end -->

- 怎样判断列表已经有序
<!-- bilingual-en:start -->
- How can you determine whether a list is sorted?
<!-- bilingual-en:end -->

这也提醒你，排序算法并不是和“比较相邻元素”毫无关系的神秘过程，而是不断利用局部有序去逼近整体有序。
<!-- bilingual-en:start -->
Sorting is not a mysterious process detached from local comparisons; these algorithms repeatedly use local order to build toward global order.
<!-- bilingual-en:end -->

### 5. bubble sort：把大元素一轮轮冒到右边
<!-- bilingual-en:start -->
*5. Bubble sort: moving large elements to the right one pass at a time*
<!-- bilingual-en:end -->
接下来课堂正式进入第一个正常算法：bubble sort。
<!-- bilingual-en:start -->
The lecture then moves to its first conventional sorting algorithm: bubble sort.
<!-- bilingual-en:end -->

老师的讲解方式很形象：
<!-- bilingual-en:start -->
The instructor explains it visually:
<!-- bilingual-en:end -->

- 从左到右比较相邻元素
- 如果前一个比后一个大，就交换
- 这样一轮下来，最大元素会被“冒泡”到最右边
<!-- bilingual-en:start -->
- Compare adjacent elements from left to right
- If the earlier element is larger than the later one, swap them
- After one round, the largest element will be 'bubbled' to the far right
<!-- bilingual-en:end -->

然后对剩余未排序部分重复。
<!-- bilingual-en:start -->
Then repeat this process with the remaining unsorted portion.
<!-- bilingual-en:end -->

所以 bubble sort 的主线不是“我在排序整张表”，而是：
<!-- bilingual-en:start -->
The useful way to think about bubble sort is not “sort the whole list at once,” but:
<!-- bilingual-en:end -->

- 每一轮固定一个最大的元素
<!-- bilingual-en:start -->
- Each round fixes the largest remaining element in place
<!-- bilingual-en:end -->

### 6. bubble sort 为什么会是平方级
<!-- bilingual-en:start -->
*6. Why bubble sort is `Theta(n^2)`*
<!-- bilingual-en:end -->
从复杂度上看，bubble sort 的直觉来源也很直接：
<!-- bilingual-en:start -->
From a complexity standpoint, the intuition behind bubble sort is also straightforward:
<!-- bilingual-en:end -->

- 一轮要扫过接近整个列表
- 而这样的轮数又要接近列表长度
<!-- bilingual-en:start -->
- Each round requires scanning almost the entire list
- The number of such rounds is roughly equal to the length of the list
<!-- bilingual-en:end -->

于是总工作量大致是：
<!-- bilingual-en:start -->
So the total work is approximately:
<!-- bilingual-en:end -->

- `n * n`

也就是 `Theta(n^2)` 级别。
<!-- bilingual-en:start -->
That is `Theta(n^2)` work.
<!-- bilingual-en:end -->

课堂里老师还会强调一些局部优化，比如如果一轮没有 swap 就提前停，但整体阶数不会因此改变。
<!-- bilingual-en:start -->
The instructor also mentions local optimizations, such as stopping early when a pass makes no swaps, but these do not change the overall asymptotic order.
<!-- bilingual-en:end -->

### 7. selection sort：每一轮选最小值放到前面
<!-- bilingual-en:start -->
*7. Selection sort: placing the minimum element at the front each round*
<!-- bilingual-en:end -->
第二个正常算法是 selection sort。
<!-- bilingual-en:start -->
The second conventional algorithm is selection sort.
<!-- bilingual-en:end -->

它和 bubble sort 的区别是：
<!-- bilingual-en:start -->
The key difference from bubble sort is:
<!-- bilingual-en:end -->

- bubble 是通过一连串相邻交换把最大值推到后面
- selection 是在剩余部分里扫描出最小值，再一次性放到前面
<!-- bilingual-en:start -->
- Bubble sort moves the maximum element to the end through a series of adjacent swaps
- Selection sort scans the remaining portion for its minimum and places it at the front of that portion
<!-- bilingual-en:end -->

所以 selection sort 的思维更像：
<!-- bilingual-en:start -->
Selection sort therefore proceeds position by position:
<!-- bilingual-en:end -->

1. 先把第 0 个位置该放什么找出来
2. 再把第 1 个位置该放什么找出来
3. 依此类推
<!-- bilingual-en:start -->
1. First, determine what should go in position 0
2. Then, determine what should go in position 1
3. And so on
<!-- bilingual-en:end -->

### 8. selection sort 和 bubble sort 的差别
<!-- bilingual-en:start -->
*8. How selection sort differs from bubble sort*
<!-- bilingual-en:end -->
老师把这两个算法放在一起讲，是为了让你看到：
<!-- bilingual-en:start -->
The instructor presents these two algorithms together to show that:
<!-- bilingual-en:end -->

- 它们都属于平方级排序
- 但“局部动作”不一样
<!-- bilingual-en:start -->
- Both are quadratic-time sorting algorithms
- Their elementary operations are different
<!-- bilingual-en:end -->

bubble 强调相邻交换和元素上浮；  
selection 强调在剩余区间里选极值。
<!-- bilingual-en:start -->
Bubble sort relies on adjacent swaps to bubble large elements toward the end; selection sort finds the minimum in the remaining range.
<!-- bilingual-en:end -->

这很适合理解算法设计中的一个事实：
<!-- bilingual-en:start -->
This illustrates an important fact about algorithm design:
<!-- bilingual-en:end -->

- 相同复杂度类内部，算法工作方式仍然可以差很多
<!-- bilingual-en:start -->
- Algorithms within the same complexity class can still operate very differently
<!-- bilingual-en:end -->

### 9. selection sort 的变体：减少不必要交换
<!-- bilingual-en:start -->
*9. A selection sort variation: reducing unnecessary swaps*
<!-- bilingual-en:end -->
老师还给了一个 selection sort variation：
<!-- bilingual-en:start -->
The instructor also introduces a variation of selection sort:
<!-- bilingual-en:end -->

- 先记住当前最小值位置
- 扫完一轮后再交换一次
<!-- bilingual-en:start -->
- First, keep track of the index of the current minimum
- After scanning one round, swap only once
<!-- bilingual-en:end -->

这说明即使同一个大算法思路里，也可以继续优化常数开销。  
但大方向仍然不变：
<!-- bilingual-en:start -->
This shows that the same basic algorithm can reduce constant-factor overhead. Its overall growth rate, however, remains unchanged:
<!-- bilingual-en:end -->

- 还是每轮扫描剩余部分
- 还是总计平方级
<!-- bilingual-en:start -->
- We still scan the remaining portion in each round
- The total complexity is still quadratic
<!-- bilingual-en:end -->

### 10. merge sort 登场：开始换思路，不再局部交换
<!-- bilingual-en:start -->
*10. Merge sort: a change of strategy away from local swaps*
<!-- bilingual-en:end -->
讲完两个平方级算法后，老师转向 merge sort。
<!-- bilingual-en:start -->
After covering two quadratic-time algorithms, the instructor turns to merge sort.
<!-- bilingual-en:end -->

这里课堂明显在换思路：
<!-- bilingual-en:start -->
This marks a clear shift in strategy:
<!-- bilingual-en:end -->

- 不再一遍遍在原列表上做局部比较交换
- 而是先把问题拆小
<!-- bilingual-en:start -->
- Stop making repeated local comparisons and swaps in the original list
- Break the problem into smaller subproblems first
<!-- bilingual-en:end -->

merge sort 的骨架是：
<!-- bilingual-en:start -->
The structure of merge sort is:
<!-- bilingual-en:end -->

1. divide：把列表分成两半
2. recursively sort 两半
3. merge：把两个已排序子列表线性合并
<!-- bilingual-en:start -->
1. Divide: Split the list into two halves
2. Recursively sort each half
3. Merge: Combine the two sorted sublists in linear time
<!-- bilingual-en:end -->

这一步非常重要，因为它把排序问题重新写成了递归问题。
<!-- bilingual-en:start -->
This is crucial because it recasts sorting as a recursive problem.
<!-- bilingual-en:end -->

### 11. merge step：为什么合并两个有序列表这么聪明
<!-- bilingual-en:start -->
*11. The merge step: why two sorted lists can be combined efficiently*
<!-- bilingual-en:end -->
老师在讲 merge sort 时，用了大量时间单讲 merge。
<!-- bilingual-en:start -->
The instructor devotes substantial time to the merge step itself.
<!-- bilingual-en:end -->

假设：
<!-- bilingual-en:start -->
Suppose that:
<!-- bilingual-en:end -->

- left 已排序
- right 已排序
<!-- bilingual-en:start -->
- Left sublist is already sorted
- Right sublist is already sorted
<!-- bilingual-en:end -->

那么合并时你只需要不断比较：
<!-- bilingual-en:start -->
During the merge, you need only compare:
<!-- bilingual-en:end -->

- left 当前最小未取元素
- right 当前最小未取元素
<!-- bilingual-en:start -->
- The smallest unmerged element in `left`
- The smallest unmerged element in `right`
<!-- bilingual-en:end -->

每次拿更小的那个放入结果。
<!-- bilingual-en:start -->
Take the smaller one each time and place it into the result.
<!-- bilingual-en:end -->

一旦某边用尽，直接把另一边剩余元素接上即可。
<!-- bilingual-en:start -->
Once one side is exhausted, simply append the remaining elements from the other side.
<!-- bilingual-en:end -->

> [!note]
> merge 之所以能线性完成，关键前提是两个输入子列表已经各自有序。
> <!-- bilingual-en:start -->
> The merge step runs in linear time because both input sublists are already sorted.
> <!-- bilingual-en:end -->

### 12. merge sort 的递归骨架
<!-- bilingual-en:start -->
*12. The recursive structure of merge sort*
<!-- bilingual-en:end -->
在代码里，merge sort 的典型形式是：
<!-- bilingual-en:start -->
In code, the typical form of merge sort is:
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
This makes it very clear that:
<!-- bilingual-en:end -->

- base case：长度 0 或 1 的列表天然已排序
- recursive step：排序左右两半
- combine step：merge
<!-- bilingual-en:start -->
- Base case: A list of length 0 or 1 is naturally sorted
- Recursive step: Sort the left and right halves
- Combine step: Merge
<!-- bilingual-en:end -->

这就是典型的 divide and conquer。
<!-- bilingual-en:start -->
This exemplifies the classic divide-and-conquer paradigm.
<!-- bilingual-en:end -->

### 13. merge sort 为什么会到 `n log n`
<!-- bilingual-en:start -->
*13. Why merge sort is `Theta(n log n)`*
<!-- bilingual-en:end -->
老师后面开始解释 merge sort 的复杂度直觉。
<!-- bilingual-en:start -->
The instructor then explains the intuition behind merge sort's complexity.
<!-- bilingual-en:end -->

可以从两层看：
<!-- bilingual-en:start -->
The argument has two levels:
<!-- bilingual-en:end -->

- 拆分层数大约是 `log n`
- 每一层所有 merge 工作加起来大约是 `n`
<!-- bilingual-en:start -->
- The recursion tree has approximately `log n` levels
- At each level, all merge operations together do roughly `n` work
<!-- bilingual-en:end -->

于是整体复杂度就是：
<!-- bilingual-en:start -->
Thus, the overall complexity is:
<!-- bilingual-en:end -->

- `Theta(n log n)`

这和前面平方级排序的差距会在大输入上越来越明显。
<!-- bilingual-en:start -->
The advantage over quadratic-time sorts becomes increasingly visible as the input grows.
<!-- bilingual-en:end -->

### 14. 这节课真正让你比较的是“算法设计范式”
<!-- bilingual-en:start -->
*14. What this lecture really compares: algorithm design paradigms*
<!-- bilingual-en:end -->
Lecture 24 的价值不只是背：
<!-- bilingual-en:start -->
The value of Lecture 24 is not merely in memorizing a list of algorithms:
<!-- bilingual-en:end -->

- bubble
- selection
- merge

而是让你看到三种很不同的设计范式：
<!-- bilingual-en:start -->
Its real value is showing three very different design paradigms:
<!-- bilingual-en:end -->

- 笨拙随机尝试：bogo sort
- 局部交换 / 局部选择：bubble / selection
- divide and conquer：merge sort
<!-- bilingual-en:start -->
- Blind random trial and error: bogo sort
- Local swaps or local selection: bubble sort and selection sort
- Divide and conquer: merge sort
<!-- bilingual-en:end -->

从这节课开始，你应该能把排序算法看成算法思想的展示窗口，而不是孤立 API。
<!-- bilingual-en:start -->
From this lecture onward, you should be able to view sorting algorithms as showcases of algorithmic ideas rather than isolated APIs.
<!-- bilingual-en:end -->

## Exercise log

> [!warning] No official finger exercise
> 这讲官方没有单独的 finger exercise 文件。
> <!-- bilingual-en:start -->
> This lecture does not have a separate finger exercise file.
> <!-- bilingual-en:end -->

最适合的课堂后自测是：
<!-- bilingual-en:start -->
The best post-class self-test is:
<!-- bilingual-en:end -->

- 不看代码，手动模拟一轮 bubble sort 和一轮 selection sort。
- 自己把 `[8, 4, 1, 6, 5, 11, 2, 0]` 画出 merge sort 的拆分树和合并顺序。
<!-- bilingual-en:start -->
- Without looking at the code, manually simulate one round of bubble sort and one round of selection sort.
- Draw out the split tree and merge order for `[8, 4, 1, 6, 5, 11, 2, 0]` using merge sort.
<!-- bilingual-en:end -->

这两步正好对应本讲两种最核心的排序思路：
<!-- bilingual-en:start -->
These two steps correspond directly to the lecture's two central sorting ideas:
<!-- bilingual-en:end -->

- 原地局部交换
- 递归拆分再合并
<!-- bilingual-en:start -->
- In-place local swaps
- Recursive splitting and merging
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
- [ ] I can explain how binary search's need for a sorted list naturally motivates the sorting problem.
- [ ] I can describe when sorting before binary search is worthwhile.
- [ ] I can recount the bogo sort approach and explain why it is only a cautionary example.
- [ ] I can manually simulate one round of bubble sort.
- [ ] I can manually simulate one round of selection sort.
- [ ] I can explain the differences in how bubble sort and selection sort work.
- [ ] I can explain why the merge step is linear time.
- [ ] I can draw out the recursive split and merge process for merge sort.
- [ ] I can explain why merge sort is `Theta(n log n)` while the previous two are `Theta(n^2)`.
- [ ] I can reconstruct the lecture sequence: search motivation -> bad sort -> bubble -> selection -> merge sort.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 只背排序名字，不理解每轮在“固定”什么信息。
> - 把 merge sort 看成只是“又一个排序”，没看到 divide-and-conquer 的结构。
> - 看到 selection sort 交换次数更少，就误以为它复杂度阶数也更低。
> - 忘记 merge step 之所以快，前提是两边已经排序好。
> <!-- bilingual-en:start -->
> - Memorizing algorithm names without understanding what each pass fixes in place.
> - Treating merge sort as just another sorting algorithm and missing its divide-and-conquer structure.
> - Assuming that fewer swaps give selection sort a lower asymptotic complexity.
> - Forgetting that merge is fast only because both halves are already sorted.
> <!-- bilingual-en:end -->

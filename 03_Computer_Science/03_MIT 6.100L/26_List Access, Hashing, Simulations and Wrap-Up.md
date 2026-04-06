---
aliases:
  - MIT 6.100L Lecture 26
  - 6.100L L26
  - List Access, Hashing, Simulations, and Wrap-Up
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 26
---

# Lecture 26: List Access, Hashing, Simulations, and Wrap-Up

> [!tip] Hint
> - 最后一讲先说今天要“收尾若干 loose ends”，所以内容确实跨几个主题，但推进顺序很清楚。
> - 第一部分先回到 list：为什么 index access 是 `Theta(1)`，它在内存里到底怎么放。
> - contiguous memory 和地址偏移量，是理解 list access complexity 的关键。
> - 然后课程转到 dictionary：如果把 dict 朴素存成 list of entries，lookup 为什么会退化成 `Theta(n)`。
> - hashing 的核心不是记 hash 值，而是理解“用 hash function 把 key 映到 bucket/index”。
> - 只有 hashable 对象才能做 dict key，这又把 immutable / mutable 的概念拉回来了。
> - 好 hash function 和合适 hash table 大小决定了碰撞多少，也决定了平均性能。
> - 后半段 simulations 把 randomness、probability 和 computation 接起来，说明你已经能做很多实验型建模。
> - 骰子和 fill_pool 例子都在练同一个框架：定义一次实验，重复很多次，用频率逼近概率或平均值。
> - 听完这节课，你应该能把 list、dict、hashing、simulation 当成课程前面所有内容的一次回收总结。

## Lecture flow

### 1. 开场先说明：今天是收尾和 wrap-up
Lecture 26 一开始就说这会是最后一讲，内容会分成几个 loose ends：

- lists
- dictionaries / hashing
- simulations
- 课程总结

所以这讲表面上看跨度很大，但它其实是在把前面学过的东西重新连回到底层实现和更广的应用场景上。

### 2. 先回到 list：为什么某些操作是 `Theta(1)`，某些是 `Theta(n)`
老师先回顾我们已经知道的 list 操作复杂度：

- equality：`Theta(n)`
- membership：`Theta(n)`
- iteration：`Theta(n)`
- direct index access：`Theta(1)`

然后问了一个非常关键的问题：

- 为什么 index access 会是常数时间？

这就把课堂从“背复杂度表”推进到“理解实现原因”。

### 3. contiguous memory：列表为什么能常数时间取第 i 个元素
老师解释，list 在内存里可以想成一段连续内存块。

如果：

- 列表长度是 `L`
- 每个元素占固定大小（为便于解释，先假设只是整数）

那么一旦知道：

- 列表起始地址
- 每个元素大小
- 索引 `i`

就能直接算出第 `i` 个元素在哪个地址。

所以：

- 不需要一个个数过去
- 只需要做地址偏移运算

这就是 `Theta(1)` 的来源。

### 4. list 里装的往往不是值本身，而是引用
老师随后补充，真实 Python list 里不一定直接放原始整数值。

如果 list 元素是：

- 另一个 list
- 一个 dictionary
- 更复杂的对象

那 list 本身存放的更像是引用 / 指针。

这一步很重要，因为它帮你把前面学过的 aliasing、对象、嵌套结构与底层表示重新连起来。

### 5. 为什么 equality / membership / iteration 是线性
和 index access 对照起来，老师再次强调：

- 判断两个 list 是否相等，要逐元素比
- 判断某元素是否在 list 里，要一个个扫
- 遍历 list 当然也要逐个访问

所以这些操作天然和长度成正比。

这一段实际上是在示范一种更成熟的复杂度理解：

- 不是背表，而是回到“为了完成这个任务，最少得看多少数据”

### 6. 从 list 过渡到 dict：为什么 dict 不能像 list 那样按位置找
接着老师切到 dictionary。

如果我们天真地把 dict 存成一串 entries：

- 每个 entry 是 `[key, value]`
- 所有 entry 排成一个长 list

那么查某个 key 时，就只能：

- 从头扫到尾
- 一个个比 key

这就会退化成 `Theta(n)`。

所以 dict 想快，必须有别的组织方法。

### 7. hashing：把 key 映射到 hash table 的某个位置
这时课程正式引出 hashing。

hashing 的核心思路是：

1. 对 key 运行一个 hash function
2. 得到某个数值
3. 再把它映到 hash table 的某个 index / bucket

于是查找时就不再是：

- 在整个条目列表里线性扫

而是：

- 直接跳到某个预期位置附近去看

> [!note]
> 哈希的本质是“用计算换定位”，先算出 key 应该落在哪，而不是全表扫描。

### 8. hash table 为什么常用 list 来做底层容器
老师特别解释了 hash table 常被想成“一个很长的 list”。

原因不是它和普通 list 语义一样，而是因为：

- list indexing 本身是常数时间

如果 hash function 能给出目标 bucket 的 index，  
那么 hash table 的底层就可以借助 list 的 O(1) index access。

所以 այստեղ课程其实把：

- list 的底层优势
- dict 的高层接口

通过 hashing 连接起来了。

### 9. collision：不同 key 可能 hash 到同一个 bucket
老师接着解释一个不可避免的问题：collision。

因为：

- key 的可能空间巨大
- hash table 的桶数量有限

所以不同 key 可能映到同一位置。

这时常见做法是：

- 该 bucket 里再存一个小列表
- 把所有碰撞进去的 entries 放进去

于是查找时：

- 先靠 hash function 快速缩小范围
- 再在那个小 bucket 里局部扫描

### 10. 什么叫 hashable：为什么 list 不能做 dict key
老师随后把前面 mutable / immutable 的内容重新拉回来。

一个对象若想做 dict key，必须是 **hashable**。

核心要求是：

- 多次对同一个对象做 hash，结果必须稳定

这也是为什么：

- int、str、tuple 通常可以
- list 不行

因为 list 是 mutable，如果内容变了，原来的 hash 位置就会失去意义。

### 11. 好 hash function 要满足什么
课堂还讨论了 hash function / hash table pair 的好坏标准。

大致包括：

- 结果要稳定
- 计算不能太慢
- 尽量把 keys 分散开
- 尽量减少碰撞

这一步的意义在于让你知道：

- “平均接近常数时间”不是白送的
- 它建立在合适的 hash design 上

### 12. simulations：把概率问题交给计算机反复试验
讲完 hashing 后，课堂最后一个技术主题是 simulation。

老师把 simulation 的一般框架说得很清楚：

1. 定义一次随机实验
2. 重复很多次
3. 统计结果
4. 用相对频率或平均值近似真实概率 / 期望

这部分很重要，因为它展示了 computation 在“解析解不好写”时的另一种力量。

### 13. 骰子例子：频率逼近概率
第一个 simulation 例子是掷骰子。

老师没有从公式出发，而是直接：

- 设定骰子六个面
- 重复滚很多次
- 统计某一面出现比例

随着模拟次数增加，得到的比例会越来越接近真实概率。

这让“probability as long-run frequency”在程序里变得非常具体。

### 14. 更复杂的骰子实验：at least k times out of N rolls
老师接着把单次掷骰扩展成更复杂事件：

- 一次实验里掷 `N` 次
- 统计某一面至少出现 `k` 次的概率

这时 simulation 框架仍然完全一样，只是：

- 单次实验内部的结构更复杂

这说明 simulation 的一般性很强：

- 你只要能定义一次实验如何进行
- 就能把它重复很多次

### 15. `fill_pool(size)`：simulation 也能近似连续随机量问题
最后老师给出一个更贴近现实建模的例子：

- 水龙头流速在 1 到 3 gallons/min 间随机波动
- 想估计填满泳池要多久

这时每次实验不是掷离散骰子，而是：

- 生成一个区间上的随机实数
- 计算对应填满时间

然后再做很多次，取平均。

这一步非常好，因为它告诉你：

- simulation 不只适用于离散概率
- 连续随机变量也能做

### 16. 这节课最后真正做的是全课程回收
Lecture 26 到最后其实是在把整门课的很多主线重新收回来：

- list 的底层表示
- dict 与 hashing
- complexity intuition
- randomness and simulation

再加上老师对后续课程方向的提示，这一讲更像整门 6.100L 的 closure。

## Exercise log

> [!warning] No official finger exercise
> 这讲官方没有单独的 finger exercise 文件。

最适合按课堂内容做的自测有三类：

- 口头解释为什么 `L[i]` 是 `Theta(1)` 而 `x in L` 是 `Theta(n)`。
- 自己画一个小 hash table，模拟几次不同名字 hash 到 bucket 的过程。
- 自己实现一次简单 simulation，比如 10000 次掷骰统计某事件概率。

这三步分别对应本讲三条主线：

- list implementation intuition
- hashing intuition
- simulation framework

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec26.pdf|Lecture 26 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec26_code.py|Lecture 26 code (py)]]
- Finger exercise: no official file for this lecture
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec26_transcript.pdf|Lecture 26 transcript]]
- Recitation: none attached to this lecture week
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 12.3, Ch 17)

## Review checklist
- [ ] 我能解释为什么 list index access 是 `Theta(1)`。
- [ ] 我能说明 contiguous memory / 地址偏移为什么支持常数时间访问。
- [ ] 我能解释为什么 equality、membership、iteration 对 list 是线性的。
- [ ] 我能说明如果把 dict 朴素存成 entry 列表，lookup 为什么会是 `Theta(n)`。
- [ ] 我能解释 hash function 和 hash table 分别扮演什么角色。
- [ ] 我能说明 collision 是什么，以及为什么它不可避免。
- [ ] 我能解释为什么 mutable list 不能做 dict key。
- [ ] 我能复述 simulation 的一般框架：定义实验、重复、统计。
- [ ] 我能用骰子或 fill_pool 例子说明 simulation 如何逼近概率或平均值。
- [ ] 我能按课堂顺序复述：list internals -> hashing -> simulations -> wrap-up。

> [!warning] Common mistakes
> - 只背“dict 查找快”，却不知道快在哈希和底层索引。
> - 把 hashable 理解成“任何对象都能 hash”。
> - 忘记 collision 后还需要在 bucket 里继续区分条目。
> - 把 simulation 理解成“随机跑一遍”，却没有大量重复和统计。

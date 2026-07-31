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
> <!-- bilingual-en:start -->
> - The final lecture ties up several loose ends. Although it spans multiple topics, its sequence is clear.
> - It first returns to lists: why indexed access is `Theta(1)` and how a list is represented in memory.
> - Contiguous memory and address offsets are the key to understanding list-access complexity.
> - It then turns to dictionaries and asks why lookup would degrade to `Theta(n)` if a dictionary were stored as a simple list of entries.
> - The central idea of hashing is not the hash value itself, but using a hash function to map a key to a bucket or index.
> - Only hashable objects can be dictionary keys, reconnecting the discussion to mutability and immutability.
> - A good hash function and a suitable table size control collision rates and average performance.
> - Simulations then connect randomness, probability, and computation, showing how much experimental modeling the course now enables.
> - The dice and `fill_pool` examples use one framework: define a trial, repeat it many times, and use frequencies or averages to approximate a probability or expectation.
> - By the end, lists, dictionaries, hashing, and simulation should feel like a synthesis of earlier course material.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 开场先说明：今天是收尾和 wrap-up
<!-- bilingual-en:start -->
*1. Opening: Tying Up Loose Ends and Wrapping Up*
<!-- bilingual-en:end -->
Lecture 26 一开始就说这会是最后一讲，内容会分成几个 loose ends：

- lists
- dictionaries / hashing
- simulations
- 课程总结

所以这讲表面上看跨度很大，但它其实是在把前面学过的东西重新连回到底层实现和更广的应用场景上。
<!-- bilingual-en:start -->
Lecture 26 opens by identifying itself as the final lecture and dividing its loose ends into lists, dictionaries and hashing, simulations, and a course summary. The surface range is broad, but the lecture reconnects earlier ideas to their underlying implementations and to wider applications.
<!-- bilingual-en:end -->

### 2. 先回到 list：为什么某些操作是 `Theta(1)`，某些是 `Theta(n)`
<!-- bilingual-en:start -->
*2. Returning to Lists: Why Some Operations Are `Theta(1)` and Others `Theta(n)`*
<!-- bilingual-en:end -->
老师先回顾我们已经知道的 list 操作复杂度：

- equality：`Theta(n)`
- membership：`Theta(n)`
- iteration：`Theta(n)`
- direct index access：`Theta(1)`

然后问了一个非常关键的问题：

- 为什么 index access 会是常数时间？

这就把课堂从“背复杂度表”推进到“理解实现原因”。
<!-- bilingual-en:start -->
The instructor reviews familiar list complexities: equality, membership, and iteration are `Theta(n)`, while direct indexed access is `Theta(1)`. Asking why indexed access is constant-time moves the lesson from memorizing a table to understanding its implementation.
<!-- bilingual-en:end -->

### 3. contiguous memory：列表为什么能常数时间取第 i 个元素
<!-- bilingual-en:start -->
*3. Contiguous Memory: Why the ith Element Can Be Accessed in Constant Time*
<!-- bilingual-en:end -->
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
<!-- bilingual-en:start -->
The list can be modeled as a contiguous region of memory. Given its starting address, the fixed size of each slot, and an index `i`, the program calculates the address of slot `i` directly. It need not count through preceding elements; one offset calculation produces the `Theta(1)` access bound.
<!-- bilingual-en:end -->

### 4. list 里装的往往不是值本身，而是引用
<!-- bilingual-en:start -->
*4. A List Usually Stores References Rather than the Objects Themselves*
<!-- bilingual-en:end -->
老师随后补充，真实 Python list 里不一定直接放原始整数值。

如果 list 元素是：

- 另一个 list
- 一个 dictionary
- 更复杂的对象

那 list 本身存放的更像是引用 / 指针。

这一步很重要，因为它帮你把前面学过的 aliasing、对象、嵌套结构与底层表示重新连起来。
<!-- bilingual-en:start -->
The instructor adds that a real Python list does not necessarily store raw values directly. When elements are other lists, dictionaries, or more complex objects, its slots hold references. This reconnects aliasing, objects, and nested structures with their underlying representation.
<!-- bilingual-en:end -->

### 5. 为什么 equality / membership / iteration 是线性
<!-- bilingual-en:start -->
*5. Why Equality, Membership, and Iteration Are Linear*
<!-- bilingual-en:end -->
和 index access 对照起来，老师再次强调：

- 判断两个 list 是否相等，要逐元素比
- 判断某元素是否在 list 里，要一个个扫
- 遍历 list 当然也要逐个访问

所以这些操作天然和长度成正比。

这一段实际上是在示范一种更成熟的复杂度理解：

- 不是背表，而是回到“为了完成这个任务，最少得看多少数据”
<!-- bilingual-en:start -->
Unlike indexed access, list equality may require comparing every element, membership may require scanning the entire list, and iteration visits each element. The mature way to reason about complexity is therefore to ask how much data a task must inspect, not merely to memorize a table.
<!-- bilingual-en:end -->

### 6. 从 list 过渡到 dict：为什么 dict 不能像 list 那样按位置找
<!-- bilingual-en:start -->
*6. From Lists to Dictionaries: Why a Dictionary Needs More than Positional Access*
<!-- bilingual-en:end -->
接着老师切到 dictionary。

如果我们天真地把 dict 存成一串 entries：

- 每个 entry 是 `[key, value]`
- 所有 entry 排成一个长 list

那么查某个 key 时，就只能：

- 从头扫到尾
- 一个个比 key

这就会退化成 `Theta(n)`。

所以 dict 想快，必须有别的组织方法。
<!-- bilingual-en:start -->
The lecture next considers a naive dictionary represented as a long list of `[key, value]` entries. Finding a key would require comparing entries from beginning to end, yielding `Theta(n)` lookup. Fast dictionary access therefore requires a different organization.
<!-- bilingual-en:end -->

### 7. hashing：把 key 映射到 hash table 的某个位置
<!-- bilingual-en:start -->
*7. Hashing: Mapping a Key to a Position in a Hash Table*
<!-- bilingual-en:end -->
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
> <!-- bilingual-en:start -->
> Hashing exchanges computation for location: calculate where a key should be instead of scanning the entire table.
> <!-- bilingual-en:end -->
> <!-- bilingual-en:start -->
> Hashing applies a hash function to a key, obtains a numerical value, and maps it to an index or bucket. Lookup can then jump near the expected location rather than scanning every entry.
> <!-- bilingual-en:end -->

### 8. hash table 为什么常用 list 来做底层容器
<!-- bilingual-en:start -->
*8. Why a Hash Table Often Uses a List as Its Underlying Container*
<!-- bilingual-en:end -->
老师特别解释了 hash table 常被想成“一个很长的 list”。

原因不是它和普通 list 语义一样，而是因为：

- list indexing 本身是常数时间

如果 hash function 能给出目标 bucket 的 index，  
那么 hash table 的底层就可以借助 list 的 O(1) index access。

所以 այստեղ课程其实把：

- list 的底层优势
- dict 的高层接口

通过 hashing 连接起来了。
<!-- bilingual-en:start -->
A hash table is often modeled as a long list because list indexing is constant-time. Once the hash function supplies a bucket index, the table can reuse that direct access. Hashing thereby connects the low-level advantage of lists to the higher-level dictionary interface.
<!-- bilingual-en:end -->

### 9. collision：不同 key 可能 hash 到同一个 bucket
<!-- bilingual-en:start -->
*9. Collisions: Different Keys Can Hash to the Same Bucket*
<!-- bilingual-en:end -->
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
<!-- bilingual-en:start -->
Collisions are unavoidable because the space of possible keys is much larger than the finite number of table buckets. One common strategy stores a small list of entries within each bucket: hashing narrows the search quickly, and lookup then scans only that local list.
<!-- bilingual-en:end -->

### 10. 什么叫 hashable：为什么 list 不能做 dict key
<!-- bilingual-en:start -->
*10. Hashability: Why a List Cannot Be a Dictionary Key*
<!-- bilingual-en:end -->
老师随后把前面 mutable / immutable 的内容重新拉回来。

一个对象若想做 dict key，必须是 **hashable**。

核心要求是：

- 多次对同一个对象做 hash，结果必须稳定

这也是为什么：

- int、str、tuple 通常可以
- list 不行

因为 list 是 mutable，如果内容变了，原来的 hash 位置就会失去意义。
<!-- bilingual-en:start -->
To serve as a dictionary key, an object must be hashable: its hash must remain stable while it is in the dictionary. Integers, strings, and suitable tuples are normally hashable, whereas a list is not. A mutable list could change after insertion and invalidate the bucket chosen from its earlier contents.
<!-- bilingual-en:end -->

### 11. 好 hash function 要满足什么
<!-- bilingual-en:start -->
*11. What Makes a Good Hash Function*
<!-- bilingual-en:end -->
课堂还讨论了 hash function / hash table pair 的好坏标准。

大致包括：

- 结果要稳定
- 计算不能太慢
- 尽量把 keys 分散开
- 尽量减少碰撞

这一步的意义在于让你知道：

- “平均接近常数时间”不是白送的
- 它建立在合适的 hash design 上
<!-- bilingual-en:start -->
The lecture evaluates the hash-function and table pair by whether hashing is stable and inexpensive, distributes keys broadly, and limits collisions. Near-constant average lookup is not free; it depends on an appropriate hash design.
<!-- bilingual-en:end -->

### 12. simulations：把概率问题交给计算机反复试验
<!-- bilingual-en:start -->
*12. Simulations: Repeating Probability Experiments on a Computer*
<!-- bilingual-en:end -->
讲完 hashing 后，课堂最后一个技术主题是 simulation。

老师把 simulation 的一般框架说得很清楚：

1. 定义一次随机实验
2. 重复很多次
3. 统计结果
4. 用相对频率或平均值近似真实概率 / 期望

这部分很重要，因为它展示了 computation 在“解析解不好写”时的另一种力量。
<!-- bilingual-en:start -->
The lecture's final technical topic is simulation. Its general framework is to define one random trial, repeat it many times, record outcomes, and use relative frequencies or averages to approximate a probability or expectation. This provides a computational route when an analytic solution is difficult to derive.
<!-- bilingual-en:end -->

### 13. 骰子例子：频率逼近概率
<!-- bilingual-en:start -->
*13. Dice Example: Frequencies Approach Probabilities*
<!-- bilingual-en:end -->
第一个 simulation 例子是掷骰子。

老师没有从公式出发，而是直接：

- 设定骰子六个面
- 重复滚很多次
- 统计某一面出现比例

随着模拟次数增加，得到的比例会越来越接近真实概率。

这让“probability as long-run frequency”在程序里变得非常具体。
<!-- bilingual-en:start -->
The first simulation represents a six-sided die, rolls it many times, and records the proportion of one face. As the number of trials grows, the observed proportion approaches the true probability, making the long-run frequency interpretation of probability concrete in code.
<!-- bilingual-en:end -->

### 14. 更复杂的骰子实验：at least k times out of N rolls
<!-- bilingual-en:start -->
*14. A More Complex Dice Event: At Least k Successes in N Rolls*
<!-- bilingual-en:end -->
老师接着把单次掷骰扩展成更复杂事件：

- 一次实验里掷 `N` 次
- 统计某一面至少出现 `k` 次的概率

这时 simulation 框架仍然完全一样，只是：

- 单次实验内部的结构更复杂

这说明 simulation 的一般性很强：

- 你只要能定义一次实验如何进行
- 就能把它重复很多次
<!-- bilingual-en:start -->
The next example defines one trial as `N` rolls and asks for the probability that a chosen face appears at least `k` times. The simulation framework is unchanged; only the internal structure of one trial becomes more complex. Any experiment that can be specified once can be repeated many times.
<!-- bilingual-en:end -->

### 15. `fill_pool(size)`：simulation 也能近似连续随机量问题
<!-- bilingual-en:start -->
*15. `fill_pool(size)`: Simulating a Continuous Random Quantity*
<!-- bilingual-en:end -->
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
<!-- bilingual-en:start -->
The final example models a faucet whose flow rate varies randomly between one and three gallons per minute and estimates the time required to fill a pool. Each trial draws a continuous random value, calculates the corresponding fill time, and contributes to an average over many trials. Simulation therefore applies to continuous as well as discrete random variables.
<!-- bilingual-en:end -->

### 16. 这节课最后真正做的是全课程回收
<!-- bilingual-en:start -->
*16. The Final Lecture Synthesizes the Whole Course*
<!-- bilingual-en:end -->
Lecture 26 到最后其实是在把整门课的很多主线重新收回来：

- list 的底层表示
- dict 与 hashing
- complexity intuition
- randomness and simulation

再加上老师对后续课程方向的提示，这一讲更像整门 6.100L 的 closure。
<!-- bilingual-en:start -->
Lecture 26 closes by reconnecting list representation, dictionaries and hashing, complexity intuition, randomness, and simulation. Together with the instructor's pointers to later courses, it functions as a conclusion to 6.100L as a whole.
<!-- bilingual-en:end -->

## Exercise log

> [!warning] No official finger exercise
> 这讲官方没有单独的 finger exercise 文件。
> <!-- bilingual-en:start -->
> There is no separate official finger exercise file for this lecture.
> <!-- bilingual-en:end -->

最适合按课堂内容做的自测有三类：

- 口头解释为什么 `L[i]` 是 `Theta(1)` 而 `x in L` 是 `Theta(n)`。
- 自己画一个小 hash table，模拟几次不同名字 hash 到 bucket 的过程。
- 自己实现一次简单 simulation，比如 10000 次掷骰统计某事件概率。

这三步分别对应本讲三条主线：

- list implementation intuition
- hashing intuition
- simulation framework
<!-- bilingual-en:start -->
Three self-tests fit the lecture particularly well:

- Explain aloud why `L[i]` is `Theta(1)` whereas `x in L` is `Theta(n)`.
- Draw a small hash table and simulate several names mapping to buckets.
- Implement a simple simulation, such as rolling a die 10,000 times to estimate an event probability.

These test the three main strands: intuition about list implementation, hashing, and the simulation framework.
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
- [ ] I can explain why indexed list access is `Theta(1)`.
- [ ] I can explain how contiguous memory and address offsets support constant-time access.
- [ ] I can explain why list equality, membership, and iteration are linear.
- [ ] I can explain why a dictionary stored naively as a list of entries would have `Theta(n)` lookup.
- [ ] I can distinguish the roles of a hash function and a hash table.
- [ ] I can explain what a collision is and why collisions are unavoidable.
- [ ] I can explain why a mutable list cannot be a dictionary key.
- [ ] I can reconstruct the simulation framework: define a trial, repeat it, and summarize outcomes.
- [ ] I can use the dice or `fill_pool` example to show how simulation approximates a probability or mean.
- [ ] I can reconstruct the lecture sequence: list internals -> hashing -> simulations -> wrap-up.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 只背“dict 查找快”，却不知道快在哈希和底层索引。
> - 把 hashable 理解成“任何对象都能 hash”。
> - 忘记 collision 后还需要在 bucket 里继续区分条目。
> - 把 simulation 理解成“随机跑一遍”，却没有大量重复和统计。
> <!-- bilingual-en:start -->
> - Memorizing that dictionary lookup is fast without understanding hashing and indexed access.
> - Interpreting “hashable” as meaning that every object can be hashed.
> - Forgetting that entries in a collided bucket still need to be distinguished.
> - Treating simulation as one random run rather than many repetitions followed by statistical summarization.
> <!-- bilingual-en:end -->

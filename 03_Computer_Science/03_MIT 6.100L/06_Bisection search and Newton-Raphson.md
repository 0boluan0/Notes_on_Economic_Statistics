---
aliases:
  - MIT 6.100L Lecture 06
  - 6.100L L06
  - Bisection search and Newton-Raphson
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 06
---

# Lecture 06: Bisection search and Newton-Raphson

> [!tip] Hint
> - 这节课一开始不是讲新算法，而是先回顾上节 approximation method 为什么正确却太慢。
> - 老师用 “448 页课本找藏着的 100 美元” 的游戏把 bisection search 的直觉先种下去，再讲平方根代码。
> - 二分搜索的关键不是“取中点”四个字，而是问题必须有内在顺序，且答案始终被当前区间包住。
> - low/high/guess 三个变量不是并列的：low 和 high 描述不确定区间，guess 只是当前试探点。
> - `x < 1` 时区间不能再写成 `[0, x]`，这是课堂中专门拎出来修补的边界。
> - cube root 的 you try it 是要你把 “平方根上的二分” 迁移到另一种单调函数上。
> - Newton-Raphson 不是凭空降临的新魔法，而是“利用局部斜率决定修正步长”的第三种近似策略。
> - 本讲一直在比较三件事：fixed increment、bisection、Newton，不只是背它们各自代码。
> - bisection 的速度来自每次砍半区间，Newton 的速度来自利用导数信息做更聪明的跳跃。
> - 这节课的核心不是把公式背熟，而是理解“如何用问题结构换速度”。

## Lecture flow

### 1. 开场先回顾：approximation method 为什么不够好
Lecture 6 的出发点非常直接：  
上节课我们已经有了 approximation method，它是对的，但它太慢了。

老师先回顾上一讲的平方根近似代码：

```python
while abs(guess**2 - x) >= epsilon and guess**2 <= x:
    guess += increment
```

这个方法的问题不是会不会成功，而是：

- 走得太小步
- 每次只前进一个固定 increment
- 对大输入时效率极差

所以这一讲的问题变成：

> 我们能不能保留“逐步逼近”这件事，  
> 但让每一步走得更聪明？

### 2. 先用“猜书页上的 100 美元”建立 bisection 的直觉
在正式讲代码之前，老师先用了一个非常经典的猜页码游戏。

设定是：

- 一本 448 页的书
- 某一页夹着 100 美元
- 如果只允许在 8 次以内猜中页码就赢

如果你每次只得到“对/错”，这个游戏并不友好；  
但如果你每次都能知道“猜大了还是猜小了”，事情就完全变了。

因为这时最自然的策略就是：

- 每次猜当前区间的 midpoint
- 根据“太大/太小”决定保留哪一半

老师用这个例子不是为了讲游戏，而是为了让你先在离散场景里感受到：

> [!note]
> bisection search 的威力来自“每一步都能排除一半候选空间”。

### 3. bisection search 的适用前提：问题必须自带顺序
老师随后把直觉收成正式条件：

- 问题要有 inherent order
- 你要知道答案落在某个 interval 里
- midpoint 左右两边必须能根据反馈排除掉一半

这几点缺一不可。

如果没有顺序，根本谈不上 “左半边 / 右半边”；  
如果不知道答案一开始在哪个区间，也没法安全地半分。

所以 bisection 不是一个可以胡乱套用的模板，而是一种依赖问题结构的搜索策略。

### 4. 从 approximation 的“线性前进”切换到 bisection 的“区间收缩”
讲完动机后，老师把平方根问题重新画成区间。

设我们要找 `sqrt(x)`，并且先假设 `x >= 1`。  
那么答案一定落在区间：

- `low = 0`
- `high = x`

然后我们取中点：

```python
guess = (high + low) / 2
```

此时有三种情况：

- `guess**2` 已经足够接近 `x`
- `guess**2 < x`，说明 guess 偏小，应该抬高下界
- `guess**2 > x`，说明 guess 偏大，应该压低上界

所以 unlike fixed increment：

- 我们不再机械向前加一个小步长
- 我们改成重设搜索区间

### 5. square root 上的 bisection 代码：每一轮都维护不变量
对应代码大致是：

```python
x = 54321
epsilon = 0.01
low = 0.0
high = x
guess = (high + low) / 2

while abs(guess**2 - x) >= epsilon:
    if guess**2 < x:
        low = guess
    else:
        high = guess
    guess = (high + low) / 2.0
```

这段代码的真正逻辑不是 if/else，而是它一直维护一个关键不变量：

> [!note]
> 每一轮结束后，真正答案仍然被包在 `[low, high]` 这个区间里。

如果这个不变量没守住，二分搜索就失去意义了。

所以你在读这段程序时，不能只盯着 “更新了哪个变量”，而要盯：

- 为什么更新后的新区间仍然合法？

### 6. 为什么它比 approximation 快：不是常数更好，而是增长方式变了
老师在讲义里明确对比了两种搜索：

- exhaustive / approximation 风格：每次只缩掉一点点
- bisection：每次直接砍掉一半

这意味着 candidate space 的大小变化完全不同：

- 线性式减少：`N -> N-1`
- 对数式减少：`N -> N/2 -> N/4 -> N/8 ...`

哪怕你还没有正式学复杂度记号，此时也应该先形成直觉：

- 不是所有“逐步逼近”都一样快
- 关键看每一步到底缩掉了多少不确定性

### 7. 边界修补：`x < 1` 时，区间不能再设成 `[0, x]`
老师随后专门做了一个 you try it：  
如果 `x = 0.5`，那 low 和 high 应该怎么设？

这一步很关键，因为它说明：

- bisection 的框架没变
- 但初始区间必须真的包住答案

对平方根来说：

- 如果 `x >= 1`，答案在 `[1, x]` 或 `[0, x]` 都可描述
- 如果 `0 < x < 1`，平方根其实比 `x` 大，而不是更小

所以更完整的版本写成：

```python
if x >= 1:
    low = 1.0
    high = x
else:
    low = x
    high = 1.0
```

> [!warning]
> 很多二分搜索 bug 都不是出在循环体，而是出在初始区间压根没把答案包进去。

### 8. cube root you try it：把同一搜索结构迁移到另一种单调函数
讲完 square root 后，老师立刻让大家写 cube root 版本：

```python
cube = 27
epsilon = 0.01
low = 0
high = cube
guess = (high + low) / 2.0
while abs(guess**3 - cube) >= epsilon:
    if guess**3 < cube:
        low = guess
    else:
        high = guess
    guess = (high + low) / 2.0
```

这一步的意义很明确：  
老师要你意识到 bisection 不是“平方根专属模板”，而是依赖以下结构：

- 有有序区间
- 有单调关系
- 中点比较能告诉你保留哪一半

只要这些结构还在，目标函数可以换。

### 9. 再进一步：对负 cube 也能做，但要先处理 sign
老师随后又展示了 all-cubes 版本，包括 negative cube。

做法和前面整数 cube root 类似：

- 先记下原数是否为负
- 对绝对值做搜索
- 最后再把结果符号改回来

这段仍然是在训练同一个习惯：

- 算法主骨架可以保留
- 输入预处理和结果后处理单独写

### 10. Newton-Raphson：第三种逼近方式，不再靠区间，而靠局部导数
讲到最后一部分，老师引入 Newton-Raphson。

这时课堂的语气其实已经不是“再学一个公式”，而是：

> 我们已经见过两种方法了，  
> 现在来看一种更 aggressively smart 的方法。

对于平方根问题，Newton 更新大致写成：

```python
guess = guess - (((guess**2) - k) / (2 * guess))
```

老师没有把重点放在推导证明，而是强调它的思路：

- 你当前 guess 附近，函数有局部斜率
- 这个斜率能告诉你，应该修正多少
- 所以下一步不再是固定小步，也不再是区间中点
- 而是根据局部信息跳到一个更有希望的位置

### 11. Newton 为什么常常更快，但也更依赖函数结构
课堂里这一部分虽然简短，但你最好在笔记里明确记下来：

- approximation method：最笨，但很直观
- bisection：利用 order 和 interval，每次砍半
- Newton：利用 derivative 信息，通常更快

它们的差别不是只在代码长短，而在于每一步使用了多少问题结构：

- fixed increment：几乎不用结构
- bisection：用顺序和单调
- Newton：再进一步用局部斜率

所以速度提升的本质，是你愿意并且能够利用更多信息。

### 12. 本讲真正的收尾：算法不是越神奇越好，而是越贴合问题结构越好
Lecture 6 到最后，其实是在帮你重建一个更成熟的算法直觉：

- 同一个数值问题可以有多种算法
- 快慢差别很大
- 差别来自你用了什么结构信息

从现在开始，看到一个问题时，除了问 “能不能做”，你也应该开始问：

- 它有没有 order？
- 有没有 natural interval？
- 有没有 derivative 或其他局部信息？
- 我能不能别再傻乎乎一个小步一个小步试？

## Exercise log
> [!example] Finger exercise 06
> 官方题目要求：在 `0 <= N <= 1000` 的前提下，用 bisection search 猜出 `N`，并打印猜了多少次和最终答案。
>
> ```python
> low = 0
> high = 1001
> guess = (high + low) // 2
> count = 1
> while guess != N:
>     if guess < N:
>         low = guess
>     elif guess > N:
>         high = guess
>     guess = (high + low) // 2
>     count += 1
> print("count:", count)
> print("answer:", guess)
> ```
>
> 这题非常好，因为它把 bisection 的骨架抽得更干净了：
> - 没有平方和立方，只有 ordered interval
> - 没有 epsilon，直接是精确整数答案
> - 但 low / high / midpoint / shrink-half 的结构完全一样
>
> 如果你能把这题做顺，说明你已经抓住的是搜索结构，而不是某个具体数学公式。

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec06.pdf|Lecture 06 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec06_code.py|Lecture 06 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex06_sol.pdf|Lecture 06 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec06_transcript.pdf|Lecture 06 transcript]]
- Recitation 3: [[MIT 6.100L-recitations/mit6_100l_rec03.zip|Recitation 03 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 3.4-3.5)

## Review checklist
- [ ] 我能解释为什么 approximation method 虽然正确却常常太慢。
- [ ] 我能复述“猜课本页码”例子为什么会自然导向 midpoint strategy。
- [ ] 我能说清楚 bisection search 的适用前提，而不是只背代码模板。
- [ ] 我能解释 low、high、guess 在每一轮中各自承担什么角色。
- [ ] 我能说明为什么更新 low 或 high 后，答案仍然被包在区间里。
- [ ] 我能处理 `x < 1` 时平方根区间的特殊情况。
- [ ] 我能把 bisection 从 square root 迁移到 cube root。
- [ ] 我能用自己的话解释 Newton-Raphson 的更新思想，而不是只背公式。
- [ ] 我能比较 fixed increment、bisection、Newton-Raphson 三者使用了哪些结构信息。
- [ ] 我能解释为什么“如何利用问题结构”决定了算法快慢。

> [!warning] Common mistakes
> - 没有保证真实答案一开始就在 `[low, high]` 区间内。
> - 把 bisection 机械套用到不满足单调/有序结构的问题上。
> - 只记 Newton 公式，不理解它为什么可能快、也为什么可能不稳。

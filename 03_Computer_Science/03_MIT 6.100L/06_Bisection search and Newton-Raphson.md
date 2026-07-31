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
> <!-- bilingual-en:start -->
> - The lecture begins not with a new algorithm, but by revisiting why the previous approximation method is correct yet slow.
> - A game about finding a hidden $100 bill in a 448-page textbook develops the intuition for bisection before the square-root code appears.
> - Bisection is not merely “take the midpoint.” The search space must be ordered, and the current interval must continue to contain the answer.
> - `low` and `high` describe the remaining uncertainty; `guess` is only the current probe, so the three variables do not play equivalent roles.
> - For `x < 1`, `[0, x]` cannot contain the square root. The lecture treats this as an explicit boundary correction.
> - The cube-root exercise transfers bisection from one monotone function to another.
> - Newton–Raphson is not an unrelated trick, but a third approximation strategy that uses local slope to choose a correction.
> - The lecture continually compares fixed increments, bisection, and Newton's method rather than presenting three isolated code templates.
> - Bisection gains speed by halving an interval; Newton's method gains speed by using derivative information to take a better-informed step.
> - The central lesson is not memorizing formulas, but exchanging additional problem structure for speed.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 开场先回顾：approximation method 为什么不够好
<!-- bilingual-en:start -->
*1. Opening Review: Why the Approximation Method Is Not Good Enough*
<!-- bilingual-en:end -->
Lecture 6 的出发点非常直接：  
上节课我们已经有了 approximation method，它是对的，但它太慢了。
<!-- bilingual-en:start -->
Lecture 6 starts from a direct observation: the previous approximation method works, but it is too slow.
<!-- bilingual-en:end -->

老师先回顾上一讲的平方根近似代码：

```python
while abs(guess**2 - x) >= epsilon and guess**2 <= x:
    guess += increment
```

这个方法的问题不是会不会成功，而是：

- 走得太小步
- 每次只前进一个固定 increment
- 对大输入时效率极差
<!-- bilingual-en:start -->
Its problem is not correctness. It takes a tiny fixed step on every iteration and consequently performs very poorly on large inputs.
<!-- bilingual-en:end -->

所以这一讲的问题变成：

> 我们能不能保留“逐步逼近”这件事，  
> 但让每一步走得更聪明？
> <!-- bilingual-en:start -->
> Can we preserve gradual approximation while making each step better informed?
> <!-- bilingual-en:end -->

### 2. 先用“猜书页上的 100 美元”建立 bisection 的直觉
<!-- bilingual-en:start -->
*2. Building Bisection Intuition with the Hidden $100 Page Game*
<!-- bilingual-en:end -->
在正式讲代码之前，老师先用了一个非常经典的猜页码游戏。
<!-- bilingual-en:start -->
Before presenting code, the instructor uses a page-guessing game.
<!-- bilingual-en:end -->

设定是：

- 一本 448 页的书
- 某一页夹着 100 美元
- 如果只允许在 8 次以内猜中页码就赢
<!-- bilingual-en:start -->
- A book has 448 pages.
- A $100 bill is hidden on one page.
- You win only if you identify the page within eight guesses.
<!-- bilingual-en:end -->

如果你每次只得到“对/错”，这个游戏并不友好；  
但如果你每次都能知道“猜大了还是猜小了”，事情就完全变了。
<!-- bilingual-en:start -->
Simple right-or-wrong feedback is not very helpful. Learning whether each guess is too high or too low changes the problem completely.
<!-- bilingual-en:end -->

因为这时最自然的策略就是：

- 每次猜当前区间的 midpoint
- 根据“太大/太小”决定保留哪一半
<!-- bilingual-en:start -->
The natural strategy is to guess the midpoint of the current interval and retain the half consistent with the feedback.
<!-- bilingual-en:end -->

老师用这个例子不是为了讲游戏，而是为了让你先在离散场景里感受到：

> [!note]
> bisection search 的威力来自“每一步都能排除一半候选空间”。
> <!-- bilingual-en:start -->
> Bisection is powerful because every step eliminates half of the remaining candidate space.
> <!-- bilingual-en:end -->

### 3. bisection search 的适用前提：问题必须自带顺序
<!-- bilingual-en:start -->
*3. A Prerequisite for Bisection: The Problem Must Be Ordered*
<!-- bilingual-en:end -->
老师随后把直觉收成正式条件：
<!-- bilingual-en:start -->
The instructor turns the intuition into explicit conditions:
<!-- bilingual-en:end -->

- 问题要有 inherent order
- 你要知道答案落在某个 interval 里
- midpoint 左右两边必须能根据反馈排除掉一半
<!-- bilingual-en:start -->
- The problem has an inherent order.
- The answer is known to lie in a specified interval.
- Comparing at the midpoint identifies one half that can be discarded.
<!-- bilingual-en:end -->

这几点缺一不可。

如果没有顺序，根本谈不上 “左半边 / 右半边”；  
如果不知道答案一开始在哪个区间，也没法安全地半分。
<!-- bilingual-en:start -->
Without order, “left half” and “right half” have no useful meaning; without an initial containing interval, neither half can be discarded safely.
<!-- bilingual-en:end -->

所以 bisection 不是一个可以胡乱套用的模板，而是一种依赖问题结构的搜索策略。
<!-- bilingual-en:start -->
Bisection is therefore a search strategy justified by problem structure, not a template that can be applied indiscriminately.
<!-- bilingual-en:end -->

### 4. 从 approximation 的“线性前进”切换到 bisection 的“区间收缩”
<!-- bilingual-en:start -->
*4. Moving from Linear Progress to Interval Contraction*
<!-- bilingual-en:end -->
讲完动机后，老师把平方根问题重新画成区间。
<!-- bilingual-en:start -->
The square-root problem is now reformulated as an interval search.
<!-- bilingual-en:end -->

设我们要找 `sqrt(x)`，并且先假设 `x >= 1`。  
那么答案一定落在区间：

- `low = 0`
- `high = x`
<!-- bilingual-en:start -->
For `x >= 1`, `sqrt(x)` is contained between `low = 0` and `high = x`.
<!-- bilingual-en:end -->

然后我们取中点：
<!-- bilingual-en:start -->
The current probe is the midpoint:
<!-- bilingual-en:end -->

```python
guess = (high + low) / 2
```

此时有三种情况：

- `guess**2` 已经足够接近 `x`
- `guess**2 < x`，说明 guess 偏小，应该抬高下界
- `guess**2 > x`，说明 guess 偏大，应该压低上界
<!-- bilingual-en:start -->
If `guess**2` is close enough to `x`, the search is complete. If it is too small, `guess` becomes the new lower bound; if it is too large, `guess` becomes the new upper bound.
<!-- bilingual-en:end -->

所以 unlike fixed increment：

- 我们不再机械向前加一个小步长
- 我们改成重设搜索区间
<!-- bilingual-en:start -->
Unlike fixed-increment search, this method does not advance mechanically by a small constant. It repeatedly replaces the uncertainty interval.
<!-- bilingual-en:end -->

### 5. square root 上的 bisection 代码：每一轮都维护不变量
<!-- bilingual-en:start -->
*5. Square-Root Bisection: Maintaining an Invariant on Every Iteration*
<!-- bilingual-en:end -->
对应代码大致是：
<!-- bilingual-en:start -->
The corresponding code is approximately:
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
The essential logic is not the surface `if`/`else`, but the invariant maintained throughout the loop:
<!-- bilingual-en:end -->

> [!note]
> 每一轮结束后，真正答案仍然被包在 `[low, high]` 这个区间里。
> <!-- bilingual-en:start -->
> At the end of every iteration, the true answer remains inside `[low, high]`.
> <!-- bilingual-en:end -->

如果这个不变量没守住，二分搜索就失去意义了。
<!-- bilingual-en:start -->
If that invariant fails, the bisection procedure is no longer justified.
<!-- bilingual-en:end -->

所以你在读这段程序时，不能只盯着 “更新了哪个变量”，而要盯：

- 为什么更新后的新区间仍然合法？
<!-- bilingual-en:start -->
When reading the code, ask not only which bound changes, but why the resulting interval is still guaranteed to contain the answer.
<!-- bilingual-en:end -->

### 6. 为什么它比 approximation 快：不是常数更好，而是增长方式变了
<!-- bilingual-en:start -->
*6. Why It Is Faster: The Scaling Changes, Not Merely a Constant*
<!-- bilingual-en:end -->
老师在讲义里明确对比了两种搜索：

- exhaustive / approximation 风格：每次只缩掉一点点
- bisection：每次直接砍掉一半
<!-- bilingual-en:start -->
The notes contrast exhaustive or fixed-increment approximation, which removes only a small amount of uncertainty per step, with bisection, which removes half.
<!-- bilingual-en:end -->

这意味着 candidate space 的大小变化完全不同：

- 线性式减少：`N -> N-1`
- 对数式减少：`N -> N/2 -> N/4 -> N/8 ...`
<!-- bilingual-en:start -->
The candidate space therefore shrinks linearly as `N -> N-1`, or geometrically as `N -> N/2 -> N/4 -> N/8 ...`.
<!-- bilingual-en:end -->

哪怕你还没有正式学复杂度记号，此时也应该先形成直觉：

- 不是所有“逐步逼近”都一样快
- 关键看每一步到底缩掉了多少不确定性
<!-- bilingual-en:start -->
Even before formal complexity notation, the correct intuition is that approximation methods differ dramatically in speed according to how much uncertainty each step removes.
<!-- bilingual-en:end -->

### 7. 边界修补：`x < 1` 时，区间不能再设成 `[0, x]`
<!-- bilingual-en:start -->
*7. Boundary Correction: `[0, x]` Fails When `x < 1`*
<!-- bilingual-en:end -->
老师随后专门做了一个 you try it：  
如果 `x = 0.5`，那 low 和 high 应该怎么设？
<!-- bilingual-en:start -->
The next exercise asks how to choose `low` and `high` when `x = 0.5`.
<!-- bilingual-en:end -->

这一步很关键，因为它说明：

- bisection 的框架没变
- 但初始区间必须真的包住答案
<!-- bilingual-en:start -->
The bisection framework remains unchanged, but its initial interval must actually contain the answer.
<!-- bilingual-en:end -->

对平方根来说：

- 如果 `x >= 1`，答案在 `[1, x]` 或 `[0, x]` 都可描述
- 如果 `0 < x < 1`，平方根其实比 `x` 大，而不是更小
<!-- bilingual-en:start -->
For `x >= 1`, `[1, x]` or `[0, x]` contains the square root. For `0 < x < 1`, however, the square root is greater than `x`, so `[0, x]` does not contain it.
<!-- bilingual-en:end -->

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
> <!-- bilingual-en:start -->
> Many bisection bugs are caused not by the loop body, but by an initial interval that never contained the answer.
> <!-- bilingual-en:end -->

### 8. cube root you try it：把同一搜索结构迁移到另一种单调函数
<!-- bilingual-en:start -->
*8. Cube-Root Exercise: Transferring the Search Structure to Another Monotone Function*
<!-- bilingual-en:end -->
讲完 square root 后，老师立刻让大家写 cube root 版本：
<!-- bilingual-en:start -->
Immediately after square roots, the instructor asks for a cube-root version:
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
The exercise shows that bisection is not a square-root-specific template. It depends on more general structure:
<!-- bilingual-en:end -->

- 有有序区间
- 有单调关系
- 中点比较能告诉你保留哪一半
<!-- bilingual-en:start -->
- An ordered interval.
- A monotone relationship.
- A midpoint comparison that identifies which half to keep.
<!-- bilingual-en:end -->

只要这些结构还在，目标函数可以换。
<!-- bilingual-en:start -->
As long as these properties remain, the target function can change.
<!-- bilingual-en:end -->

### 9. 再进一步：对负 cube 也能做，但要先处理 sign
<!-- bilingual-en:start -->
*9. Extending Further: Handling Negative Cubes Through Their Sign*
<!-- bilingual-en:end -->
老师随后又展示了 all-cubes 版本，包括 negative cube。
<!-- bilingual-en:start -->
The instructor then extends the procedure to all cubes, including negative inputs.
<!-- bilingual-en:end -->

做法和前面整数 cube root 类似：

- 先记下原数是否为负
- 对绝对值做搜索
- 最后再把结果符号改回来
<!-- bilingual-en:start -->
The algorithm records the original sign, searches using the absolute value, and restores the sign to the result.
<!-- bilingual-en:end -->

这段仍然是在训练同一个习惯：

- 算法主骨架可以保留
- 输入预处理和结果后处理单独写
<!-- bilingual-en:start -->
The same design habit appears again: preserve the main algorithmic skeleton while isolating input preprocessing and output postprocessing.
<!-- bilingual-en:end -->

### 10. Newton-Raphson：第三种逼近方式，不再靠区间，而靠局部导数
<!-- bilingual-en:start -->
*10. Newton–Raphson: Using a Local Derivative Instead of an Interval*
<!-- bilingual-en:end -->
讲到最后一部分，老师引入 Newton-Raphson。
<!-- bilingual-en:start -->
The final part introduces Newton–Raphson.
<!-- bilingual-en:end -->

这时课堂的语气其实已经不是“再学一个公式”，而是：

> 我们已经见过两种方法了，  
> 现在来看一种更 aggressively smart 的方法。
> <!-- bilingual-en:start -->
> We have seen two methods; now consider a more aggressively informed one.
> <!-- bilingual-en:end -->

对于平方根问题，Newton 更新大致写成：

```python
guess = guess - (((guess**2) - k) / (2 * guess))
```

老师没有把重点放在推导证明，而是强调它的思路：
<!-- bilingual-en:start -->
Rather than centering the derivation, the instructor emphasizes the mechanism:
<!-- bilingual-en:end -->

- 你当前 guess 附近，函数有局部斜率
- 这个斜率能告诉你，应该修正多少
- 所以下一步不再是固定小步，也不再是区间中点
- 而是根据局部信息跳到一个更有希望的位置
<!-- bilingual-en:start -->
- The function has a local slope near the current guess.
- That slope indicates how large a correction to make.
- The next point is therefore neither a fixed step nor an interval midpoint, but a location chosen from local information.
<!-- bilingual-en:end -->

### 11. Newton 为什么常常更快，但也更依赖函数结构
<!-- bilingual-en:start -->
*11. Why Newton's Method Is Often Faster but More Structure-Dependent*
<!-- bilingual-en:end -->
课堂里这一部分虽然简短，但你最好在笔记里明确记下来：
<!-- bilingual-en:start -->
Although brief, this comparison should remain explicit:
<!-- bilingual-en:end -->

- approximation method：最笨，但很直观
- bisection：利用 order 和 interval，每次砍半
- Newton：利用 derivative 信息，通常更快
<!-- bilingual-en:start -->
- Fixed-increment approximation is crude but intuitive.
- Bisection uses order and an interval to discard half the search space.
- Newton's method uses derivative information and is often faster.
<!-- bilingual-en:end -->

它们的差别不是只在代码长短，而在于每一步使用了多少问题结构：

- fixed increment：几乎不用结构
- bisection：用顺序和单调
- Newton：再进一步用局部斜率
<!-- bilingual-en:start -->
The code differs because each method exploits a different amount of structure: almost none for fixed increments, order and monotonicity for bisection, and local slope as well for Newton's method.
<!-- bilingual-en:end -->

所以速度提升的本质，是你愿意并且能够利用更多信息。
<!-- bilingual-en:start -->
The speed gain comes from being able and willing to use more information about the problem.
<!-- bilingual-en:end -->

### 12. 本讲真正的收尾：算法不是越神奇越好，而是越贴合问题结构越好
<!-- bilingual-en:start -->
*12. Closing Lesson: A Good Algorithm Fits the Problem's Structure*
<!-- bilingual-en:end -->
Lecture 6 到最后，其实是在帮你重建一个更成熟的算法直觉：
<!-- bilingual-en:start -->
Lecture 6 closes by building a more mature algorithmic intuition:
<!-- bilingual-en:end -->

- 同一个数值问题可以有多种算法
- 快慢差别很大
- 差别来自你用了什么结构信息
<!-- bilingual-en:start -->
The same numerical problem may admit several algorithms with very different running times, depending on the structural information they exploit.
<!-- bilingual-en:end -->

从现在开始，看到一个问题时，除了问 “能不能做”，你也应该开始问：

- 它有没有 order？
- 有没有 natural interval？
- 有没有 derivative 或其他局部信息？
- 我能不能别再傻乎乎一个小步一个小步试？
<!-- bilingual-en:start -->
Beyond asking whether a problem is solvable, ask whether it has an order, a natural containing interval, a derivative or other local information, and a way to avoid blindly taking one tiny step at a time.
<!-- bilingual-en:end -->

## Exercise log
> [!example] Finger exercise 06
> 官方题目要求：在 `0 <= N <= 1000` 的前提下，用 bisection search 猜出 `N`，并打印猜了多少次和最终答案。
> <!-- bilingual-en:start -->
> The official task is to identify `N`, where `0 <= N <= 1000`, by bisection and print both the number of guesses and the final answer.
> <!-- bilingual-en:end -->
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
> <!-- bilingual-en:start -->
> The exercise isolates the bisection skeleton particularly well:
> - There are no squares or cubes, only an ordered interval.
> - There is no tolerance; the target is an exact integer.
> - The `low` / `high` / midpoint / discard-half structure is unchanged.
> <!-- bilingual-en:end -->
>
> 如果你能把这题做顺，说明你已经抓住的是搜索结构，而不是某个具体数学公式。
> <!-- bilingual-en:start -->
> Solving it fluently shows that you understand the search structure rather than one particular mathematical formula.
> <!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
- [ ] I can explain why a correct fixed-increment approximation may still be very slow.
- [ ] I can reconstruct how the textbook-page game motivates a midpoint strategy.
- [ ] I can state the prerequisites for bisection instead of merely memorizing its code.
- [ ] I can explain the distinct roles of `low`, `high`, and `guess` on each iteration.
- [ ] I can justify why the answer remains inside the interval after either bound is updated.
- [ ] I can handle the square-root interval when `x < 1`.
- [ ] I can transfer bisection from square roots to cube roots.
- [ ] I can explain the Newton–Raphson update in my own words rather than only recite its formula.
- [ ] I can compare the structural information used by fixed increments, bisection, and Newton–Raphson.
- [ ] I can explain why the use of problem structure determines algorithmic speed.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 没有保证真实答案一开始就在 `[low, high]` 区间内。
> - 把 bisection 机械套用到不满足单调/有序结构的问题上。
> - 只记 Newton 公式，不理解它为什么可能快、也为什么可能不稳。
> <!-- bilingual-en:start -->
> - Failing to ensure that the true answer lies in the initial `[low, high]` interval.
> - Applying bisection mechanically to a problem without the required monotone or ordered structure.
> - Memorizing Newton's formula without understanding either its potential speed or its possible instability.
> <!-- bilingual-en:end -->

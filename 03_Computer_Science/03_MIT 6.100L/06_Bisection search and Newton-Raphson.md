---
aliases:
  - MIT 6.100L Lecture 06
  - 6.100L L06
  - Bisection Search and Newton-Raphson
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 06
---

# Lecture 06: Bisection Search and Newton-Raphson

> [!tip] Hint
> - 我能解释 bisection search 为何要求答案被夹在一个区间里。
> - 我能说明 low / high / guess 三者在迭代中怎样收缩不确定性。
> - 我能说出 Newton-Raphson 的更新思想：利用局部线性近似快速修正 guess。
> - 我能比较 fixed increment、bisection、Newton-Raphson 的速度与前提差异。
> - 我能围绕本讲的主轴 “Bisection search：每次砍掉一半不确定区间” / “Newton-Raphson：用局部斜率决定修正步长” / “选择哪种方法，取决于你知道多少结构”，不翻 slides 也把整节课重新讲一遍。
> - 我能说明 bisection search 成立的核心前提。
> - 我能解释 low / high / guess 在每轮迭代中的更新规则。
> - 我能口头讲出 Newton-Raphson 的更新思想，而不是只背公式。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 3.4-3.5
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Bisection search：每次砍掉一半不确定区间 / Newton-Raphson：用局部斜率决定修正步长 / 选择哪种方法，取决于你知道多少结构
> - 这一讲是对 Lecture 5 的数值近似升级：不再用笨办法一点点挪，而是用问题结构加速搜索。
> - Bisection search 的关键词是‘有序区间’，Newton-Raphson 的关键词是‘用导数信息快速修正’。
> - 这两种方法不仅是求根技巧，更是在训练你如何利用额外结构减少搜索空间。

## Core ideas
### Bisection search：每次砍掉一半不确定区间
当你确定答案在 `[low, high]` 之间，而且函数在这个区间里有稳定的顺序关系时，就可以每次拿中点试探。
- 当前 guess 取 `(low + high) / 2`；如果 guess 太小，就把 low 提到 guess，否则把 high 压到 guess。
- 算法真正维护的是 invariant：真正答案始终还在 `[low, high]` 内。
- 如果输入小于 `1`，区间选择要更小心，例如平方根问题应把区间设为 `[x, 1]` 而不是 `[0, x]`。
- 与固定步长相比，bisection 用更少猜测次数换来同等级别的精度。

> [!note] What to internalize
> - One-sentence takeaway: 当你确定答案在 `[low, high]` 之间，而且函数在这个区间里有稳定的顺序关系时，就可以每次拿中点试探。
> - Review anchor: 当前 guess 取 `(low + high) / 2`；如果 guess 太小，就把 low 提到 guess，否则把 high 压到 guess。
> - Review anchor: 算法真正维护的是 invariant：真正答案始终还在 `[low, high]` 内。

从做题角度看，只要题目在考“Bisection search：每次砍掉一半不确定区间”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：当你确定答案在 `[low, high]` 之间，而且函数在这个区间里有稳定的顺序关系时，就可以每次拿中点试探。

### Newton-Raphson：用局部斜率决定修正步长
Newton-Raphson 的直觉是：如果你知道当前点附近函数变化得多快，就能更聪明地迈下一步，而不是盲目折半。
- 对平方根问题，更新式可以写成 `guess = guess - ((guess**2 - k) / (2*guess))`。
- 它通常收敛很快，但依赖一个合理初值，也更依赖问题本身的可微结构。
- 和 bisection 不同，Newton 不一定始终保留一个安全区间，所以你更需要关注发散或异常情况。
- 本质上它是在用切线代替曲线，用局部线性模型快速逼近真正的根。

> [!note] What to internalize
> - One-sentence takeaway: Newton-Raphson 的直觉是：如果你知道当前点附近函数变化得多快，就能更聪明地迈下一步，而不是盲目折半。
> - Review anchor: 对平方根问题，更新式可以写成 `guess = guess - ((guess**2 - k) / (2*guess))`。
> - Review anchor: 它通常收敛很快，但依赖一个合理初值，也更依赖问题本身的可微结构。

从做题角度看，只要题目在考“Newton-Raphson：用局部斜率决定修正步长”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：Newton-Raphson 的直觉是：如果你知道当前点附近函数变化得多快，就能更聪明地迈下一步，而不是盲目折半。

### 选择哪种方法，取决于你知道多少结构
算法不是越高级越好，而是要看问题条件是否支持它的前提。
- 如果只知道答案在某个单调区间里，bisection 往往稳健、简单且容易证明正确。
- 如果你还能拿到导数或类似的局部变化信息，Newton-Raphson 常常更快。
- 如果问题几乎没有结构可用，才会退回固定步长或枚举。
- 写数值算法时，速度、正确性、稳健性通常是三角关系，不可能只优化其中一个维度而不付代价。

> [!note] What to internalize
> - One-sentence takeaway: 算法不是越高级越好，而是要看问题条件是否支持它的前提。
> - Review anchor: 如果只知道答案在某个单调区间里，bisection 往往稳健、简单且容易证明正确。
> - Review anchor: 如果你还能拿到导数或类似的局部变化信息，Newton-Raphson 常常更快。

从做题角度看，只要题目在考“选择哪种方法，取决于你知道多少结构”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：算法不是越高级越好，而是要看问题条件是否支持它的前提。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - Recall the approximation method code to find the square root
> - x = 54321
> - epsilon = 0.01
> - num_guesses = 0
> - guess = 0.0
> - increment = 0.0001 # try it with 0.00001
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 用 bisection 近似平方根
> ```python
> x = 0.5
> epsilon = 0.01
> low, high = x, 1.0
> guess = (low + high) / 2
> while abs(guess**2 - x) >= epsilon:
>     if guess**2 < x:
>         low = guess
>     else:
>         high = guess
>     guess = (low + high) / 2
> print(guess)
> ```
> 这个例子说明 bisection 的关键不在公式，而在于每轮都让有效区间减半，同时保证答案没有丢出区间。

> [!example] Newton-Raphson 更新平方根 guess
> ```python
> k = 24
> epsilon = 0.01
> guess = k / 2
> while abs(guess**2 - k) >= epsilon:
>     guess = guess - ((guess**2 - k) / (2 * guess))
> print(guess)
> ```
> Newton 的每一步都更激进，因为它利用了局部斜率信息；这也是它通常更快的原因。

## Exercise log
> [!note] Finger exercise snapshot
> - Official prompt: Assume you are given an integer 0 <= N <= 1000. Write a piece of Python code that uses bisection search to guess N. he code prints two oines: count: with how many guesses it took to find N, and answer: with the vaoue of...
> - What this is really testing: whether you can compress the lecture into one small, high-frequency coding move without needing the slides beside you.
> - Where to revisit if this feels shaky: go back to the first two Core ideas sections in this note, then rerun the official lecture code once with your own input.
> - Follow-on practice path: after this finger exercise, the most natural next stop is Recitation 03.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: this lecture does not have a direct calendar milestone attached, so use it as a consolidation lecture rather than a sprint lecture.
> - Recitation connection: Recitation 03 is the best place to turn the lecture ideas into shorter solved exercises.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec06.pdf|Lecture 06 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec06_code.py|Lecture 06 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex06_sol.pdf|Lecture 06 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec06_transcript.pdf|Lecture 06 transcript]]
- Recitation 3: [[MIT 6.100L-recitations/mit6_100l_rec03.zip|Recitation 03 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 3.4-3.5)

## Review checklist
- [ ] 我能说明 bisection search 成立的核心前提。
- [ ] 我能解释 low / high / guess 在每轮迭代中的更新规则。
- [ ] 我能口头讲出 Newton-Raphson 的更新思想，而不是只背公式。
- [ ] 我能比较 fixed increment、bisection、Newton-Raphson 的优缺点。
- [ ] 我能处理输入小于 1 时平方根区间的特殊情况。
- [ ] 我能围绕“Bisection search：每次砍掉一半不确定区间”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Newton-Raphson：用局部斜率决定修正步长”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：没有保证真实答案一开始就在 `[low, high]` 区间内。
- [ ] 我能说出并避免这个高频误区：把 bisection 机械套用到不满足单调/有序结构的问题上。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 没有保证真实答案一开始就在 `[low, high]` 区间内。
> - 把 bisection 机械套用到不满足单调/有序结构的问题上。
> - 只记 Newton 公式，不理解它为什么可能快、也为什么可能不稳。

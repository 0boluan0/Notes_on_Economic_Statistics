---
aliases:
  - MIT 6.100L Lecture 16
  - 6.100L L16
  - Recursion on Non-Numerics
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 16
---

# Lecture 16: Recursion on Non-Numerics

> [!tip] Hint
> - 我能把递归从数字问题迁移到 string、list 等非数值结构上。
> - 我能说明 memoization 为什么能改善某些递归的重复计算。
> - 我能解释‘递归处理 sequence’时规模是如何减少的。
> - 我能看出一个递归是不是在重复算同一个子问题。
> - 我能围绕本讲的主轴 “结构递归：把 sequence 看成‘头 + 剩余部分’” / “Fibonacci 提醒我们：递归可能有大量重复工作” / “递归能自然表达嵌套结构与分治问题”，不翻 slides 也把整节课重新讲一遍。
> - 我能在 string 或 list 上写出正确的 base case 和 recursive case。
> - 我能解释 Fibonacci 为什么会有重叠子问题。
> - 我能说明 memoization 的作用与成本。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 6.2-6.4
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: 结构递归：把 sequence 看成‘头 + 剩余部分’ / Fibonacci 提醒我们：递归可能有大量重复工作 / 递归能自然表达嵌套结构与分治问题
> - 这讲的关键不是引入新语法，而是把递归思维从 `n-1` 推广到更一般的结构缩小。
> - 一旦你能在 string、list、tree 这类对象上看见递归，你对分治与结构化算法的理解会明显升级。
> - Fibonacci 与 memoization 则提醒我们：递归写法优雅，但不自动保证高效。

## Core ideas
### 结构递归：把 sequence 看成‘头 + 剩余部分’
对字符串和列表做递归时，最自然的切法通常不是减一个数字，而是拿走第一个元素，把剩余部分交给递归处理。
- 例如处理字符串时，可以把问题拆成‘处理首字符’与‘处理剩余子串’。
- 这种写法和循环扫描 sequence 的本质目标相同，只是控制结构换成了递归。
- 如果每次递归都稳定减少一个元素，base case 往往就是空串、空 list 或单元素结构。
- 这让你开始从数据结构的形状而不是数字大小思考规模。

> [!note] What to internalize
> - One-sentence takeaway: 对字符串和列表做递归时，最自然的切法通常不是减一个数字，而是拿走第一个元素，把剩余部分交给递归处理。
> - Review anchor: 例如处理字符串时，可以把问题拆成‘处理首字符’与‘处理剩余子串’。
> - Review anchor: 这种写法和循环扫描 sequence 的本质目标相同，只是控制结构换成了递归。

从做题角度看，只要题目在考“结构递归：把 sequence 看成‘头 + 剩余部分’”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：对字符串和列表做递归时，最自然的切法通常不是减一个数字，而是拿走第一个元素，把剩余部分交给递归处理。

### Fibonacci 提醒我们：递归可能有大量重复工作
某些递归定义虽然非常贴近数学描述，但计算时会反复求解同一个子问题，效率极差。
- 朴素 Fibonacci 会重复计算相同的 `fib(k)` 很多次，这是典型的重叠子问题。
- 一旦子问题会重复出现，memoization 就能通过缓存结果显著减少开销。
- 缓存不是递归的修补，而是承认‘优雅定义’与‘高效执行’有时需要两层设计。
- 这也为后面动态规划和复杂度分析建立了直觉。

> [!note] What to internalize
> - One-sentence takeaway: 某些递归定义虽然非常贴近数学描述，但计算时会反复求解同一个子问题，效率极差。
> - Review anchor: 朴素 Fibonacci 会重复计算相同的 `fib(k)` 很多次，这是典型的重叠子问题。
> - Review anchor: 一旦子问题会重复出现，memoization 就能通过缓存结果显著减少开销。

从做题角度看，只要题目在考“Fibonacci 提醒我们：递归可能有大量重复工作”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：某些递归定义虽然非常贴近数学描述，但计算时会反复求解同一个子问题，效率极差。

### 递归能自然表达嵌套结构与分治问题
像 Towers of Hanoi、嵌套列表处理等问题，递归的优势是能直接照着问题结构写，而不是强行把结构摊平成循环状态机。
- 当一个问题天然由若干更小的同类子问题组成时，递归描述通常最自然。
- 但自然不等于免费：每一次函数调用都要付出栈与时间成本。
- 所以递归程序写完后，最好马上问两个问题：规模是否单调减小？子问题是否会重复？
- 如果这两个问题答不清，程序就算能跑，也很难算稳。

> [!note] What to internalize
> - One-sentence takeaway: 像 Towers of Hanoi、嵌套列表处理等问题，递归的优势是能直接照着问题结构写，而不是强行把结构摊平成循环状态机。
> - Review anchor: 当一个问题天然由若干更小的同类子问题组成时，递归描述通常最自然。
> - Review anchor: 但自然不等于免费：每一次函数调用都要付出栈与时间成本。

从做题角度看，只要题目在考“递归能自然表达嵌套结构与分治问题”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：像 Towers of Hanoi、嵌套列表处理等问题，递归的优势是能直接照着问题结构写，而不是强行把结构摊平成循环状态机。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - Fibonacci with a dictionary
> - print(fib_recur(34))
> - print(fib_efficient(34, d))
> - make a score of x-1 then add 1
> - and make a score of x-2 then add 2
> - and make a score of x-3 then add 3
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 递归反转字符串
> ```python
> def reverse_string(s):
>     if s == "":
>         return ""
>     return reverse_string(s[1:]) + s[0]
>
> print(reverse_string("python"))
> ```
> 问题规模通过 `s[1:]` 递减，base case 是空字符串。这是典型的非数值递归。

> [!example] 用 memoization 优化 Fibonacci
> ```python
> def fib(n, memo=None):
>     if memo is None:
>         memo = {}
>     if n in memo:
>         return memo[n]
>     if n <= 1:
>         return n
>     memo[n] = fib(n - 1, memo) + fib(n - 2, memo)
>     return memo[n]
>
> print(fib(10))
> ```
> 缓存把重叠子问题的重复求值消掉，说明递归写法与性能优化可以同时存在。

## Exercise log
> [!note] Finger exercise snapshot
> - Official prompt: Impoement the function that meets the specification beoow.: def flatten(L): """ L: a list Returns a copy of L, which is a flattened version of L """ # Your code here # Examples: L =...
> - What this is really testing: whether you can compress the lecture into one small, high-frequency coding move without needing the slides beside you.
> - Where to revisit if this feels shaky: go back to the first two Core ideas sections in this note, then rerun the official lecture code once with your own input.
> - Follow-on practice path: after this finger exercise, the most natural next stop is Recitation 08.
> - Homework bridge: this lecture is directly connected to the following calendar milestones: PS 4 out, PS 3 due.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: calendar shows this lecture touching the following milestones: PS 4 out；PS 3 due。读完本讲后，不应只会解释概念，还应能把它们搬到更长的程序里。
> - Recitation connection: Recitation 08 is the best place to turn the lecture ideas into shorter solved exercises.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec16.pdf|Lecture 16 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec16_code.py|Lecture 16 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex16_sol.pdf|Lecture 16 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec16_transcript.pdf|Lecture 16 transcript]]
- Recitation 8: [[MIT 6.100L-recitations/mit6_100l_rec08.zip|Recitation 08 materials]]
- PS 4 out: [[MIT 6.100L-problem-sets/mit6_100l_ps4.pdf|PS4 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps4_code.zip|PS4 starter code]]
- PS 3 due: [[MIT 6.100L-problem-sets/mit6_100l_ps3.pdf|PS3 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps3_code.zip|PS3 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 6.2-6.4)

## Review checklist
- [ ] 我能在 string 或 list 上写出正确的 base case 和 recursive case。
- [ ] 我能解释 Fibonacci 为什么会有重叠子问题。
- [ ] 我能说明 memoization 的作用与成本。
- [ ] 我能判断一个递归问题是不是天然适合结构递归。
- [ ] 我能比较结构递归与循环扫描 sequence 的异同。
- [ ] 我能围绕“结构递归：把 sequence 看成‘头 + 剩余部分’”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Fibonacci 提醒我们：递归可能有大量重复工作”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：只会写 `n-1` 型递归，看不出 sequence 也能递归缩小。
- [ ] 我能说出并避免这个高频误区：写出优雅但指数级重复计算的递归，却没有意识到性能问题。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 只会写 `n-1` 型递归，看不出 sequence 也能递归缩小。
> - 写出优雅但指数级重复计算的递归，却没有意识到性能问题。
> - 在需要缓存的场景里反复重算相同子问题。

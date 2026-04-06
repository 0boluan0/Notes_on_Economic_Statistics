---
aliases:
  - MIT 6.100L Lecture 11
  - 6.100L L11
  - Aliasing, Cloning
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 11
---

# Lecture 11: Aliasing, Cloning

> [!tip] Hint
> - 我能解释 aliasing 的本质：两个名字指向同一个可变对象。
> - 我能区分浅复制、显式克隆与简单赋值。
> - 我能说明 list comprehension 为什么既像循环又像构造器。
> - 我能判断什么时候共享对象是特性，什么时候是 bug 来源。
> - 我能围绕本讲的主轴 “Aliasing：多个名字共享同一个对象” / “Cloning：需要独立状态时就显式复制” / “List comprehension：更紧凑地构造新 list”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释 aliasing 为什么常见于可变对象。
> - 我能判断某个场景是否需要 clone。
> - 我能说明浅复制对嵌套对象的局限。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 5.3-5.5
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Aliasing：多个名字共享同一个对象 / Cloning：需要独立状态时就显式复制 / List comprehension：更紧凑地构造新 list
> - Lecture 10 讲 mutability，这一讲把最麻烦的后果挑明：别名和共享引用。
> - 很多‘明明没改它，为什么它也变了’的 bug，本质上都是 aliasing 问题。
> - list comprehension 则提供了一种更声明式的构造新 list 的方式，常常能避开不必要的共享副作用。

## Core ideas
### Aliasing：多个名字共享同一个对象
赋值 `b = a` 对不可变对象常常没问题，但对 list 这样的可变对象，它意味着你现在有了两个入口能改同一个东西。
- aliasing 不是错误本身；当你确实想共享状态时，它很有用。
- 真正危险的是你以为自己拿到的是副本，实际上只是拿到了同一对象的另一个名字。
- 调试时若怀疑 aliasing，可以同时 print 两个变量并观察 mutation 后是否一起变化。
- 理解 aliasing 的最稳办法，是把注意力从‘变量名’转到‘对象身份’。

> [!note] What to internalize
> - One-sentence takeaway: 赋值 `b = a` 对不可变对象常常没问题，但对 list 这样的可变对象，它意味着你现在有了两个入口能改同一个东西。
> - Review anchor: aliasing 不是错误本身；当你确实想共享状态时，它很有用。
> - Review anchor: 真正危险的是你以为自己拿到的是副本，实际上只是拿到了同一对象的另一个名字。

从做题角度看，只要题目在考“Aliasing：多个名字共享同一个对象”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：赋值 `b = a` 对不可变对象常常没问题，但对 list 这样的可变对象，它意味着你现在有了两个入口能改同一个东西。

### Cloning：需要独立状态时就显式复制
当后续操作不该影响原对象时，复制不是锦上添花，而是语义的一部分。
- 切片 `[:]`、某些构造器或复制函数都能产生新 list，但要清楚是浅复制还是更深层复制。
- 如果 list 里嵌套的还是可变对象，浅复制只复制最外层结构，不会自动复制内部对象。
- ‘我要不要 clone’ 本质上是在做状态所有权判断：后面谁有权修改这份数据？
- 把复制写得显式，比事后靠注释提醒‘别改这个 list’更可靠。

> [!note] What to internalize
> - One-sentence takeaway: 当后续操作不该影响原对象时，复制不是锦上添花，而是语义的一部分。
> - Review anchor: 切片 `[:]`、某些构造器或复制函数都能产生新 list，但要清楚是浅复制还是更深层复制。
> - Review anchor: 如果 list 里嵌套的还是可变对象，浅复制只复制最外层结构，不会自动复制内部对象。

从做题角度看，只要题目在考“Cloning：需要独立状态时就显式复制”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：当后续操作不该影响原对象时，复制不是锦上添花，而是语义的一部分。

### List comprehension：更紧凑地构造新 list
list comprehension 常常是把‘扫描 + 条件 + 构造新序列’压缩成一个表达式，适合表达数据转换。
- 它最适合那些‘从旧 list 生成新 list’的任务，而不是包含大量副作用的复杂流程。
- 在语义上，它更像‘声明我要什么样的新结果’，而不是一步步手工 append。
- 这使得代码更贴近问题本身，也更容易看出没有原地修改旧 list。
- 如果 comprehension 长到读不懂，就该拆回普通循环。

> [!note] What to internalize
> - One-sentence takeaway: list comprehension 常常是把‘扫描 + 条件 + 构造新序列’压缩成一个表达式，适合表达数据转换。
> - Review anchor: 它最适合那些‘从旧 list 生成新 list’的任务，而不是包含大量副作用的复杂流程。
> - Review anchor: 在语义上，它更像‘声明我要什么样的新结果’，而不是一步步手工 append。

从做题角度看，只要题目在考“List comprehension：更紧凑地构造新 list”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：list comprehension 常常是把‘扫描 + 条件 + 构造新序列’压缩成一个表达式，适合表达数据转换。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - remove from a list
> - L = [2,1,3,6,3,7,0]
> - L.remove(2)
> - L.remove(3)
> - del(L[1])
> - print(L.pop())
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] Aliasing 带来的联动修改
> ```python
> a = [1, 2, 3]
> b = a
> a[0] = 99
> print(a)
> print(b)
> ```
> 两个名字共享同一个 list，所以 `a` 的原地修改会直接反映在 `b` 上。

> [!example] 用 comprehension 构造新 list
> ```python
> nums = [1, 2, 3, 4]
> squares = [n * n for n in nums if n % 2 == 0]
> print(squares)
> ```
> 这里得到的是新 list，不会修改原来的 `nums`，因此特别适合表达筛选+变换。

## Exercise log
> [!note] Finger exercise snapshot
> - Official prompt: Impoement the function that meets the specification beoow.: def remove_and_sort(Lin, k): """ Lin is a list of ints k is an int >= 0 Mutates Lin to remove the first k elements in Lin and then sorts the remaining elements...
> - What this is really testing: whether you can compress the lecture into one small, high-frequency coding move without needing the slides beside you.
> - Where to revisit if this feels shaky: go back to the first two Core ideas sections in this note, then rerun the official lecture code once with your own input.
> - Follow-on practice path: after this finger exercise, the most natural next stop is Recitation 06.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: this lecture does not have a direct calendar milestone attached, so use it as a consolidation lecture rather than a sprint lecture.
> - Recitation connection: Recitation 06 is the best place to turn the lecture ideas into shorter solved exercises.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec11.pdf|Lecture 11 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec11_code.py|Lecture 11 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex11_sol.pdf|Lecture 11 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec11_transcript.pdf|Lecture 11 transcript]]
- Recitation 6: [[MIT 6.100L-recitations/mit6_100l_rec06.zip|Recitation 06 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 5.3-5.5)

## Review checklist
- [ ] 我能解释 aliasing 为什么常见于可变对象。
- [ ] 我能判断某个场景是否需要 clone。
- [ ] 我能说明浅复制对嵌套对象的局限。
- [ ] 我能比较普通循环构造新 list 与 list comprehension 的优缺点。
- [ ] 我能自己举出一个 aliasing bug 的例子。
- [ ] 我能围绕“Aliasing：多个名字共享同一个对象”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Cloning：需要独立状态时就显式复制”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：把 `b = a` 当成复制，而不是 aliasing。
- [ ] 我能说出并避免这个高频误区：需要独立状态时没有 clone，后面一改全跟着变。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 把 `b = a` 当成复制，而不是 aliasing。
> - 需要独立状态时没有 clone，后面一改全跟着变。
> - 为了写得短强行上 comprehension，反而牺牲可读性。

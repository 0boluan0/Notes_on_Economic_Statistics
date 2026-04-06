---
aliases:
  - MIT 6.100L Lecture 04
  - 6.100L L04
  - Loops over Strings, Guess-and-Check, and Binary
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 04
---

# Lecture 04: Loops over Strings, Guess-and-Check, and Binary

> [!tip] Hint
> - 我能说明为什么字符串处理常常依赖‘逐字符扫描 + 条件判断 + 累计结果’。
> - 我能解释 guess-and-check 适合什么类型的问题，以及它为什么通常比较慢。
> - 我能用程序把十进制数转成二进制，并理解过程中在累积什么。
> - 我能看出一个枚举算法的搜索空间是怎么定义出来的。
> - 我能围绕本讲的主轴 “循环处理字符串：scan、test、accumulate” / “Guess-and-check：先定义候选空间，再逐个验证” / “Binary representation 是‘重复拆分 + 反向拼接’”，不翻 slides 也把整节课重新讲一遍。
> - 我能写一个按字符扫描字符串并累计结果的程序。
> - 我能解释 guess-and-check 的正确性来自哪里，以及为什么它慢。
> - 我能从头写出十进制转二进制的循环。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 3.1, Ch 3.3
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: 循环处理字符串：scan、test、accumulate / Guess-and-check：先定义候选空间，再逐个验证 / Binary representation 是‘重复拆分 + 反向拼接’
> - 这讲把 iteration 真正用起来：不是为循环而循环，而是拿来扫描 sequence、枚举候选解、逐步构造表示。
> - 它是搜索思想的第一次正式登场，后面 bisection search 和复杂度分析都从这里生长出来。
> - Lecture code 中的字符串扫描、perfect square、cube root、binary conversion 都是同一种思维的不同外观。

## Core ideas
### 循环处理字符串：scan、test、accumulate
字符串题的通用套路是：按固定顺序扫描字符，对每个字符做测试，再把结果累积起来。
- 你可以按 index 扫，也可以直接按字符扫；如果不用位置，直接 `for char in s` 往往更清楚。
- 计数、筛选、查找、构造新字符串，本质上都是扫描过程中的不同 accumulator 设计。
- 写字符串循环时，要提前决定：我关心的是字符值本身，还是字符所在的位置？
- 像 cheerleader、unique letters 这样的例子都在训练你从 sequence 中提取结构化信息。

> [!note] What to internalize
> - One-sentence takeaway: 字符串题的通用套路是：按固定顺序扫描字符，对每个字符做测试，再把结果累积起来。
> - Review anchor: 你可以按 index 扫，也可以直接按字符扫；如果不用位置，直接 `for char in s` 往往更清楚。
> - Review anchor: 计数、筛选、查找、构造新字符串，本质上都是扫描过程中的不同 accumulator 设计。

从做题角度看，只要题目在考“循环处理字符串：scan、test、accumulate”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：字符串题的通用套路是：按固定顺序扫描字符，对每个字符做测试，再把结果累积起来。

### Guess-and-check：先定义候选空间，再逐个验证
guess-and-check 的核心不是‘瞎猜’，而是系统地枚举所有可能候选，然后用条件过滤掉不合格的值。
- 找 perfect square / cube root 时，候选空间通常是 `0` 到 `abs(x)` 这样的整数区间。
- 关键步骤是先问‘解可能出现在哪个空间里’，再问‘如何快速检查一个 guess 是否合格’。
- 这类方法的优点是容易写、容易保证正确性；缺点是搜索空间一大就会慢。
- 如果问题有额外结构可用，就应该考虑比纯枚举更聪明的搜索方法。

> [!note] What to internalize
> - One-sentence takeaway: guess-and-check 的核心不是‘瞎猜’，而是系统地枚举所有可能候选，然后用条件过滤掉不合格的值。
> - Review anchor: 找 perfect square / cube root 时，候选空间通常是 `0` 到 `abs(x)` 这样的整数区间。
> - Review anchor: 关键步骤是先问‘解可能出现在哪个空间里’，再问‘如何快速检查一个 guess 是否合格’。

从做题角度看，只要题目在考“Guess-and-check：先定义候选空间，再逐个验证”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：guess-and-check 的核心不是‘瞎猜’，而是系统地枚举所有可能候选，然后用条件过滤掉不合格的值。

### Binary representation 是‘重复拆分 + 反向拼接’
把十进制转成二进制的程序很适合训练你对循环和字符串累积的理解：每一轮都通过 `% 2` 取最低位，再通过 `// 2` 缩小问题。
- 最低位由 `num % 2` 决定，所以新得到的 bit 往往要加到结果字符串的前面。
- 整数除法 `num // 2` 让规模递减，因此算法天然会终止。
- 如果输入可能为负数，常见策略是先记录符号，再对绝对值做主算法，最后补回负号。
- 这个模式和很多 later topics 很像：把大问题拆成一串规模不断缩小的小问题。

> [!note] What to internalize
> - One-sentence takeaway: 把十进制转成二进制的程序很适合训练你对循环和字符串累积的理解：每一轮都通过 `% 2` 取最低位，再通过 `// 2` 缩小问题。
> - Review anchor: 最低位由 `num % 2` 决定，所以新得到的 bit 往往要加到结果字符串的前面。
> - Review anchor: 整数除法 `num // 2` 让规模递减，因此算法天然会终止。

从做题角度看，只要题目在考“Binary representation 是‘重复拆分 + 反向拼接’”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：把十进制转成二进制的程序很适合训练你对循环和字符串累积的理解：每一轮都通过 `% 2` 取最低位，再通过 `// 2` 缩小问题。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - mysum = 0
> - for i in range(5, 11, 2)
> - mysum += i
> - if mysum == 5
> - break
> - mysum += 1
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 扫描字符串并统计元音
> ```python
> s = "demo loops - fruit loops"
> count = 0
> for char in s:
>     if char in "aeiou":
>         count += 1
> print(count)
> ```
> 这是最典型的 scan + test + accumulate。任何需要从 sequence 中提取结构的信息，都可以从这个骨架变形出来。

> [!example] 把正整数转成二进制字符串
> ```python
> num = 13
> result = ""
> while num > 0:
>     result = str(num % 2) + result
>     num = num // 2
> print(result)
> ```
> `result` 是字符串 accumulator，`num` 是不断缩小的问题规模。这个例子把循环、算术和字符串拼接绑在了一起。

## Exercise log
> [!note] Finger exercise snapshot
> - Official prompt: Assume you are given a positive integer variaboe named N . Write a piece of Python code that finds the cube root of N . he code prints the cube root if N is a perfect cube or it prints error if N is not a perfect cube....
> - What this is really testing: whether you can compress the lecture into one small, high-frequency coding move without needing the slides beside you.
> - Where to revisit if this feels shaky: go back to the first two Core ideas sections in this note, then rerun the official lecture code once with your own input.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: this lecture does not have a direct calendar milestone attached, so use it as a consolidation lecture rather than a sprint lecture.
> - Recitation connection: there is no recitation attached to this lecture week in the official calendar.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec04.pdf|Lecture 04 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec04_code.py|Lecture 04 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex04_sol.pdf|Lecture 04 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec04_transcript.pdf|Lecture 04 transcript]]
- Recitation: none attached to this lecture week
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 3.1, Ch 3.3)

## Review checklist
- [ ] 我能写一个按字符扫描字符串并累计结果的程序。
- [ ] 我能解释 guess-and-check 的正确性来自哪里，以及为什么它慢。
- [ ] 我能从头写出十进制转二进制的循环。
- [ ] 我能比较‘按 index 扫’和‘按 character 扫’各自何时更合适。
- [ ] 我能看出一个问题的 candidate space 应该怎么定义。
- [ ] 我能围绕“循环处理字符串：scan、test、accumulate”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Guess-and-check：先定义候选空间，再逐个验证”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：明明不需要 index，却硬写 `range(len(s))`，让代码变得更绕。
- [ ] 我能说出并避免这个高频误区：把 guess-and-check 写成随意尝试，而不是系统枚举候选空间。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 明明不需要 index，却硬写 `range(len(s))`，让代码变得更绕。
> - 把 guess-and-check 写成随意尝试，而不是系统枚举候选空间。
> - 做二进制转换时把新 bit 加在字符串后面，导致结果顺序反了。

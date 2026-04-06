---
aliases:
  - MIT 6.100L Lecture 19
  - 6.100L L19
  - Inheritance
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 19
---

# Lecture 19: Inheritance

> [!tip] Hint
> - 我能解释 inheritance 解决的是什么复用问题。
> - 我能区分父类、子类、继承、override。
> - 我能说明为什么子类应当扩展或特化父类，而不是复制父类。
> - 我能判断一个层次结构是否真的表达了 is-a 关系。
> - 我能围绕本讲的主轴 “Inheritance：在已有抽象上复用与特化” / “Override 与 inherited methods” / “建模时先问是不是合理的 is-a 关系”，不翻 slides 也把整节课重新讲一遍。
> - 我能解释继承想复用的到底是什么。
> - 我能判断一个设计是否真的满足 is-a 关系。
> - 我能把本讲最关键的代码模式手写出来，并解释每一步为什么这样写。

> [!info] Lecture map
> - Readings: Ch 10.2
> - Recommended use order: read the Hint first, reconstruct the lecture from memory, then study the Core ideas, then run the official code, and only after that open the linked exercises.
> - Main threads in this lecture: Inheritance：在已有抽象上复用与特化 / Override 与 inherited methods / 建模时先问是不是合理的 is-a 关系
> - 前面两讲已经会定义单个类，这一讲讨论如何组织一组相关类。
> - inheritance 的目标是共享共性、保留差异，让重复逻辑集中在父类，特殊行为留给子类。
> - 理解继承，也是在练习接口替换与层次建模，为更复杂系统设计打基础。

## Core ideas
### Inheritance：在已有抽象上复用与特化
如果多个类共享大量相同行为或状态约束，就可以把公共部分提到父类，再让子类表达差异。
- 父类负责定义共性，子类负责扩展或改写局部行为。
- 这比复制粘贴多个相似类更稳，因为公共逻辑只维护一份。
- 继承的前提不是‘代码长得像’，而是‘语义上存在稳定共性’。
- 一旦层次设计合理，使用者也能更容易理解不同类之间的关系。

> [!note] What to internalize
> - One-sentence takeaway: 如果多个类共享大量相同行为或状态约束，就可以把公共部分提到父类，再让子类表达差异。
> - Review anchor: 父类负责定义共性，子类负责扩展或改写局部行为。
> - Review anchor: 这比复制粘贴多个相似类更稳，因为公共逻辑只维护一份。

从做题角度看，只要题目在考“Inheritance：在已有抽象上复用与特化”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：如果多个类共享大量相同行为或状态约束，就可以把公共部分提到父类，再让子类表达差异。

### Override 与 inherited methods
子类可以直接继承父类方法，也可以 override 某些方法来表达更具体的行为。
- override 的价值在于保持接口不变、语义更具体，而不是随意改掉父类约定。
- 子类仍然可以复用父类未被改写的方法，这就是继承复用的核心收益。
- 阅读继承代码时，要同时关注‘这个方法是本类定义的，还是从父类来的’。
- override 如果破坏了父类原本的使用预期，会让整个层次变得难以推理。

> [!note] What to internalize
> - One-sentence takeaway: 子类可以直接继承父类方法，也可以 override 某些方法来表达更具体的行为。
> - Review anchor: override 的价值在于保持接口不变、语义更具体，而不是随意改掉父类约定。
> - Review anchor: 子类仍然可以复用父类未被改写的方法，这就是继承复用的核心收益。

从做题角度看，只要题目在考“Override 与 inherited methods”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：子类可以直接继承父类方法，也可以 override 某些方法来表达更具体的行为。

### 建模时先问是不是合理的 is-a 关系
不是所有共享代码的场景都该用继承。有时组合更合适；继承应保留给真正的层次语义。
- 如果子类只是想借用一点实现，而不是语义上属于父类，那么继承往往是坏味道。
- 好的继承层次应该满足：在大多数使用场景里，子类都能被当作父类看待。
- 过深层次会让代码阅读成本上升，因为行为来源变得分散。
- 因此继承的价值不在于省代码行数，而在于表达清楚的语义关系。

> [!note] What to internalize
> - One-sentence takeaway: 不是所有共享代码的场景都该用继承。有时组合更合适；继承应保留给真正的层次语义。
> - Review anchor: 如果子类只是想借用一点实现，而不是语义上属于父类，那么继承往往是坏味道。
> - Review anchor: 好的继承层次应该满足：在大多数使用场景里，子类都能被当作父类看待。

从做题角度看，只要题目在考“建模时先问是不是合理的 is-a 关系”相关的表示、判断、控制流或抽象边界，就不应该只回忆表面语法，而要先回到这一节的核心句：不是所有共享代码的场景都该用继承。有时组合更合适；继承应保留给真正的层次语义。

## Code patterns from lecture
> [!note] What the official code is trying to teach
> - The official lecture code is worth reading as a notebook of small patterns, not just as a file to run once.
> - Best workflow: predict output first, then run the code, then rewrite the pattern in your own words or with slightly changed values.
> - Animal abstract data type
> - #default parameters with methods
> - print(a)
> - print(b)
> - print(a.age)
> - print(a.get_age())
> - When a code pattern feels too easy, change the input, break one line on purpose, and explain why the behavior changes.

## Worked examples
> [!example] 子类继承父类的通用行为
> ```python
> class Animal:
>     def speak(self):
>         return "..."
>
> class Dog(Animal):
>     pass
>
> print(Dog().speak())
> ```
> Dog 没有自己实现 `speak`，但可以直接继承父类版本。这说明继承先提供的是复用。

> [!example] override 父类方法
> ```python
> class Animal:
>     def speak(self):
>         return "..."
>
> class Dog(Animal):
>     def speak(self):
>         return "woof"
>
> print(Dog().speak())
> ```
> 接口还是 `speak()`，但子类提供了更具体的语义实现。

## Exercise log
> [!note] Finger exercise snapshot
> - Official prompt: In this problem, you will implement two classes according to the specification below: one Container class and one Stack class (a subclass of Container ). Our Container class will initialize an empty list. The two...
> - Official solution sketch:
> ```python
> class Container(object):
> def __init__(self):
> self.myList = []
> def size(self):
> return len(self.myList)
> def add(self, elem):
> ```
> - What this is really testing: whether you can compress the lecture into one small, high-frequency coding move without needing the slides beside you.
> - Where to revisit if this feels shaky: go back to the first two Core ideas sections in this note, then rerun the official lecture code once with your own input.
> - Follow-on practice path: after this finger exercise, the most natural next stop is Recitation 09.

## From lecture to recitation and homework
> [!abstract] How this lecture shows up in practice
> - Problem-set connection: this lecture does not have a direct calendar milestone attached, so use it as a consolidation lecture rather than a sprint lecture.
> - Recitation connection: Recitation 09 is the best place to turn the lecture ideas into shorter solved exercises.
> - Suggested workflow: read this note once, run the lecture code, solve the smallest official exercise without peeking, then open the linked recitation or problem set materials.
> - If you can explain the note but still cannot start the homework, the gap is usually not theory but translation: you need one more pass through the worked examples and lecture code.

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec19.pdf|Lecture 19 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec19_code.py|Lecture 19 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex19_sol.pdf|Lecture 19 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec19_transcript.pdf|Lecture 19 transcript]]
- Recitation 9: [[MIT 6.100L-recitations/mit6_100l_rec09.zip|Recitation 09 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 10.2)

## Review checklist
- [ ] 我能解释继承想复用的到底是什么。
- [ ] 我能区分父类、子类、继承、override。
- [ ] 我能判断一个设计是否真的满足 is-a 关系。
- [ ] 我能说明为什么复制粘贴多个相似类不如合理继承。
- [ ] 我能看懂某个方法到底来自哪一层类。
- [ ] 我能围绕“Inheritance：在已有抽象上复用与特化”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能围绕“Override 与 inherited methods”自己写出一个最小例子，并解释为什么这个例子能体现本节重点。
- [ ] 我能说出并避免这个高频误区：把继承当成纯粹省代码工具，而不是层次建模工具。
- [ ] 我能说出并避免这个高频误区：override 时破坏父类原有接口预期。
- [ ] 我能不看 slides，只看题面就判断这题主要在考本讲的哪一个知识点。

> [!warning] Common mistakes
> - 把继承当成纯粹省代码工具，而不是层次建模工具。
> - override 时破坏父类原有接口预期。
> - 明明是 has-a 关系，却硬写成 is-a 关系。

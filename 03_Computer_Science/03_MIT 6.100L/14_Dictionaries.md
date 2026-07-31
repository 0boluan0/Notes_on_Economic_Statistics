---
aliases:
  - MIT 6.100L Lecture 14
  - 6.100L L14
  - Dictionaries
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 14
---

# Lecture 14: Dictionaries

> [!tip] Hint
> - 这节课不是直接上 dict 语法，而是先证明“用多个 list 存学生信息会越来越别扭”。
> - 老师先演示 name list + grade list，再演示 list of lists，都是为了说明 lookup 很笨重。
> - dict 的核心不是“花括号”，而是 mapping：用 key 直接到 value。
> - 列表按位置找信息，字典按 key 找信息；这就是本讲的结构性变化。
> - `grades['John']` 之所以自然，是因为 key 本身承载了查找语义，不再靠平行索引硬对齐。
> - 创建、访问、添加、更新、删除 entry 的语法都围绕 key:value 展开。
> - `in` 对 dict 默认查的是 key，不是 value，这是课堂里特意强调的点。
> - `keys()`、`values()`、`items()` 是三种不同的观察角度，后面写循环时很常用。
> - 词频统计例子是本讲真正的大例子，它把 dict 当成“计数器”和“聚合器”来用。
> - 听完这节课，你应该把 dict 看成新的数据组织方式，而不只是又多了一堆 API。
> <!-- bilingual-en:start -->
> - The lecture does not begin with dictionary syntax. It first demonstrates why several lists become awkward for storing student information.
> - A name list paired with a grade list, followed by a list of lists, exposes how cumbersome lookup becomes.
> - The central idea of a dictionary is not braces but a mapping from a key directly to a value.
> - Lists retrieve information by position; dictionaries retrieve it by key. That is the structural change introduced here.
> - `grades['John']` is natural because the key itself carries the lookup meaning instead of relying on aligned indexes.
> - Creating, accessing, adding, updating, and deleting entries all revolve around key–value pairs.
> - Membership with `in` checks dictionary keys by default, not values.
> - `keys()`, `values()`, and `items()` provide three distinct views that become useful in loops.
> - The main example uses a dictionary as a counter and aggregator for word frequencies.
> - By the end, you should see a dictionary as a new way to organize data, not merely another collection of APIs.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 先故意用 list 存成绩，暴露结构缺陷
<!-- bilingual-en:start -->
*1. Exposing Structural Problems by Storing Grades in Lists*
<!-- bilingual-en:end -->
Lecture 14 的开场不是定义 dictionary，而是先把旧方法走到难受的地方。

老师继续用熟悉的学生成绩例子：

- 一个 list 存学生名字
- 一个 list 存对应成绩
- 约定相同 index 代表同一个学生

这在数据很小时还能勉强工作，但一旦你想：

- 查某个学生成绩
- 增加更多字段
- 保证不同列表一直同步

就会变得非常脆弱。
<!-- bilingual-en:start -->
Lecture 14 begins by pushing the old representation until its weakness becomes obvious. One list stores names, another stores grades, and equal indexes are assumed to identify the same student. Small data may tolerate this convention, but lookup, additional fields, and keeping several lists synchronized quickly make it fragile.
<!-- bilingual-en:end -->

### 2. list of lists 也能做，但仍然很笨重
<!-- bilingual-en:start -->
*2. A List of Lists Still Produces Cumbersome Lookup*
<!-- bilingual-en:end -->
为了说明问题不是“列表数量太多”，老师又展示了另一种老办法：

- 每个学生是一条子列表
- 子列表里再装 ps、mq 等信息

这比平行列表更集中一些，但查找时仍然很绕：

- 先找到学生
- 再找到某类成绩
- 再拿到对应值

课堂这里想让你真正感受到：

> [!note]
> 当数据天然就是“名字对应信息”“键对应值”的关系时，list 不是最合适的抽象。
> <!-- bilingual-en:start -->
> When data naturally express “a name maps to information” or “a key maps to a value,” a list is not the most appropriate abstraction.
> <!-- bilingual-en:end -->
> <!-- bilingual-en:start -->
> Putting each student in a sublist centralizes the fields, but lookup still requires finding the student, finding the grade category, and then extracting the value. The problem is therefore not merely the number of lists; it is a mismatch between the data relationship and the abstraction.
> <!-- bilingual-en:end -->

### 3. dictionary 的动机：直接从 key 找 value
<!-- bilingual-en:start -->
*3. The Motivation for Dictionaries: Go Directly from Key to Value*
<!-- bilingual-en:end -->
证明完旧方法很别扭之后，老师才正式引出 dictionary。

dictionary 的本质是 **mapping**：

- key -> value

这意味着：

- 你不再依赖位置
- 你直接用有意义的 key 去访问信息

比如：

```python
grades = {'Ana': 'B', 'Matt': 'A', 'John': 'B'}
```

这里 `'John'` 自身就是查找入口，而不需要先找 John 在第几个位置。
<!-- bilingual-en:start -->
After exposing the awkward list representations, the instructor introduces a dictionary as a mapping from key to value. Information is no longer reached through position; a meaningful key provides direct access. In `grades`, `'John'` itself is the lookup handle, so no positional search is needed first.
<!-- bilingual-en:end -->

### 4. dict 的核心心智模型：entry，不是序列
<!-- bilingual-en:start -->
*4. The Dictionary Mental Model: Entries, Not Sequence Positions*
<!-- bilingual-en:end -->
老师在这里其实是把大家从 sequence 思维切到 mapping 思维。

list / tuple 的基本问题是：

- 第几个元素是什么

dict 的基本问题则变成：

- 某个 key 对应什么 value

所以写字典时要开始习惯：

- 关注 key 是否存在
- 关注 key 和 value 的对应关系
- 不再把“第几个位置”当作主要组织方式
<!-- bilingual-en:start -->
A sequence asks which value occupies a particular position. A dictionary asks which value corresponds to a particular key. Working with dictionaries therefore means checking whether keys exist and reasoning about key–value relationships rather than using position as the main organizing principle.
<!-- bilingual-en:end -->

### 5. 基本操作：创建、访问、添加、修改、删除
<!-- bilingual-en:start -->
*5. Basic Operations: Create, Access, Add, Update, and Delete*
<!-- bilingual-en:end -->
接着老师依次走过最常见的字典操作。
<!-- bilingual-en:start -->
The instructor then demonstrates the most common dictionary operations.
<!-- bilingual-en:end -->

创建：

```python
d = {}
grades = {'Ana': 'B', 'Matt': 'A'}
```

访问：

```python
grades['John']
```

新增或修改：

```python
grades['Grace'] = 'A'
grades['Grace'] = 'C'
```

删除：

```python
del(grades['Ana'])
```

这里课堂的重点不是 API 数量，而是你要真正接受：

- dict 的更新几乎总围绕 key 展开
<!-- bilingual-en:start -->
The individual APIs matter less than the common model: dictionary access and mutation are organized around keys.
<!-- bilingual-en:end -->

### 6. membership test：`in` 默认查的是 key
<!-- bilingual-en:start -->
*6. Membership Testing: `in` Checks Keys by Default*
<!-- bilingual-en:end -->
老师随后强调了一个很容易误判的地方：
<!-- bilingual-en:start -->
The instructor then emphasizes an easy mistake:
<!-- bilingual-en:end -->

```python
'John' in grades
'B' in grades
```

这两句不等价。

对于 dict：

- `in` 默认只查 key
- 不查 value

所以 `'John' in grades` 可能为真，  
但 `'B' in grades` 即使确实是某人的成绩，也仍然可能为假。

这一步很重要，因为它把 dict 从 list 的 membership 直觉里彻底剥离出来。
<!-- bilingual-en:start -->
For a dictionary, `in` tests keys rather than values. Thus, `'John' in grades` may be true while `'B' in grades` is false even when `'B'` appears as a grade. This separates dictionary membership from the membership intuition developed for lists.
<!-- bilingual-en:end -->

### 7. 看 dict 的三个窗口：keys、values、items
<!-- bilingual-en:start -->
*7. Three Views of a Dictionary: Keys, Values, and Items*
<!-- bilingual-en:end -->
老师接下来引入：

- `d.keys()`
- `d.values()`
- `d.items()`

它们分别对应三种观察角度：

- 只看所有 key
- 只看所有 value
- 同时看 `(key, value)` 对

这在写循环时很关键。

例如：

```python
for k, v in grades.items():
    print(k, v)
```

这里你已经不是在遍历一个 sequence 的位置，而是在遍历一个 mapping 的 entry。
<!-- bilingual-en:start -->
`d.keys()`, `d.values()`, and `d.items()` expose keys alone, values alone, or `(key, value)` pairs. A loop over `grades.items()` traverses mapping entries rather than positions in a sequence.
<!-- bilingual-en:end -->

### 8. 第一批 you-try-it：把“查找语义”写进函数
<!-- bilingual-en:start -->
*8. First “You Try It” Exercises: Encoding Lookup Semantics in Functions*
<!-- bilingual-en:end -->
课堂中段几个 you-try-it 都很典型：

- `find_grades(grades, students)`
- `find_in_L(Ld, k)`
- `count_matches(d)`

这些题的共同点是：  
它们不再问“怎么按位置处理”，而是问你能不能围绕 key/value 关系写逻辑。

例如 `find_grades` 的重点就是：

- 按给定学生顺序输出对应成绩

而不是去管这些成绩原本在字典里内部以什么顺序存放。
<!-- bilingual-en:start -->
Exercises such as `find_grades(grades, students)`, `find_in_L(Ld, k)`, and `count_matches(d)` ask you to write logic around key–value relationships rather than positions. `find_grades`, for example, returns grades in the requested student order without relying on any internal dictionary order.
<!-- bilingual-en:end -->

### 9. 嵌套 dict：字典可以组织更复杂的数据
<!-- bilingual-en:start -->
*9. Nested Dictionaries: Organizing Hierarchical Data*
<!-- bilingual-en:end -->
老师随后把难度往前推，给出类似：
<!-- bilingual-en:start -->
The instructor then introduces a nested structure:
<!-- bilingual-en:end -->

```python
my_d = {
    'Ana': {'mq': [10], 'ps': [10, 10]},
    'Fredo': {'ps': [7, 8], 'mq': [8]},
    'Eric': {'mq': [3], 'ps': [0]}
}
```

这时 dict 不再只是“名字到单个值”的简单映射，而是可以承载层级化信息：

- 学生名 -> 某个学生的整套记录
- 记录里再按 `ps`、`mq` 分类

`get_average(data, what)` 这种函数正是在训练你如何穿过这个嵌套结构。
<!-- bilingual-en:start -->
The outer mapping associates a student name with the student's complete record; the inner mapping separates problem sets and quizzes. A function such as `get_average(data, what)` practices traversing those two levels deliberately.
<!-- bilingual-en:end -->

### 10. 词频统计：dict 最典型的大用途之一
<!-- bilingual-en:start -->
*10. Word Frequencies: A Canonical Dictionary Use Case*
<!-- bilingual-en:end -->
本讲最完整的例子是 song lyrics 的 frequency dictionary。

老师的推进顺序很清楚：

1. 先把歌词转小写、按空格拆词
2. 建一个空 dict
3. 扫描每个词
4. 如果词已出现，频数加 1
5. 如果是第一次出现，插入频数 1

代码模式大致是：
<!-- bilingual-en:start -->
The lecture's most complete example builds a frequency dictionary from song lyrics: normalize and split the text, start with an empty dictionary, scan each word, increment an existing count, or initialize a new count to one. The central pattern is:
<!-- bilingual-en:end -->

```python
for w in words_list:
    if w in word_dict:
        word_dict[w] += 1
    else:
        word_dict[w] = 1
```

这个模式非常重要，因为后面看到“统计频率”“聚合计数”“直方图”类任务时，第一反应就应该是 dict。
<!-- bilingual-en:start -->
This pattern generalizes to frequency counting, aggregation, and histogram-like tasks, where a dictionary should become an immediate candidate representation.
<!-- bilingual-en:end -->

### 11. 从 frequency dict 再做分析：最大频数、常见词列表
<!-- bilingual-en:start -->
*11. Analyzing a Frequency Dictionary: Maximum Counts and Common Words*
<!-- bilingual-en:end -->
老师没有停在“建出来”这一步，而是继续往下做分析：

- `find_frequent_word(word_dict)`：找最大频数对应的词
- `occurs_often(word_dict, x)`：不断提取超过阈值的高频词

这让你看到 dict 不只是存结果，它还能成为后续分析的基础结构。

这里课堂顺手也提醒了一个重要问题：

- 有些函数会 mutate 传入的 dict

所以又一次回到了前面 lists 讲过的副作用意识。
<!-- bilingual-en:start -->
The instructor continues beyond construction. `find_frequent_word(word_dict)` finds words with the maximum count, while `occurs_often(word_dict, x)` repeatedly extracts words above a threshold. The dictionary becomes an input to further analysis, and functions that mutate it reintroduce the earlier concern about side effects.
<!-- bilingual-en:end -->

### 12. 这节课真正转变的是“如何组织数据”
<!-- bilingual-en:start -->
*12. The Real Change Is How Data Are Organized*
<!-- bilingual-en:end -->
Lecture 14 的重要性不在于又学了一个容器，而在于课程在逼你改变组织信息的方式。

从这节课开始，你面对问题要学会先问：

- 我的数据更像 sequence 吗
- 还是更像 mapping

如果查找入口天然是名字、ID、标签、单词本身，那 dict 往往比 list 更贴切。
<!-- bilingual-en:start -->
The lecture matters because it changes the representation question. Ask first whether the data are naturally a sequence or a mapping. When names, IDs, labels, or words provide the natural lookup handles, a dictionary usually fits better than a list.
<!-- bilingual-en:end -->

## Exercise log

> [!example] Finger exercise 14
> 官方练习有两问：
> - `keys_with_value(aDict, target)`：返回所有 value 等于 `target` 的 key，并排序。
> - `all_positive(d)`：对 `int -> list` 的字典，返回 value 列表和为正的所有 key，并排序。
> <!-- bilingual-en:start -->
> The official exercise has two parts:
> - `keys_with_value(aDict, target)` returns, in sorted order, every key whose value equals `target`.
> - For an `int -> list` dictionary, `all_positive(d)` returns, in sorted order, every key whose value list has a positive sum.
> <!-- bilingual-en:end -->

这两问都很适合本讲，因为它们分别训练：

- 从 value 反向筛 key
- 遍历 `k, v` 对进行聚合判断

官方解法都不复杂，但能逼你把 dict 思维真正用起来，而不是下意识又回到“按位置找东西”的序列思路。
<!-- bilingual-en:start -->
The two parts practice filtering keys by values and aggregating over `(key, value)` pairs. Their solutions are short, but they force mapping-based reasoning instead of a reflexive return to positional sequence logic.
<!-- bilingual-en:end -->

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec14.pdf|Lecture 14 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec14_code.py|Lecture 14 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex14_sol.pdf|Lecture 14 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec14_transcript.pdf|Lecture 14 transcript]]
- Recitation 7: [[MIT 6.100L-recitations/mit6_100l_rec07.zip|Recitation 07 materials]]
- PS 3 halfway hand-in due: [[MIT 6.100L-problem-sets/mit6_100l_ps3.pdf|PS3 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps3_code.zip|PS3 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 5.7)

## Review checklist
- [ ] 我能解释为什么学生成绩例子更适合 dict 而不是多个 list。
- [ ] 我能说清 sequence 和 mapping 这两种组织数据方式的差别。
- [ ] 我能熟练区分 dict 的创建、访问、更新、删除是如何围绕 key 展开的。
- [ ] 我能解释为什么 `in` 对 dict 默认查 key。
- [ ] 我能区分 `keys()`、`values()`、`items()` 的用途。
- [ ] 我能读懂和编写嵌套 dict 结构上的访问代码。
- [ ] 我能复述词频统计字典的标准构建模式。
- [ ] 我能说明 `find_frequent_word` 这类函数如何基于 frequency dict 再做分析。
- [ ] 我能把 finger exercise 14 和“遍历 entry、筛 key”联系起来。
- [ ] 我能按课堂顺序复述：list 方案为什么差 -> dict 作为 mapping -> dict operations -> frequency dictionary。
<!-- bilingual-en:start -->
- [ ] I can explain why the student-grade example fits a dictionary better than several lists.
- [ ] I can distinguish sequence-based from mapping-based data organization.
- [ ] I can create, access, update, and delete dictionary entries through their keys.
- [ ] I can explain why `in` checks dictionary keys by default.
- [ ] I can distinguish the uses of `keys()`, `values()`, and `items()`.
- [ ] I can read and write access code for nested dictionaries.
- [ ] I can reconstruct the standard pattern for building a word-frequency dictionary.
- [ ] I can explain how functions such as `find_frequent_word` analyze a frequency dictionary.
- [ ] I can connect finger exercise 14 to traversing entries and filtering keys.
- [ ] I can reconstruct the lecture sequence: weaknesses of list representations -> dictionaries as mappings -> dictionary operations -> frequency dictionaries.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 还在用 list 的位置思维理解 dict。
> - 忘记 `in` 对字典查的是 key。
> - 用 dict 时不先想清楚 key 应该是什么，导致结构本身就难查。
> - 写嵌套 dict 访问时没分层，容易把 `data[stud][what]` 这类索引次序写错。
> <!-- bilingual-en:start -->
> - Continuing to interpret a dictionary through list positions.
> - Forgetting that `in` tests dictionary keys.
> - Choosing keys without first deciding the natural lookup meaning, producing a structure that remains difficult to query.
> - Traversing a nested dictionary without separating its levels and consequently reversing access such as `data[stud][what]`.
> <!-- bilingual-en:end -->

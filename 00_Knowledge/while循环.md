---
aliases:
  - "设计为结束的 while 循环需在保持不变量时推进终止度量"
  - A terminating while loop must preserve its invariant while advancing a termination measure
student_os: knowledge-atom
atom_id: CS-PY-CORE-006
atom_set: python-core
atom_type: control-flow
status: source-checked
mastery_state: unassessed
part_of:
  - "[[计算模型、Python 表达式与控制流.canvas]]"
requires:
  - "[[条件分支]]"
related:
  - "[[循环不变量与终止]]"
  - "[[控制转移语句]]"
---

# 设计为结束的 while 循环需在保持不变量时推进终止度量
<!-- bilingual-en:start -->
*A `while` loop intended to terminate must preserve its invariant while advancing a termination measure*
<!-- bilingual-en:end -->

> [!summary] 原子控制流
> Python 在每轮开始前测试 `while` 条件：为真就执行循环体并回到条件，为假就离开循环；第一次测试即为假时，循环体执行零次。语法只规定重复机制，不保证程序会停止。对设计为结束的循环，每条会回到条件的路径都要更新相关状态，在保持不变量的同时让一个良基终止度量严格推进。
> <!-- bilingual-en:start -->
> Python tests a `while` condition before every iteration. If it is true, the body runs and control returns to the condition; if it is false, the loop ends. A false first test therefore executes the body zero times. The syntax specifies repetition but does not guarantee termination. For a loop intended to finish, every path that returns to the condition must update relevant state, preserve the invariant, and strictly advance a well-founded termination measure.
> <!-- bilingual-en:end -->

```python
remaining = 3
while remaining > 0:
    print(remaining)
    remaining -= 1
```

这里循环头可保持 `0 <= remaining <= 3`，终止度量可取 `remaining`。若某个分支在回到条件前没有减少它，循环可能停在同一状态。`continue` 会提前回到下一轮测试，因此尤其要检查它是否绕过必要更新；完整的不变量与终止证明复用 [[循环不变量与终止]]，不在 Python 语法卡中重复展开。
<!-- bilingual-en:start -->
At the loop head, `0 <= remaining <= 3` is an invariant and `remaining` is a termination measure. If some path returns to the guard without decreasing it, the loop may remain in the same state. Because `continue` jumps early to the next test, it deserves special scrutiny for bypassing a required update. The complete proof method is reused from [[循环不变量与终止|Loop invariant and termination]] rather than duplicated in this Python syntax atom.
<!-- bilingual-en:end -->

有意长期运行的事件循环并不承诺自然终止；此时仍需说明它如何响应取消、异常或外部关闭。`break`、`return` 与异常也可能使循环在条件变假之前离开，因此退出后的性质要覆盖真实的所有出口。
<!-- bilingual-en:start -->
An intentionally persistent event loop does not promise natural termination, but it still needs a defined response to cancellation, exceptions, or external shutdown. `break`, `return`, and exceptions may also leave a loop before its condition becomes false, so post-loop reasoning must cover every actual exit.
<!-- bilingual-en:end -->

## 来源与核验

- [[03_Computer_Science/03_MIT 6.100L/MIT 6.100L-OCW-offline-site/static_resources/mit6_100l_f22_lec03.pdf|MIT 6.100L Lecture 3 slides]]：核对先测试、重复测试、状态更新与无限循环示例。
- [Python Language Reference: the while statement](https://docs.python.org/3/reference/compound_stmts.html#the-while-statement)：核对零次执行、`continue` 回测与 `break` 的语言语义。
- [[循环不变量与终止]]：复用初始化、保持、退出推出后置条件以及终止度量的通用证明方法。

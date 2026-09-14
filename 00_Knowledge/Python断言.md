---
aliases:
  - "Python 断言是检查应成立条件并在启用时以 AssertionError 报告失败的调试语句"
  - "Python assertion"
student_os: knowledge-atom
atom_id: CS-EXC-010
atom_type: definition
status: source-checked
part_of:
  - "[[测试、调试、异常与断言.canvas]]"
requires:
  - "[[Python异常]]"
leads_to:
  - "[[断言不替代输入验证]]"
related:
  - "[[异常类型匹配]]"
---

# Python 断言是检查应成立条件并在启用时以 AssertionError 报告失败的调试语句

<!-- bilingual-en:start -->
*A Python assertion is a debugging statement that checks an expected condition and reports failure with AssertionError when enabled*
<!-- bilingual-en:end -->

Python 断言（assertion）用 `assert condition, message` 表达程序员认为此处应成立的条件；消息可省略。在断言启用时，条件为真则继续，条件为假则抛 `AssertionError`，并仅在失败时求值所给消息。它适合暴露内部假设被破坏的位置，而不是另一种无条件终止命令。

<!-- bilingual-en:start -->
A Python assertion uses `assert condition, message` to express a condition the programmer expects to hold at that point; the message is optional. With assertions enabled, a true condition permits continuation, while a false condition raises `AssertionError` and evaluates the supplied message only on failure. It helps expose broken internal assumptions; it is not an unconditional termination command.
<!-- bilingual-en:end -->

```python
try:
    assert 2 + 2 == 5, "internal arithmetic assumption failed"
except AssertionError:
    result = "caught"
# With assertions enabled, result == "caught".
```

`AssertionError` 是普通的可捕获异常；是否继续取决于外层处理。断言也可能被优化设置禁用，见[[断言不替代输入验证]]。若条件或消息的求值本身抛出别的异常，该异常照常传播，不能把每次断言执行的失败都归类成 `AssertionError`。

<!-- bilingual-en:start -->
`AssertionError` is catchable, so continuation depends on outer handling. Optimization settings can also disable assertions; see [[断言不替代输入验证|why assertions cannot replace required validation]]. If evaluating the condition or message itself raises another exception, that exception propagates normally; not every failure while executing an assertion is an `AssertionError`.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[03_Computer_Science/03_MIT 6.100L/MIT 6.100L-slides/mit6_100l_lec13.pdf#page=13|MIT 6.100L Lecture 13，PDF pp. 13–15、22–23]]；[[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf#page=107|Guttag，印刷 p. 90 / PDF p. 107]]：核对调试假设、状态条件与 `AssertionError`；课程“执行停止”须限定为断言启用且异常未被外层处理。
  <!-- bilingual-en:start -->
  These sources support debugging assumptions, state conditions, and `AssertionError`. The course's stopping description requires enabled assertions and no outer handler.
  <!-- bilingual-en:end -->
- [Python Language Reference §7.3, assert statement](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement)：核对条件、消息的求值顺序与 `__debug__` 语义；可捕获示例和表达式自身抛错已执行核对。
  <!-- bilingual-en:start -->
  This section supports condition/message evaluation and `__debug__` semantics. Catchability and exceptions during expression evaluation were checked by execution.
  <!-- bilingual-en:end -->

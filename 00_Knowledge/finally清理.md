---
aliases:
  - "finally 在控制流离开受保护代码时执行清理但不会自动回滚状态"
  - "Cleanup with finally"
student_os: knowledge-atom
atom_id: CS-EXC-006
atom_type: execution-rule
status: source-checked
part_of:
  - "[[测试、调试、异常与断言.canvas]]"
requires:
  - "[[Python异常]]"
related:
  - "[[控制转移语句]]"
  - "[[回溯状态恢复]]"
  - "[[返回值与打印]]"
---

# finally 在控制流离开受保护代码时执行清理但不会自动回滚状态

<!-- bilingual-en:start -->
*finally performs cleanup as control leaves protected code but does not automatically roll back state*
<!-- bilingual-en:end -->

在 Python 正常的控制流语义下，离开 `try` 及其处理路径前会执行对应 `finally`，无论此前是正常完成、抛出异常，还是执行 `return`、`break`、`continue`。若 `finally` 自身正常完成，先前待传播的异常或待完成的控制转移继续进行；清理不等于捕获并消除失败。

<!-- bilingual-en:start -->
Under Python's ordinary control-flow semantics, the corresponding `finally` runs before leaving a `try` and its handling paths, whether they completed normally, raised an exception, or executed `return`, `break`, or `continue`. If `finally` itself completes normally, the pending exception or transfer continues. Cleanup does not mean that the failure has been caught and removed.
<!-- bilingual-en:end -->

```python
events = []
try:
    try:
        events.append("changed")
        raise ValueError("failed")
    finally:
        events.append("cleanup")
except ValueError:
    pass
# events == ["changed", "cleanup"]
```

`finally` 只执行写在其中的动作，不自动撤销列表修改、文件写入或其他副作用；需要撤销时须显式设计恢复。递归回溯中的具体恢复义务见[[回溯状态恢复]]。清理代码若自己抛异常或执行新的跳转，可能覆盖原结果，见[[控制转移语句]]。因此不要在清理分支随意 `return`。

<!-- bilingual-en:start -->
`finally` performs only its explicit actions; it does not undo list changes, file writes, or other effects. Required rollback needs an explicit restoration design; the specific obligations in recursive search are covered by [[回溯状态恢复|backtracking state restoration]]. A new exception or transfer in cleanup can replace the pending outcome; see [[控制转移语句|control-transfer boundaries]]. Avoid casual returns from cleanup clauses.
<!-- bilingual-en:end -->

这里的保证不是进程生命周期的绝对保证：例如 `os._exit` 直接退出而不运行清理处理器。它也不保证尚未执行完的清理代码必然成功。

<!-- bilingual-en:start -->
This is not an absolute process-lifetime guarantee: for example, `os._exit` exits without running cleanup handlers. Nor does it guarantee that cleanup itself will succeed.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[03_Computer_Science/03_MIT 6.100L/MIT 6.100L-slides/mit6_100l_lec13.pdf#page=8|MIT 6.100L Lecture 13，PDF p. 8]]；[Python Tutorial §8.7, Defining Clean-up Actions](https://docs.python.org/3/tutorial/errors.html#defining-clean-up-actions)：核对清理时机、待传播异常与新控制转移；示例执行核对“清理后仍保留修改”。
  <!-- bilingual-en:start -->
  These sources support cleanup timing, pending exceptions, and replacement transfers. Execution of the example confirms that the prior mutation remains after cleanup.
  <!-- bilingual-en:end -->
- [Python os._exit](https://docs.python.org/3/library/os.html#os._exit)：核对直接退出不调用清理处理器的边界；未执行退出操作。
  <!-- bilingual-en:start -->
  This entry supports the direct-exit boundary; no exit operation was executed.
  <!-- bilingual-en:end -->

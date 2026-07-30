---
aliases: [Gale-Shapley Correctness Proof, 延迟接受算法正确性证明]
tags: [proof, discrete-mathematics, algorithms]
type: proof
---
# Gale-Shapley Correctness Proof

## 假设

两侧各有 $n$ 人，每人都对另一侧给出严格、完整偏好；proposer 每次向尚未求婚者中偏好最高的一人求婚，receiver 始终暂留当前收到的最佳求婚。

## 终止

每次循环产生一对以前未发生的求婚；有限参与者给出有限对，所以算法终止。

终止时不可能还有未匹配 proposer。否则他必已向全部 $n$ 位 receiver 求过婚并被拒绝；每次拒绝都意味着 receiver 当时留有某位 proposer，且 receiver 此后始终保持暂配。于是全部 $n$ 位 receiver 都有暂配，需占用 $n$ 位不同 proposer，与仍有一位 proposer 未匹配矛盾。因此输出是完全匹配。

## 稳定性

假设输出存在 blocking pair $(m,w)$。$m$ 更喜欢 $w$ 胜过最终对象，故他必在向最终对象求婚前向 $w$ 求过婚。$w$ 当时拒绝了 $m$，或后来为更喜欢的人抛弃 $m$。receiver 的暂配对象只会沿自身偏好变好，所以最终对象至少与当时拒绝 $m$ 的对象一样受偏好。于是 $w$ 不可能更喜欢 $m$，矛盾。

因此输出没有 blocking pair。

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```

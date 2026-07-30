---
aliases:
  - Discrete Mathematics Hub
  - 离散数学 Hub
  - Mathematics for Computer Science Hub
tags:
  - hub
  - discrete-mathematics
---

# Discrete Mathematics Hub

离散数学研究有限或可数结构上的逻辑、组合、图与概率。本 Hub 以 MIT 6.042J 的四条主线组织可复用知识卡，而不是复制课程的 Session 目录。

## 课程入口

- [[00_MIT OCW 6.042J course map|MIT 6.042J course map]]
- [[01_Proofs|Unit 1: Proofs]]
- [[02_Structures|Unit 2: Structures]]
- [[03_Counting|Unit 3: Counting]]
- [[04_Probability|Unit 4: Probability]]
- [[05_Review and exam roadmap|Review and exam roadmap]]

## Proofs

### 逻辑对象

- [[Propositional Logic]]
- [[Truth Table]]
- [[Logical Equivalence]]
- [[Predicate Logic and Quantifiers]]
- [[Mathematical Statement Types]]
- [[Mathematical Proof]]
- [[Set Operations]]
- [[Cardinality]]
- [[Binary Relation]]
- [[Countability]]

### 证明选择与执行

- [[Proof Strategy Selection]]
- [[Direct Proof]]
- [[Proof by Contradiction]]
- [[Proof by Contrapositive]]
- [[Proof by Cases]]
- [[Well-Ordering Principle]]
- [[Mathematical Induction]]
- [[Induction Proof Procedure]]
- [[State Machine Invariant]]
- [[State Machine]]
- [[Ranking Function]]
- [[Invariant Proof Procedure]]
- [[Recursive Definition]]
- [[Structural Induction]]
- [[Cantor Diagonal Argument]]
- [[Cantor-Schroeder-Bernstein Theorem]]
- [[Cantor's Theorem]]
- [[Halting Problem]]

## Structures

### 数论与密码

- [[Divisibility]]
- [[Greatest Common Divisor]]
- [[Euclidean Algorithm]]
- [[Bezout Identity]]
- [[Modular Arithmetic]]
- [[Modular Inverse]]
- [[Chinese Remainder Theorem]]
- [[Fermat's Little Theorem]]
- [[Euler Totient Function]]
- [[Euler Totient Theorem]]
- [[RSA Cryptosystem]]

### 图、序与匹配

- [[Directed Graph]]
- [[Graph Adjacency Matrix]]
- [[Walk Path and Cycle]]
- [[Directed Acyclic Graph]]
- [[Topological Sort]]
- [[Partial Order]]
- [[Chain and Antichain]]
- [[Hasse Diagram]]
- [[Equivalence Relation]]
- [[Simple Graph]]
- [[Vertex Degree]]
- [[Graph Isomorphism]]
- [[Graph Coloring]]
- [[Bipartite Graph]]
- [[Graph Connectivity]]
- [[Graph Tree]]
- [[Spanning Tree]]
- [[Minimum Spanning Tree]]
- [[Graph Matching]]
- [[Hall's Marriage Theorem]]
- [[Stable Matching]]
- [[Gale-Shapley Algorithm]]

### 可复用结构证明

- [[Euler Totient Theorem Proof]]
- [[RSA Correctness Proof]]
- [[Handshake Lemma Proof]]
- [[Gale-Shapley Correctness Proof]]

## Counting

- [[Series]]
- [[Integral]]
- [[Geometric Series]]
- [[Annuity]]
- [[Harmonic Number]]
- [[Harmonic Series Divergence]]
- [[Stirling's Approximation]]
- [[Rule of Sum and Product]]
- [[Counting Strategy Selection]]
- [[Asymptotic Notation]]
- [[Bijective Counting Principle]]
- [[Binomial Theorem]]
- [[Multinomial Theorem]]
- [[Stars and Bars]]
- [[Pigeonhole Principle]]
- [[Inclusion-Exclusion Principle]]

## Probability

- [[Discrete Probability Space]]
- [[Four-Step Probability Method]]
- [[Conditional Probability and Bayes Theorem]]
- [[Independence of Events]]
- [[Mutual Independence]]
- [[Random Variable]]
- [[Probability Mass Function]]
- [[Probability Density Function]]
- [[Indicator Random Variable]]
- [[Expected Value]]
- [[Linearity of Expectation]]
- [[Variance]]
- [[Markov Inequality]]
- [[Chebyshev Inequality]]
- [[IID]]
- [[Law of Large Numbers]]
- [[Convergence in Probability]]
- [[Confidence Level]]
- [[Confidence Interval]]
- [[Random Walk on a Graph]]
- [[Markov Chain]]
- [[Markov Matrix]]
- [[Stationary Distribution]]
- [[PageRank]]

### 可复用概率证明

- [[Markov Inequality Proof]]
- [[Chebyshev Inequality Proof]]
- [[Weak Law of Large Numbers Proof]]

## 使用边界

- 课程例题、完整作业和考试题解留在 Unit 长笔记。
- concept 卡只保存定义、符号和最小例子。
- procedure 卡保存可执行步骤；framework 卡保存选择条件和失败模式。
- proof 卡只保留可复用的假设—推导—结论链条。
- 时间序列中的 random walk、期权 Greek `Theta` 与统计学中的 broader asymptotic theory 都可能同名或近名；链接时必须指向本课程语义对应的卡片。

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

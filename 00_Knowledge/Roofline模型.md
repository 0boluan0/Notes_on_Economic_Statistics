---
aliases:
  - Roofline 模型用计算上限和带宽上限共同约束给定算术强度下的性能
  - The Roofline model bounds performance by both compute capacity and memory bandwidth at a given arithmetic intensity
  - Roofline performance model
student_os: knowledge-atom
atom_id: LLM-INF-033
atom_type: definition
status: source-checked
part_of:
  - "[[LLM 推理效率.canvas]]"
---

# Roofline 模型用计算上限和带宽上限共同约束给定算术强度下的性能
<!-- bilingual-en:start -->
*The Roofline model bounds performance by both compute capacity and memory bandwidth at a given arithmetic intensity*
<!-- bilingual-en:end -->

Roofline 是一个性能上界模型：计算单元每秒能做的操作有限，存储系统每秒能提供的数据也有限。对固定工作与实现，取匹配精度的计算上限 $P$（FLOP/s）、选定存储层级的带宽上限 $B$（byte/s），以及该层级的 [[算术强度]] $I=F/D$，则

<!-- bilingual-en:start -->
Roofline is a performance-bound model: both arithmetic throughput and data supply are finite. For fixed work and implementation, use a precision-matched compute limit $P$, bandwidth limit $B$ at the chosen memory level, and [[算术强度|arithmetic intensity]] $I=F/D$:
<!-- bilingual-en:end -->

$$
\text{实际性能}\leq\min(P,BI),
\qquad
T\geq\max\!\left(\frac FP,\frac DB\right).
$$

两条限制在 $I^*=P/B$ 相交。$I<I^*$ 时，带宽给出的上界更低；$I>I^*$ 时，计算能力给出的上界更低。这是在模型中识别限制，不等于已经用 profiler 证明真实程序的瓶颈。公式中的 max 也不是把计算时间和搬运时间直接相加；逼近该下界需要足够并行性及有效重叠。

<!-- bilingual-en:start -->
The bounds intersect at $I^*=P/B$. Below this intensity, bandwidth imposes the lower roof; above it, compute does. This identifies the tighter modeled constraint, not a profiler-confirmed bottleneck. Approaching the time lower bound requires enough parallelism and effective overlap; the two times are not simply added.
<!-- bilingual-en:end -->

取一台假想设备：$P=100$ TFLOP/s、$B=1$ TB/s，均用十进制单位。一次计算有 $F=10^{12}$ FLOP、$D=50\times10^9$ byte，故 $I=20$ FLOP/byte；计算至少需 10 ms，搬运至少需 50 ms，整体至少 50 ms。只把计算上限提高两倍，仍不能突破这条 50 ms 搬运下界。若减少实际流量，才可能抬高这一侧的上界。

<!-- bilingual-en:start -->
On a hypothetical device with 100 TFLOP/s compute and 1 TB/s bandwidth, work of $10^{12}$ FLOPs and 50 GB of traffic has intensity 20 FLOP/byte. Compute needs at least 10 ms and traffic at least 50 ms. Doubling compute capacity leaves the 50 ms traffic bound unchanged. Reducing actual traffic may raise that side of the roof.
<!-- bilingual-en:end -->

实际运行还可能受指令混合、内核形状、缓存、同步、通信或启动开销限制。低于 roof 很多时，不能只给程序贴“memory-bound”标签；应进入 [[推理瓶颈诊断]] 查明哪些时间没有被这个两资源模型解释。尤其不要用整台设备的峰值矩阵算力，解释只能调用普通标量单元的操作。

<!-- bilingual-en:start -->
Instruction mix, kernel shapes, caching, synchronization, communication, and launch overhead may impose further limits. A large gap below the roof calls for [[推理瓶颈诊断|bottleneck diagnosis]], rather than an unsupported memory-bound label. Peak matrix throughput is not the applicable compute roof for operations that use different execution units.
<!-- bilingual-en:end -->

## 来源与核验

- [Samuel Williams, CS267 Roofline lecture](https://amcr.lbl.gov/wp-content/uploads/2025/11/CS267-Roofline-SWWilliams-compressed.pdf)，Roofline Model、Superscalar vs. instruction mix、Locality Walls：支持计算/带宽屋顶、附加限制和流量口径。
- [NVIDIA, *GPU Performance Background User's Guide*](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)，§4：支持 $\max(T_{\rm mem},T_{\rm math})$ 的重叠模型及强度阈值。时间不等式和假想设备算例由模型直接推导，不是实测性能。

<!-- bilingual-en:start -->
Williams supplies the Roofline model and additional ceilings. NVIDIA states the overlap model and intensity threshold. The time inequality and hypothetical-device example are direct derivations, not benchmark results.
<!-- bilingual-en:end -->

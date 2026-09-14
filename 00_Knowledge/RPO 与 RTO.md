---
aliases:
  - "RPO 定义可接受的数据回退点，RTO 定义可接受的服务恢复时限"
  - RPO defines the acceptable data recovery point while RTO defines the acceptable restoration time
  - RPO 与 RTO
  - 恢复点目标与恢复时间目标
student_os: knowledge-atom
atom_id: CS-SEC-017
atom_set: cryptographic-primitives-security-models
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[密码学原语与安全模型.canvas]]"
related:
  - "[[备份恢复需演练验证]]"
---

# RPO 定义可接受的数据回退点，RTO 定义可接受的服务恢复时限
<!-- bilingual-en:start -->
*RPO defines the acceptable data recovery point, while RTO defines the acceptable restoration time*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 恢复点目标（RPO）说明故障后数据必须恢复到故障前哪个时间点，因此限定可容忍的数据丢失窗口；恢复时间目标（RTO）说明系统资源最多可中断多久，超过该时限就会造成不可接受影响。
> <!-- bilingual-en:start -->
> A recovery point objective states the point in time before a disruption to which data must be recovered and therefore bounds tolerable data loss. A recovery time objective states the maximum time a system resource may remain disrupted before the impact becomes unacceptable.
> <!-- bilingual-en:end -->

例如，RPO 为 15 分钟意味着恢复方案要能把数据带回到故障前最多 15 分钟的位置；它不意味着系统 15 分钟内重新上线。RTO 为 4 小时意味着服务恢复必须在 4 小时内完成；它不说明会丢失多少数据。
<!-- bilingual-en:start -->
For example, an RPO of 15 minutes means the recovery design must restore data to a point no more than 15 minutes before the disruption. It does not mean the service will be online within 15 minutes. An RTO of four hours means restoration must complete within four hours; it says nothing by itself about how much data may be lost.
<!-- bilingual-en:end -->

两者把“能恢复”改写为可验证目标。备份频率、复制方式和日志保留要对准 RPO；基础设施准备、依赖重建、人员流程和演练速度要对准 RTO。只看到备份任务成功，尚未证明任何一个目标已经满足。
<!-- bilingual-en:start -->
Together they turn “recoverable” into testable objectives. Backup frequency, replication, and log retention must support the RPO; infrastructure readiness, dependency reconstruction, operating procedures, and exercise speed must support the RTO. A successful backup job alone proves neither objective.
<!-- bilingual-en:end -->

> [!question]- 自检
> “最多丢 30 分钟数据”和“2 小时内恢复服务”分别是哪一个目标？
>
> **答案：** 前者是 RPO，限定可接受的数据回退窗口；后者是 RTO，限定可接受的服务中断时间。

## 来源与核验

- [NIST SP 800-34 Rev. 1, Sections 3.4.1–3.4.2](https://csrc.nist.gov/pubs/sp/800/34/r1/upd1/final)：核对 RPO 是故障前数据必须恢复到的时间点，RTO 是系统资源不可用的最长可接受时段；本卡采用通用定义，不移植美国联邦机构特有流程。

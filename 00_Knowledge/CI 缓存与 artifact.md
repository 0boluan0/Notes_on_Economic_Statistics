---
aliases:
  - "CI 缓存是可缺失的加速层而 artifact 是需要保留或传递的产物"
  - CI cache versus workflow artifact
  - CI 缓存与 artifact
student_os: knowledge-atom
atom_id: CS-BUILD-018
atom_set: build-dependencies-ci
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[CI 工作流结构]]"
part_of:
  - "[[构建、依赖与 CI.canvas]]"
---

# CI 缓存是可缺失的加速层而 artifact 是需要保留或传递的产物
<!-- bilingual-en:start -->
*A CI cache is an optional acceleration layer, while an artifact is output that must be retained or transferred*
<!-- bilingual-en:end -->

> [!summary] 原子区别
> cache 保存昂贵但可重新下载或生成的依赖与中间结果，cache miss 只应让 job 更慢。artifact 保存 job 产生、需要在 run 后查看或交给其他 job 的 binary、报告或日志。两者都存文件，但生命周期和正确性角色不同，不能互换。
> <!-- bilingual-en:start -->
> A cache stores expensive but re-downloadable or regenerable dependencies and intermediate results; a cache miss should merely make the job slower. An artifact preserves binaries, reports, or logs produced by a job for later viewing or transfer to another job. Both store files, but their lifecycle and correctness roles differ and they are not interchangeable.
> <!-- bilingual-en:end -->

若删除缓存后构建失败，缓存中已有某个未声明依赖或生成物。修复是把它变成正式安装、下载或生成步骤，而不是提高 cache hit rate。反过来，需要作为发布候选或调试证据保留的输出，应明确上传为 artifact。
<!-- bilingual-en:start -->
If clearing the cache breaks the build, the cache contained an undeclared dependency or generated file. The fix is to make it a real install, download, or generation step, not to increase the cache-hit rate. Conversely, an output that must survive as a release candidate or diagnostic record should be uploaded explicitly as an artifact.
<!-- bilingual-en:end -->

## 安全边界

GitHub 明确要求把恢复的 cache 当作不可信输入，并禁止在 cache 中保存 secrets。能读取 cache 的低信任 workflow 可能读取敏感内容，污染 cache 还可能影响后续可信 workflow。
<!-- bilingual-en:start -->
GitHub explicitly requires restored caches to be treated as untrusted input and forbids storing secrets in a cache. A low-trust workflow able to read a cache may expose sensitive contents, and cache poisoning can affect a later trusted workflow.
<!-- bilingual-en:end -->

> [!question]- 自检
> 删除 dependency cache 后 job 需要多花三分钟重新下载但仍通过，这说明了什么？
>
> **答案：** cache 正在充当可缺失的性能优化，没有成为正确性所必需的隐藏输入。

## 来源与核验

- [GitHub Actions Documentation, Dependency caching](https://docs.github.com/en/actions/concepts/workflows-and-actions/dependency-caching)：核对 cache miss 可重新生成、artifacts 的用途差异，以及 cache 不可信和不得保存 secrets。
<!-- bilingual-en:start -->
- [GitHub Actions Documentation: Dependency caching](https://docs.github.com/en/actions/concepts/workflows-and-actions/dependency-caching) was checked for regenerating after cache misses, artifact use cases, and the untrusted/no-secrets cache boundary.
<!-- bilingual-en:end -->

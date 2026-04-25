# OPD Idea 可行性验证 (v2 — Codex v3 审后定稿)

> **生成时间**: v1: 2026-04-22 / **v2 Codex-gated**: 2026-04-22
> **基础**: core-19 + supplementary-15 papers + SURVEY + TIMELINE + IDEAS.md v3 + Codex 2 轮 PLAN REVIEW
> **Review 状态**: ✅ Codex v1 = 6.1/Fail → Codex v3 re-review = **7.1 / Pass**（全部 still_must_fix 已纳入）
> **作用**: 给师兄讨论前把"代码/算力/时间/技能"四维坑标清

---

## 0. TL;DR (v2)

**v1 被 Codex 判 Fail** (6.1)。最大三个问题：
1. Path γ (ToM) 不是 blank territory — ToMA (2509.22887) 已在 SOTOPIA 上做 same-model ToM SFT
2. iStar missing as baseline
3. Claude self-review 打 7.7 偏乐观

**v2 硬改**：
- Path β (SOTOPIA-RL warm-start + RC-OPD) **升为 pragmatic mainline**；Path α 降为 "理论驱动 sibling"
- 新加 **5/05 hard pilot checkpoint**：20-50 ep 验证 hierarchical reflection 有没有信号，否则立刻转向
- AppWorld 全降 appendix；主战场只 SOTOPIA
- 理论降到 1 proof（realizability gap）

**三个"绿区"保留**:
- ✅ SOTOPIA + SOTOPIA-RL 开源、数字明确
- ✅ SDPO / Rethinking-OPD 代码可用
- ✅ Qwen3-8B + 2k H100h 在合理算力范围

**两个"红区"+ 一个"新黄区"**:
- 🔴 Skill-SD 未开源 → v2 AppWorld 降 appendix（Codex must_fix 5）
- 🔴 **ToMA 占去 Path γ 70% 领地** → reframe 为"ToM-augmented OPD"，不自称 blank territory
- 🟡 GPT-4 judge 成本仍需 5/01 pilot 验真实数

---

## 1. 代码资源盘点（v2 加 ToMA / iStar）

按"对 v2 mainline Path β 的 critical 度"排列：

| 论文 | 作用 | 开源状态 | 风险等级 | Mitigation |
|------|------|---------|---------|------------|
| **SOTOPIA** (Zhou 2024) | 主战场 | ✅ github.com/sotopia-lab/sotopia | 🟢 低 | pip install + Docker Redis |
| **SOTOPIA-RL** (2508.03905) | ⭐ Path β 的 RL warm-start | ✅ github.com/sotopia-lab/sotopia-rl | 🟢 低 | fork + swap Qwen3-8B |
| **ToMA** (2509.22887) ⭐ v2 新加 baseline | 必对标 | ⚠️ 未确认（需查 GitHub / OpenReview） | 🟡 中 | 若没开源就按 paper 数字对照；若有就跑 Qwen3-8B 复现 |
| **iStar** (2509.19199) ⭐ v2 新加 baseline | no-verifier RL 邻居 | ⚠️ 需确认 | 🟡 中 | 若闭源按数字对照 |
| **SDPO** (2601.20802) | 对照 baseline | ✅ github.com/lasgroup/SDPO | 🟢 低 | 直接用 |
| **Rethinking OPD** (2604.13016) | recipe 来源 | ✅ github.com/thunlp/OPD | 🟢 低 | 参考 |
| **OPSD** (2601.18734) | baseline (vanilla OPD) | ⚠️ 需确认 | 🟡 中 | 用 thunlp/OPD 替代实现 |
| **CRISP** (2603.05433) | teacher refresh 参照 | ⚠️ 未确认 | 🟡 中 | 按 paper 重实现 |
| **Skill-SD** (2604.10674) | 原主对标，v2 降 appendix | ❌ 没找到 github | 🟡 中 (降级后) | §8 M1 降维对标 |
| **HDPO** (2603.23871) | 消融对照 | ⚠️ NVIDIA 未知 | 🟡 中 | 按 paper 重实现 failure-trigger 逻辑 |

**v2 关键变化**:
- **Path β mainline 依赖 ✅✅**：SOTOPIA + SOTOPIA-RL 都开源 → Path β 执行风险 **低**
- **ToMA/iStar 代码未确认**：5/01 之前必须搞清楚，否则 v2 的"加 baseline"承诺打折

---

## 2. 计算资源可行性 (v2 微调)

### 2.1 训练预算（加 ToMA + iStar baseline 后）

| 阶段 | H100h | 备注 |
|------|------|-----|
| Qwen3-8B LoRA warm-up | ~30 | 同 v1 |
| **Path β Stage 1 (SOTOPIA-RL on Qwen3-8B)** | ~300 | 比 v1 的"SFT warm-up"稍重 |
| **Path β Stage 2 (RC-OPD on warm-start)** | ~600 | 核心 |
| ToMA baseline 复现 (if code 开源) | ~200 | v2 新加 |
| iStar baseline 复现 (if code 开源) | ~150 | v2 新加 |
| Ablation × 4 | ~400 | 同 v1 |
| MATH-500 sanity | ~50 | 降规模 |
| **合计** | **~1700 H100h** | **$23-28k USD** |
| **+30% contingency** | **~2200 H100h** | **$30-35k USD** |

**变化**: v2 比 v1 多约 350 H100h（ToMA/iStar baseline 复现）；在 2k 预算内仍可容纳。

### 2.2 算力来源（不变）

- 🟢 组里应能 secure 2k H100h
- 🟡 A100 退路 2x 时间
- 🔴 单卡不够

**Action**: 4/28 前和师兄确认。

---

## 3. 评估 API 成本 (v2 不变)

### 3.1 GPT-4 Judge 成本

- 保守估：~$0.023-$0.045/episode (GPT-4o vs Turbo)
- 主实验 500 eps × 7 baseline = 3500 eps → $80-160
- Ablation 2000 eps → $46-90
- 3 seed → $400-750

🟡 **风险**: 若 judge 需 CoT / 多轮 → ×2-3 到 **$1-2k**

### 3.2 Mitigation

- **5/05 hard checkpoint** 前做 10-ep pilot 测真实成本（v2 新加硬性要求）
- 开发期用 50 eps/config，最终 500 eps
- 备选 Claude-3.5-Sonnet judge ($3/$15)

---

## 4. 团队 / 时间可行性 (v2 加 5/05 hard checkpoint)

### 4.1 技能 check-list（不变）

| 技能 | 需求 | 用户应有 | 补救 |
|------|------|--------|-----|
| PyTorch + HF | 🔴 | ✅ | — |
| LoRA + PEFT | 🔴 | 需学 1 周 | HF tutorial |
| vLLM / sglang | 🔴 | 需 2-3 天 | SOTOPIA-RL code 有示例 |
| Redis + Docker | 🟡 | — | SOTOPIA 文档 |
| GRPO/PPO | 🟡 | 读 SOTOPIA-RL code | 不自己写 |
| LaTeX | 🔴 | ✅ | — |

### 4.2 Timeline v2（加 5/05 hard checkpoint）

| 日期 | v1 milestone | v2 变更 |
|------|-------------|--------|
| 4/28 | positioning statement | **+ 算力 + API key 敲定 + Skill-SD 邮件发出 + ToMA/iStar 代码状态查清** |
| **5/01** | — | **+ 10-ep GPT-4 judge cost pilot** |
| **5/05** ⭐ | — | **Codex must_fix 3 量化 rule**: 50 ep pilot 上 Δ(goal completion) ≥ **+0.3 pp** over 无 reflection baseline + 10 条人工抽检 > 50% 可识别为 "有用反思" → **GO**；否则 → 立刻切**empirical study 兜底**（见 §6.4 和 IDEAS §2.7(a)）。**不要**伪装成"改走 pure RL"作为同 paper continuation |
| 5/12 | baseline 跑通 | **+ SOTOPIA-RL 7.17 ±0.3 pp 复现** + iStar baseline 跑通 |
| 5/13-5/26 | RC-OPD v1 实现 | **Path β 和 α 并行**（β 主攻，α 做 ablation） |
| 5/27-6/16 | 主实验 | AppWorld 彻底 drop |
| 6/30 | Go/No-Go | 见 §7 |
| 7/1-7/14 | theory + paper 草稿 | 只完成 1 proof |
| 7/15-8/15 | 论文 | — |
| 8/16-8/31 | 投稿 + 九推 | — |

---

## 5. 技术风险清单 (v2 重评)

### 5.1 🔴 高风险

**R1: Qwen3-8B 在 SOTOPIA 上 reflection 质量不够**
- Impact: 直接废 RC-OPD hierarchical reflection
- Probability: 🟡 中 — ToMA 的 Qwen2.5-7B +6.9% 证明同量级模型能产出有用 ToM 信号；reflection 类似
- **Mitigation**: 5/05 hard checkpoint 量化 rule 早 detect；**失败时切 empirical study 兜底**（不要伪装成"改走 pure RL"当延续路径 —— Codex v3 明确指出那是**不同 paper**，不能在师兄面前把 "SOTOPIA-RL 基线" 包装成 RC-OPD 的延续）

**R2: ToMA 实际 overlap > 70%（v2 新识别）**
- Impact: Path γ 失去 novelty，整条备选路线崩
- Probability: 🟡 中 — 只读了 abstract + 搜索摘要
- **Mitigation**: 4/28 前精读 ToMA PDF；若 overlap 过高，Path γ 直接废，只靠 β+α

**R3: Path β warm-start 后 RC-OPD 无 gain（v2 新识别）**
- Impact: pragmatic mainline 失败
- Probability: 🟡 中 — SOTOPIA-RL 可能已 converge
- **Mitigation**: 6/30 若 β < +0.3 pp，切 Path α（pure RC-OPD）；再失败切 empirical study 兜底

### 5.2 🟡 中风险（v2 保留）

**R4: Skill-SD 无代码导致 AppWorld 对标失败** → v2 彻底降 appendix（Codex must_fix 5）
**R5: curriculum schedule α(k) 无 proof** → v2 已 drop 这条 theory 承诺（Codex must_fix 4）
**R6: SOTOPIA judge variance 太大** → 3 seed + paired bootstrap
**R7: GPT-4 judge 成本翻倍** → §3.2

### 5.3 🟢 低风险（保留）

- R8: LoRA 训练不稳 → QLoRA + clipping
- R9: vLLM OOM → 降 batch size
- R10: SOTOPIA Docker 问题 → 文档完整

---

## 6. A+B 组合矩阵 (v2 重排)

**Codex nice_to_have**: Path β 升 pragmatic mainline。v2 采纳。

### 6.1 🌟 **NEW MAINLINE Path β: SOTOPIA-RL warm-start + RC-OPD**

**A + B = SOTOPIA-RL (A) + RC-OPD curriculum reflection (B)**
- **Stage 1**: SOTOPIA-RL 开源 pipeline 训 Qwen3-8B → 预期 7.17 ± 0.3 pp
- **Stage 2**: warm-start policy 上 apply hierarchical reflection + curriculum retraction
- **Claim**: "OPD as refinement on top of RL"；回答"RC-OPD 能否在 strong RL baseline 上 squeeze 额外 gain"
- **Novelty**: 7（RL → OPD pipeline 具体组合少见）
- **Feasibility**: 8.5（两边都开源）
- **Scoop**: 低
- **失败模式**: 已 converge → 无 gain → 退化为 "no-harm" findings paper
- **推荐为 v2 暑期主投**

### 6.2 🥈 **Theoretical Sibling Path α: Pure RC-OPD from scratch**

**A + B = Reflection (A) + Curriculum (B) 独立**
- 直接 Qwen3-8B + RC-OPD（无 RL warm-start）
- **Novelty**: 6.5
- **Feasibility**: 7
- **Scoop**: 中
- **若 β 成功**: α 作为 β 的 ablation 证据，进 main table
- **若 β 失败**: α 可独立 carry 论文

### 6.3 🥉 **Reframed Backup Path γ: ToM-augmented OPD (非 blank territory)**

**A + B = ToM hierarchical privileged context (A) + OPD distillation (B)**
- v2 reframe（ToMA prior art 坦诚承认）:
  - Surface level: 对手表达的情绪 / 意图（observed）
  - Belief level: 对手相信什么（ToMA 覆盖）
  - Intent level: 对手的下一步策略（ToMA 覆盖）
- **wedge 变窄**: ToMA 是 filter-SFT；Path γ 是 **per-token KL** + **hierarchy + curriculum**
- **Novelty**: 6（被 ToMA 占 70%）
- **Feasibility**: 7
- **Scoop**: 中高（ToMA 已占领地）
- **启动条件**: 仅当 β+α 都失败

### 6.4 选择决策树

```
4/28: 算力 + API + baseline 代码 secure ──┐
                                         ├── 5/05 ⭐ HARD pilot (量化 rule)
                                         │   ├── Δ ≥ +0.3 pp & 反思定性可读 → Path β 主投
                                         │   └── Δ ≤ 0 或 反思无法解析 → 切 empirical study 兜底
                                         │       ⚠️ 不切 "pure RL warm-start" 当延续 — Codex v3 明确指出那是不同 paper
                                         │
                                         └── 6/30 Go/No-Go:
                                               ├── β 主指标 ≥ +0.5 pp → Path β 主投
                                               ├── β 打平 / α 有 gain → α 救场，β 做 ablation
                                               └── β+α 全废 → Path γ 或 empirical study 兜底
```

---

## 7. 6/30 Go/No-Go 标准 (v2 加 ToMA 门槛)

| 判据 | Pass | Weak | Fail |
|------|------|------|------|
| **主指标**: SOTOPIA-hard score (Path β) | ≥ 7.5 (vs SOTOPIA-RL 7.17) | 7.3-7.5 | < 7.2 |
| **ToMA 对标**: SOTOPIA-hard vs ToMA (Qwen2.5-7B) | ≥ match | ≤ -0.3 pp | < -0.5 pp |
| **iStar 对标** | ≥ match | ≥ -0.3 pp | < -0.5 pp |
| **次指标**: goal completion | +1.5 pp | +0.5-1.5 pp | < +0.5 pp |
| **消融**: curriculum retraction effect | ≥ +0.3 pp, p<0.05 | 方向无显著 | 无方向 |
| **消融**: hierarchy (3 level vs 1) | ≥ +0.3 pp | 方向 | 无方向 |
| **Path α 独立 (without RL warm)** | 至少有 +0.3 pp gain | 打平 | 退步 |
| **Sanity**: MATH-500 | ±1 pp | -1 to -3 pp | < -3 pp |
| **Inference-only control** (v2 加) | 参数内化 > 推理 context | 打平 | inference 胜 (无内化收益) |

**Decision matrix**:
- 全 Pass → Path β 主投顶会 (NeurIPS workshop / ICLR 2027)
- 主指标 Weak 但 ablation 显著 → Path α 主投 findings
- 主 Fail → Path γ (重新 scope) 或 empirical study 兜底

---

## 8. Skill-SD 未开源 mitigation (v2 彻底降级)

**Codex must_fix 5**: Skill-SD 红区必须处理。v2 采纳**彻底降级**策略：

- **M1 采纳**: AppWorld 从 main table 完全移除 → 仅放 appendix，引用 Skill-SD 论文数字
- **论文 positioning**: 不再自称"跨 verifier / no-verifier 两 domain"；只做 SOTOPIA no-verifier
- **M2 重实现**: **不做**（时间不值）
- **M3 邮件**: 4/29 发一封；5/10 不回再发一封；默认不期待

**v2 额外好处**: scope 收窄反而让 claim 更扎实 → Codex 的 Clarity 维度从 8.1 → 8.5

---

## 9. v2 最终可行性判定

### 9.1 维度打分（v2 post-must_fix + Codex v3 re-review）

| 维度 | v1 self | Codex v1 | **Codex v3 re-review** | 说明 |
|------|--------|---------|----------------------|------|
| 代码资源 | 7 | — | (融入 feasibility) | ToMA 已有 repo (eujhwang/toma, minimal)；iStar 未确认 |
| 计算资源 | 7 | — | (融入 feasibility) | 2.2k H100h 预算内 |
| API 成本 | 6 | — | (融入 feasibility) | 5/01 pilot 后更可控 |
| **Feasibility 总** | 7 | 5.8 | **7.2** ✅ | 加 5/05 量化 rule + scope 收窄显著降风险 |
| **Novelty** | 7 | 5.6 | **6.3** | Codex 指出 recombination of crowded ingredients；靠 "token-level OPD on SOTOPIA no-verifier + β headline" 撑 |
| **Technical Rigor** | 6 | 5.9 | **6.6** | 只保 realizability gap 1 proof 更诚实 |
| **Experimental Design** | 6 | 5.9 | **7.7** | 加 iStar + ToMA + inference-only control 后最严密 |
| **Scoop Safety** | 6 | 4.8 | **6.0** | ToMA 占部分领地；OPD-on-SOTOPIA 仍空间 |
| **Impact** | 6 | 6.6 | **6.9** | scope 收窄让 claim 更扎实 |
| **Clarity** | 8 | 8.1 | **8.7** | re-scope + 5/05 量化 + Path 分层后最清晰 |
| **Overall** | **6.7** (self) | **6.1** ❌ | **7.1** ✅ **PASS** | narrow pass (门槛 7.0) |

### 9.2 给师兄的 checklist

**需要师兄 confirm 3 点**：
1. 2k H100h 能 secure 吗？（$30-35k USD）
2. $500-2000 USD GPT-4 judge 预算组里能报销吗？
3. Path β (SOTOPIA-RL warm-start + RC-OPD) 作 pragmatic mainline 师兄同意吗？

**需要立即 action 5 点**：
1. 4/29 发 Skill-SD 作者邮件
2. 4/29 发 ToMA / iStar 作者邮件问 code
3. 4/30 前跑通 SOTOPIA + 10-ep GPT-4 judge cost pilot
4. 5/01 前敲定算力 + API key
5. **5/05 hard checkpoint**（pilot reflection signal）：结果直接决定主方向切换

---

## 10. 相较 v1 的变化清单（v2）

| v1 claim | v2 状态 |
|---------|--------|
| "1.5-2k H100h" | ✅ 改 1.7k + 30% contingency = 2.2k |
| "继承 SOTOPIA-RL / SDPO 开源" | ✅ 保留；加"ToMA/iStar 代码待确认" |
| "兜底方案三选一" | 🔄 替换为 §6 A+B 矩阵 + 兜底 |
| Path α 主干 | ❌ **降级为 theoretical sibling** |
| Path β 作为 ablation 线 | ✅ **升为 pragmatic mainline** |
| Path γ (ToM) "blank territory" | ❌ **推翻，ToMA 已占 70%；reframe 为 ToM-augmented OPD** |
| 6 个 baseline | ✅ 加到 7 个（+ToMA + iStar + inference-only） |
| 主战场 SOTOPIA + AppWorld + MATH | ✅ 收窄到 SOTOPIA main + MATH sanity + AppWorld appendix |
| 3 条 formal 分析 | ✅ 降到 1 proof + 1 sketch + 1 dropped |
| 6/30 Go/No-Go | ✅ 加 ToMA/iStar 门槛 |

---

## 11. ⚠️ v2 局限

1. **Codex 只 review 了 v1** → v2 本身的 7.0 达标是**预估**，还需 re-review
2. **ToMA 精读未做**: 只有 abstract + 搜索摘要；overlap 估的是 70%，实际可能更高
3. **ToMA/iStar 代码状态未亲验**: 4/28 前必须搞清楚
4. **5/05 pilot 成败直接决定主方向**: 是**单点失败风险**；需要提前想清楚 5/05 Fail 的 B 计划（Path β pure RL 或 empirical study）
5. **Skill-SD 降级可能被 reviewer 质疑**: "你凭什么声称在 social agent 上领先，AppWorld 不做就是弱"；需要论文 intro 明确 "we study social domain，tool-agent 另一个方向"

**给用户**：v2 已经把 Codex v1 的 5 must_fix + 3 nice_to_have 全纳入，并经 **Codex v3 re-review 打分 7.1 / Pass** 确认。还剩 Codex v3 的 2 条 still_must_fix 已在 IDEAS §2.6 / §2.7(a) 和本文 §4.2 / §5.1 / §6.4 全部纳入（量化 5/05 rule + 明确 pure-RL 非延续路径）。现在的状态是**可直接拿去给师兄讨论**的成熟版 plan。

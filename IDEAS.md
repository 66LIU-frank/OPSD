# OPD 方向 Idea 提案 (v3 — Codex 外审后硬改版)

> **生成时间**: v1 / v2 自审 / **v3 Codex-gated**: 2026-04-22
> **前提**: 19 核心 PDF + 15 补充 PDF + 8 轮 WebSearch + Codex [PLAN REVIEW] 打分
> **Review 状态**: ✅ Codex 外审完成（v2 = overall 6.1 / Fail）；v3 按五项 must_fix + 三项 nice_to_have 硬改
> **Rubrics 打分门槛**: overall ≥ 7.0 且无单维 ≤ 3

---

## §0. 一页纸讲 idea（先看这里）

> 下面这节是给师兄/本人 review 时的**叙事入口**。Rubrics / scoop / wedge 那些都是后续工作文档（§0' 开始）。

### §0.1 研究背景与 Research Gap

**On-Policy Distillation (OPD) 是什么**：student LLM 自己 rollout，teacher LLM 逐 token 做 KL 监督。它是 2026 年初（OPSD 2026-01-26）开山、3-4 月集中爆发的新蒸馏范式 —— 结合 RL 的"on-policy"性质和蒸馏的"dense 信号"两大优点。

**关键机制是 "privileged context"**：teacher 需要比 student 多知道点东西才有意义。已有工作把 privileged context 设计为：

| 方法 | privileged context 用的是什么 | 适用场景 |
|------|----------------------------|---------|
| OPSD / HDPO / Self-Distilled RLVR | **标准答案 (GT)** | math / code（可验证）|
| CRISP | 静态 "be concise" 指令 | reasoning 压缩 |
| OEL / Skill-SD | 外部累积的 skill library | tool agent（有 verifier）|
| SDPO | runtime verifier 反馈 | 有 ground-truth 可比对 |

**Gap**：**社交 agent 场景（典型如 SOTOPIA）没人做 OPD** —— 因为既没 GT 答案（"这句话该怎么说"没有对错），也没 verifier（社交结果要人或 LLM judge 事后评）。这一块是个清晰的空白窗口。

更具体的 gap：
- 🔴 **没有 privileged context 设计 for 社交 no-verifier**：GT / skill library / verifier 这三个最常用设计都不能用
- 🔴 **反思（reflection）作为训练信号在"same-model + on-policy"下没人做过**：PreFlect 是 offline，SaMuLe 用独立 reflector，ERL 用 retrieval —— 没有"学生自己反思当自己 teacher"的纯 self-distill
- 🔴 **没有"课程式回撤 privileged context"**：训练早期 teacher 给拐杖，后期把拐杖撤走逼学生内化，这个思路在 vision (Progressive Privileged KD, TPAMI 2022) 证过有效，但 LLM 上完全没人做

### §0.2 我打算做什么（Method 总览）

**名字**: **RC-OPD** (Retrospective-Curriculum On-Policy Distillation) for SOTOPIA no-verifier social agents.

**一句话**: 把 student 自己生成的**分层事后反思**当作 teacher 的 privileged context，用 per-token KL 蒸馏回 student，训练过程中按 curriculum 逐步把反思信号**收走**，逼 student 内化反思能力。

**训练一轮（k = 1..K）**：

```
① Student rollout：student 在 SOTOPIA 跑一个对话 episode，query-only，不给任何帮助

② Hierarchical self-reflection（同一个模型，切到反思模式）：
   r_step    = "每一步哪里没推进目标？"      ← 细粒度
   r_turn    = "每 turn 的社交策略对不对？"  ← 中粒度
   r_episode = "整体目标达成没？"            ← 粗粒度

③ Critique quality filter：用模型自评 confidence，丢掉低质量反思

④ Curriculum schedule α(k) ∈ [0,1]：
   早期（α≈1）: teacher 看到全部三层反思
   中期: 逐步撤掉 step-level → turn-level
   晚期（α≈0）: teacher 只看 episode-level 或什么都不看
   （retraction = 从"多拐杖"到"无拐杖"的过渡）

⑤ Teacher forward: π_T(·| query, student_rollout, privileged_context(α_k))

⑥ Loss = Σ_t D_KL(π_T ‖ π_θ) + λ·quality_regularizer
```

**为什么每个部分都必要**：
- **Same-model（不用外 teacher/外 reflector）** → 便宜、可扩展、不依赖更强模型
- **Hierarchical 3-level** → 反思的粒度不同能捕获不同错误（step 语法错、turn 策略错、episode 目标错）
- **Curriculum retraction** → 防"学生离不开反思信号"；最终推理时 student 无需再反思就能做对
- **Quality filter** → 不是所有反思都有用，过滤掉噪声

**论文 headline question**（Codex 审后确定的一句话版本）：
> *"Can token-level RC-OPD improve a strong SOTOPIA-RL warm start **beyond inference-only reflection** and **beyond ToMA-style filter-SFT training signals**?"*

### §0.3 相关工作一张图

```
OPD 开山
  │
  ├── OPSD (2601.18734) ── GT-based privileged context ────┐
  │                                                         │
  ├── CRISP (2603.05433) ── 静态 prompt + teacher refresh  │  已被占满
  │                                                         │  (都在 math/code)
  ├── HDPO (2603.23871) ── failure-triggered GT ───────────┤
  │                                                         │
  ├── OEL / Skill-SD (2603.16856 / 2604.10674)             │
  │   ── 外部累积 skill library (tool agent with verifier) ┘
  │
  │                                                   ↓  ← 这里是我的空白
  │                                  ┌────────────────────────────────┐
  │                                  │  社交 no-verifier + self-reflect │
  │                                  │  + same-model + curriculum       │
  │                                  │  = RC-OPD（本工作）              │
  │                                  └────────────────────────────────┘
  │                                                   ↑
  │                                                   │ 最近邻
  │                                                   │
反思训 agent (非 OPD)                               社交 agent (非反思)
  ├── PreFlect (2602.07187)          │              ├── SOTOPIA-RL (2508.03905)
  │     offline + error modes 库     │              │     utterance-level RL
  ├── SaMuLe (2509.20562)             │              │     Qwen2.5-7B = 7.17
  │     独立 reflector LM             │              ├── ToMA (2509.22887)
  ├── ERL (2603.24639)                │              │     same-model filter-SFT on
  │     retrieval-based               │              │     ToM-lookahead trajectories
  └── CSQ (2601.00885)                │              └── iStar (2509.19199)
        self-critique + RL            │                    implicit-PRM RL
```

**我相对每条邻居的差异化（wedge）**：

| 最近邻 | 他们做的 | 我的 wedge |
|-------|---------|-----------|
| PreFlect | offline 反思 pattern 库 | on-policy 每 episode 重生成 + curriculum 回撤 |
| SaMuLe | 独立 reflector 模型 | 纯 same-model，不加额外模块 |
| ERL | retrieval 在推理时查 | training-time 内化，推理不 retrieve |
| CSQ | self-critique → RL (policy gradient) | self-critique → per-token KL（dense 信号）|
| SOTOPIA-RL | utterance-level RL reward | token-level distillation（可作其之上 refinement）|
| ToMA | ToM + filter → SFT | 反思 + per-token KL + hierarchy + curriculum |
| iStar | implicit step reward RL | distillation 不是 RL，dense vs sparse |
| Progressive Priv. KD (TPAMI 2022) | vision 领域 + 外部 teacher | LLM + **自生成反思** 当 privileged |

### §0.4 为什么我觉得能做起来

三个动机：

**动机 1 — 窗口期**: OPD 在 math/code 完成 SOTA 爆发；社交 no-verifier 明显是下一个方向。时间窗口 3-6 个月，本科生暑期恰好能抓住。

**动机 2 — 社交场景的特殊性质**: **事后反思比事中决策容易**。对话结束回看"我刚才那句话太强硬"，是同一个模型就能想明白的；让它事中做对，很难。**这个"事后-事中"的 gap 就是 privileged context 的天然来源**。

**动机 3 — 技术上三件事都单独被证过**: 
- On-policy distillation 能打（OPSD 系列）
- 反思对 agent training 有帮助（PreFlect/SaMuLe/ERL）
- Curriculum privileged retraction 在 vision 上有效（Progressive Privileged KD）

把这三件事**合到一起放在社交 no-verifier 上**，既是合理推论也是空白窗口 —— 这就是这个 idea 的立论。

### §0.5 结果预期（暑期可达）

- **主战场 SOTOPIA-hard**: 打过 SOTOPIA-RL (7.17 @ Qwen2.5-7B)，希望达到 7.5+ @ Qwen3-8B
- **消融**: 证明 hierarchy 3 层 > 单层；curriculum retraction > 常数 α；same-model > 外 teacher
- **理论**: 一条 realizability gap bound（hierarchical reflection 不劣于 single-level 反思之和）
- **论文**: NeurIPS 2026 workshop / ICLR 2027 主会 / findings 级；配合九推（2026-08~09）

---

## §0'. v3 核心变更（对比 v2）

**Codex 评 v2 = 6.1/Fail**（Claude self-7.7 被认定显著偏乐观；Scoop safety 仅 4.8 为最低维）。五项 must_fix + 三项 nice_to_have 全部纳入 v3：

| must_fix (Codex) | v3 处理位置 |
|-----------------|-----------|
| (1) Re-scope claim 到 SOTOPIA no-verifier | §2.1 主张改为"OPD for no-verifier social agents on SOTOPIA"；§2.4 AppWorld 降到 appendix sanity |
| (2) iStar 必须作 baseline | §2.4 baseline 列表加 iStar，与 SOTOPIA-RL / ToMA 并列主对标 |
| (3) 20-50 ep 早 pilot | §2.6 加 **5/05 hard checkpoint**：hierarchical reflection 在 50 ep 上必须可见 signal |
| (4) 理论降级 | §2.3 只保 realizability gap proof；bias-variance 退经验+sketch；**drop schedule convergence** |
| (5) Skill-SD 红区 | §2.4 AppWorld 降 appendix；论文 positioning 不再暗示"beat Skill-SD" |
| nice: Path β → mainline | §2.7 Path β (SOTOPIA-RL warm-start + RC-OPD) **升为 pragmatic mainline**；Path α 变"理论驱动 sibling" |
| nice: inference-only 控制 | §2.4 新加 baseline：reflection/ToM at inference only，区分"参数内化"vs"额外 context" |
| nice: scope 收窄 | §2.4 主战场只 SOTOPIA；MATH-500 仅 500 题 sanity；AppWorld → appendix |

**新发现的 scoop 威胁**（Codex 指出 + 我用 WebFetch/WebSearch 实测）:

**ToMA** (arXiv 2509.22887, "Infusing Theory of Mind into Socially Intelligent LLM Agents")
- Base: Qwen2.5-3B / **Qwen2.5-7B**（与我 Qwen3-8B 同家族，直接对比）
- Method: **same-model SFT** on ToM-lookahead-filtered successful trajectories
- SOTOPIA: **+18.9% (3B) / +6.9% (7B)**，competitive with GPT-5 nano
- **Overlap with v2 Path γ ≈ 70%**：same-model + Qwen2.5-7B base + SOTOPIA + ToM 作为 training signal
- **Path γ 残余 wedge（v3 reframe）**: ToMA 是 filter→SFT；RC-OPD-ToM 要做 **per-token KL 蒸馏 + hierarchical ToM levels + curriculum retraction**。wedge 变窄但仍可区分。
- **v3 必须把 ToMA 加 baseline**；Path γ 不能再自称 "blank territory"

---

## 0. v2 scoop 地图（保留为历史记录）

v2 针对性 WebSearch 追加发现 **6 篇** v1 漏掉的工作，大多集中在 "reflection-based agent training" 和 "curriculum KD" 两个子方向：

| 论文 | arXiv | 占的位置 | 对原 Idea 的影响 |
|------|-------|---------|----------------|
| **PreFlect** | 2602.07187 | Offline distillation 提取 error modes + success patterns | ⚠️⚠️ RT-OPD 被削弱（但 offline vs on-policy 仍有 wedge）|
| **SaMuLe** | 2509.20562 | 训练独立 retrospective LM 改进 agent | ⚠️ 用独立 reflector 模块（RT-OPD 是 same-model，wedge 在）|
| **ERL** | 2603.24639 | Trajectory → reusable heuristics + retrieval | ⚠️ Retrieval-based（RT-OPD 是 training-time distill，wedge 在）|
| **ProductResearch** | 2602.23716 | Multi-agent synthetic trajectory distillation + reflective internalization | ⚠️ E-commerce 域、多 agent（SOTOPIA 场景无冲突）|
| **Synthetic Self-Reflected Trajectories + Partial Masking** | 2505.20023 | 合成反思轨迹训 agent | ⚠️ Offline synthetic，非 on-policy |
| **POCL** "Being Strong Progressively" | 2506.05695 | Progressive overload curriculum for KD | ⚠️⚠️ **直接占据 curriculum KD** |
| **Progressive Privileged KD** | TPAMI 2022 (Vision) | Vision 领域的 progressive privileged KD | ⚠️⚠️ **占据 "progressive privilege" 这个名字** |
| **Scheduled Checkpoint Distillation** | 2601.10114 | Scheduled teacher convergence | ⚠️ 和 curriculum 思想重叠 |
| **SOTOPIA-RL** (已知) | 2508.03905 | Qwen2.5-7B + utterance-level reward on SOTOPIA | ✅ **正是我要对标的 baseline**，7.17 score on Sotopia-hard |
| **ToMA** (v3 新加) | 2509.22887 | Qwen2.5-7B + SOTOPIA + same-model SFT on ToM-lookahead trajectories | ⚠️⚠️⚠️ **直接冲击 Path γ**，+6.9% on Qwen2.5-7B |

**影响**（v3 重评）：
- 原 Idea A (RT-OPD)：scoop 风险 中 → **高**（reflection cluster 过热 + ToMA 在 SOTOPIA 上已站队）
- 原 Idea B (PPC-OPD)：scoop 风险 低 → **中**（curriculum KD 已被占）
- Path γ (ToM privileged): scoop 风险 低 → **中高**（ToMA 占据 70%）

---

## 1. 原 idea 打分（保留历史）

| 维度 | Idea A (RT-OPD) | Idea B (PPC-OPD) |
|------|----------------|------------------|
| Overall | **6.6 ❌** | **6.6 ❌** |

结论：两个 idea 都**没达到** CCB Rubrics 7.0 门槛。合并 → v2 RC-OPD，但 v2 也被 Codex 评为 6.1 Fail；v3 按 must_fix 重新定位。

---

## 2. v3 推荐：**RC-OPD** — Retrospective-Curriculum OPD for **SOTOPIA No-Verifier Social Agents**

> **v3 scope 收窄**（Codex must_fix 1）: 不再自称通用 agentic OPD 方法；论文主张严格限定为 **"在 SOTOPIA 这种 no-verifier social agent 场景下的 OPD 方法"**。AppWorld 只作 appendix sanity，math 只作退化检查。

> **v3 Path β one-sentence headline**（Codex minor_polish "cheapest improvement > 7.2"）:
> *"Can token-level RC-OPD improve a strong SOTOPIA-RL warm start beyond inference-only reflection and beyond ToMA-style filter-SFT training signals?"*

### 2.1 核心 Pipeline（保留 v2 数学描述，scope 限定在 SOTOPIA）

```
训练阶段 k = 1..K （K≤3 for 暑期时间线）：
  ① Student rollout on SOTOPIA: ŷ ~ π_θ(·|x)          [query only, 无 GT]

  ② Hierarchical reflection (same model, reflection mode)：
     r_step   = π_θ(·|ŷ, "反思每一句话如何推进目标")     ← 细粒度
     r_turn   = π_θ(·|ŷ, "反思每个 turn 的社交策略")
     r_episode= π_θ(·|ŷ, "反思整个 episode 的目标达成") ← 粗粒度

  ③ Critique quality filter（按 same model perplexity / 自评）

  ④ Curriculum schedule α(k)∈[0,1]：
     privileged context c_k = compose(r_episode, r_turn, r_step, α(k))

  ⑤ Teacher forward: π_T(·|x, ŷ_{<t}, c_k)
  ⑥ Loss = Σ_t D_KL(π_T‖π_θ) + λ·quality_regularizer
```

### 2.2 相对现有工作的 wedge 矩阵（v3 更新，加 ToMA + iStar）

| 挑战来源 | 他们做的 | v3 wedge |
|---------|---------|----------|
| **ToMA** (2509.22887) ⭐ 最近邻 | Same-model + Qwen2.5-7B + SOTOPIA + ToM-lookahead **filter→SFT** | **Per-token KL 蒸馏**替 filter-SFT + **hierarchy(step/turn/ep)** + **curriculum retraction**；ToM 只是 privileged context 的一种，不是整个方法 |
| **iStar** (2509.19199) ⭐ no-verifier RL 邻居 | Agentic RL + implicit step reward（替 PRM） | **Distillation 不是 RL**；RC-OPD 是 per-token supervision（dense）vs iStar 是 step-reward RL（sparse） |
| **SOTOPIA-RL** (2508.03905) ⭐ SOTOPIA SOTA | Utterance-level reward RL；Qwen2.5-7B = 7.17 on Sotopia-hard | **OPD instead of RL**；且可组合（见 Path β 用它作 warm-start）|
| **PreFlect** (2602.07187) | Offline distillation + 固定 pattern 库 | **On-policy**，reflection 每 episode 重生成 + curriculum retraction |
| **SaMuLe** (2509.20562) | 独立 reflector LM 两模型架构 | **Same-model** 纯 self-distillation |
| **ERL** (2603.24639) | Retrieval at test-time | **Training-time 内化**，推理时不检索 |
| **CSQ** (2601.00885) | Self-critique → RL policy gradient | **Self-critique → dense KL distillation** |
| **POCL / Progressive Priv. KD** | Curriculum KD 但 teacher 是外部模型 | **Curriculum over self-generated reflection**，无外 teacher |
| **CRISP** (2603.05433) | 静态 "be concise" + teacher refresh (M=50) | **动态 curriculum retraction** 与 CRISP 正交，可组合 |

**一句 v3 positioning**：*The first on-policy self-distillation method **for SOTOPIA-style no-verifier social agents** that combines **same-model hierarchical self-reflection** as privileged context with **curriculum-based privilege retraction**, distinguished from ToMA by per-token KL supervision instead of filter-SFT, and from SOTOPIA-RL by dense token supervision instead of utterance-level rewards.*

### 2.3 分析深度（v3 降级：1 proof + 2 sketch）

**Codex must_fix 4**：降下来。v3 只承诺 **1 条严格证明 + 2 条经验/草图**：

1. **[严格] Realizability gap with hierarchical privilege**: 借 HDPO Proposition 1，证明 hierarchical reflection (step ⊕ turn ⊕ ep) 的 gap **不大于 single-level 的 gap 之和**。给 hierarchy 的理论正当性。**本科生可完成**：HDPO 的证明框架可模仿，约 1-2 周推导。
2. **[经验 + 草图] Bias-variance of critique quality filter**: 给出"confidence 阈值 τ 与 student loss 方差关系"的经验曲线 + proof sketch（不追求 closed-form）。
3. **[DROPPED] Schedule convergence rate**: v2 承诺的"shared-token-mass 在 α(k) 下单调非减"不再追求严格证明；如果实验碰巧能 derive 出来再加，不作论文核心。

### 2.4 实验设计（v3 scope 大幅收窄）

**Base**: Qwen3-8B（同 OPSD / Skill-SD 量级）。Qwen3-14B 不做（算力不够）。

**Benchmark**（v3 聚焦）：
- **主战场（100% 篇幅）**: SOTOPIA-hard + SOTOPIA-all（no-verifier social）
- **Sanity (5% 篇幅)**: MATH-500（500 题退化检查，不是主结果）
- **Appendix (5% 篇幅)**: AppWorld（仅引数字，不重新跑 Skill-SD）

**Baseline**（v3 **必做 7 个**，Codex must_fix 2 + nice to have）：
1. Qwen3-8B base (no finetune)
2. **SOTOPIA-RL** (2508.03905) — SOTOPIA no-verifier SOTA，Qwen2.5-7B=7.17
3. **ToMA** (2509.22887) — SOTOPIA ToM SOTA，+6.9% over Qwen2.5-7B【v3 新加】
4. **iStar** (2509.19199) — no-verifier implicit-PRM RL 邻居【v3 新加】
5. vanilla OPSD (2601.18734) on SOTOPIA data
6. SDPO (2601.20802) — runtime feedback 分支
7. **Inference-time-only ablation**: 测试时用 hierarchical reflection 作 CoT，不蒸馏到参数【v3 新加；区分"参数内化 vs 额外 context"】

**Ablation**（保留 v2）：
- Hierarchy level: 只 step / 只 turn / 只 episode / 全 hierarchy
- Curriculum schedule: linear / cosine / constant (α≡1, 等价于 no retraction)
- Quality filter: 有 / 无
- Teacher refresh 周期（对比 CRISP M=50）

### 2.5 v3 自 review 的 Rubrics 重打分（honest）

按 Codex 反馈校准，**同时承接所有 must_fix 后**重新估：

| 维度 | v2 Claude self | v2 Codex | v3 Codex re-review | v3 理由 |
|------|---------------|---------|-------------------|--------|
| Novelty | 8 | 5.6 | **6.3** | Codex 指出 6.5 仍偏高；核心仍是 crowded ingredients 的 recombination |
| Feasibility | 7 | 5.8 | **7.2** | 加 5/05 pilot hard checkpoint + scope 收窄显著降风险 |
| Technical Rigor | 8 | 5.9 | **6.6** | 只保 1 proof 更诚实 |
| Experimental Design | 8 | 5.9 | **7.7** | 加 iStar + ToMA + inference-only 控制后最严密 |
| Scoop Safety | 7 | 4.8 | **6.0** | ToMA 确认占部分领地；OPD-on-SOTOPIA 仍有空间 |
| Impact | 8 | 6.6 | **6.9** | scope 收窄让 claim 更扎实 |
| Clarity | 8 | 8.1 | **8.7** | re-scope + 5/05 量化 rule + Path 分层让 positioning 最清晰 |
| **Overall** | **7.7** (self) | **6.1 ❌** | **7.1 ✅** | **Pass** |

**判断**（Codex v3 确认）: v3 **narrow pass**。Codex 认为 Novelty 真实值在 **6.2-6.3**（不是我 self-est 的 6.5），gap 主要来自"Path β weakens novelty"的 tradeoff —— 换了 feasibility。如果想 margin > 7.2，唯一 cheap 的方式是**收紧 β headline 到一句话**（§2 已加）。

### 2.6 可行性与时间线（v3 加 5/05 hard pilot）

- **算力**: 1.5-2k H100h (~20-35k USD)
- **代码基**: SOTOPIA-RL / SDPO / Rethinking-OPD 均开源；Skill-SD 不再强对标所以不依赖其 code

| 时间 | 里程碑 | v3 变更 |
|------|-------|--------|
| 4/22-4/28 | positioning statement + 算力/API key 敲定 + Skill-SD 作者邮件 | 与 FEASIBILITY §4 一致 |
| 4/29-5/05 | SOTOPIA + SOTOPIA-RL + ToMA baseline 跑通，**10-ep GPT-4 judge cost pilot** | 新增 cost pilot |
| **5/05** ⭐ **HARD CHECKPOINT (Codex must_fix 3)** | **量化 stop/go rule** — 50 ep pilot 上，用 SOTOPIA 官方 judge 测 RC-OPD vs 无 reflection 的 baseline 差。**GO 条件**: Δ(goal completion) ≥ **+0.3 pp** 且 hierarchical reflection 有定性信号（随机抽 10 条人工确认 >50% 可识别为 "有用的反思"）。**STOP 条件**: Δ ≤ 0 或 reflection 完全无法解析 → 立刻切 empirical study 兜底，**不切 pure RL**（见 §2.7 a 说明） | **v3 新加，最重要** |
| 5/06-5/12 | 复现 SOTOPIA-RL 7.17 ±0.3 pp + iStar baseline 跑通 | 加 iStar |
| 5/13-5/26 | RC-OPD v1 实现 + 在 SOTOPIA-sample 跑通；ToMA baseline 并行 | 加 ToMA baseline |
| 5/27-6/16 | 主实验 SOTOPIA 全量 + 消融；MATH-500 sanity；realizability gap proof 草稿 | AppWorld 移除 |
| 6/17-6/30 | Ablation 收尾 + preliminary report；**Go/No-Go 决策** | 见 FEASIBILITY §7 |
| 7/1-7/14 | 依师兄反馈扩展；formal 分析 finalize | 只完成 1 proof |
| 7/15-8/15 | 论文写作 + 图表 | — |
| 8/16-8/31 | 投稿 + 九推并行 | — |

### 2.7 Path 组合矩阵（v3 重排：β 升为 pragmatic mainline）

**Codex nice_to_have**: Path β 对本科生暑期更适合。v3 采纳：

| Path | 名称 | 定位 | Novelty | Feasibility | Scoop | 推荐 |
|------|------|------|---------|------------|-------|------|
| **β** ⭐ NEW MAINLINE | SOTOPIA-RL warm-start + RC-OPD | **pragmatic** | 7 | **8.5** | 低 | 暑期主投 |
| **α** | Pure RC-OPD from scratch | **理论驱动 sibling** | 6.5 | 7 | 中 | 与 β 并行；如 β 成功则 α 写进 ablation |
| **γ** | ToM-augmented OPD (reframed) | **backup spinoff** | 6 | 7 | 中高 (ToMA 占 70%) | 仅当 β+α 都失败时启动 |

**详细说明**：

**Path β (pragmatic mainline)**:
- Stage 1: 用 SOTOPIA-RL 开源 pipeline 训 Qwen3-8B 到 7.17±0.3 pp
- Stage 2: 在此 warm-start policy 上 apply RC-OPD 的 hierarchical reflection + curriculum retraction
- Claim: "OPD as refinement on top of RL"；回答"RC-OPD 能否在 strong RL baseline 上 squeeze 额外 gain"
- 失败模式: warm-start 已经 converge → 额外 gain <0.3 pp → 退化为 "no gain but no harm" 的 findings paper

**Path α (theoretical sibling)**:
- 直接 Qwen3-8B → RC-OPD（无 RL warm-start）
- Claim: "pure distillation approach，证明 hierarchy + curriculum 单独就能打"
- 若 β 成功，α 成为 ablation 证据
- 若 β 失败，α 独立 carry 论文

**Path γ (reframed backup)**:
- 原 v2 自称 "blank territory" —— **不成立，ToMA 已占**
- v3 reframe: **ToM-augmented OPD distillation = filter-SFT 替换为 per-token KL + hierarchical ToM (surface / belief / intent 三级) + curriculum retraction**
- 只在 β+α 都 Fail 时启动；论文必须诚实引用 ToMA

**6/30 Go/No-Go 决策树**（Codex must_fix 3 要求）:
```
Go:      β 主指标 ≥ +0.5 pp over SOTOPIA-RL → Path β 主投
Weak:    β 打平 / α 有 gain → Path α 主投 + β 做 ablation
Fail:    β+α 都无 gain → Path γ（但要重新 scope）或兜底 empirical study
```

**(a) ⚠️ 5/05 fail 的处理（Codex v3 still_must_fix 2）**:

若 5/05 hard checkpoint 显示 hierarchical reflection 在 SOTOPIA 上根本无信号（Δ ≤ 0），**不要**改道"只做 pure RL warm-start without RC-OPD"作为本项目 continuation —— 那实际上是一个**完全不同的 paper**（复述 SOTOPIA-RL 不带新方法），**没有 novelty 可投**。此时的正确处理是：

- **立刻切兜底 empirical study paper**：把 19+15 PDF 总结 + RC-OPD 在 SOTOPIA 上的失败模式 + pilot 证据组合成 "What privileged context designs actually help in no-verifier social-agent OPD?" findings/workshop 级论文
- 或 **切 Path γ (ToM-augmented OPD)** 并接受 novelty 更窄，与 ToMA 做直接对抗实验
- **不要**在师兄面前把"跑 SOTOPIA-RL 基线"包装成 RC-OPD 的延续路径

**兜底路径**（v2 保留）:
1. **Empirical study**: "OPD for no-verifier social agents: what works and what doesn't" — 把 19+15 PDF + RC-OPD failure modes 组织成 survey+empirical paper (findings / workshop 级)
2. **Realizability gap theoretical note**: 单独把 §2.3 proof 发一个 short paper / note

---

## 3. SOTOPIA privileged context 候选（v3 更新）

| # | Privileged context | 类型 | 实现代价 | v3 评估 |
|---|------------------|-----|---------|--------|
| 1 | Opponent persona profile | Static | 低 | ⭐⭐⭐ RC-OPD 可默认加 |
| 2 | Agent's own goal hint | Static | 低 | ⭐⭐ |
| 3 | Dialogue-history summary (teacher 看压缩) | Compressed | 中 | ⭐⭐ |
| 4 | **Theory-of-Mind inference** | LLM-gen | 中 | ⭐⭐ **v2 高估，ToMA 已占大部分**；Path γ 中保留但坦诚 prior art |
| 5 | Future K-turn preview (teacher lookahead) | Traj sample | 高 | ⭐⭐⭐ 注意：ToMA 用的就是 dialogue lookahead，需区分 |
| 6 | Perspective-swap reflection | Self-gen | 中 | ⭐⭐⭐ RC-OPD r_episode 可以用这种 |
| 7 | Emotion / SOTOPIA 7-dim annotation | Structured | 低 (用 SOTOPIA-RL judge) | ⭐⭐⭐ RC-OPD r_turn 可以用 |
| 8 | Post-hoc goal-completion verification | Scalar+text | 中 | ⭐⭐⭐ r_episode 的一个 flavor |

**v3 主干选择**: r_step = #6 (perspective-swap) + r_turn = #7 (7-dim) + r_episode = #8 (goal verification)。#4 ToM 保留给 Path γ（诚实引用 ToMA）。

---

## 4. 给师兄讨论的精炼问题（v3 更新）

1. **Codex 评分**: v2 Codex 外审 = 6.1/Fail；v3 改 7.0 边缘过线。师兄觉得 v3 的调整够吗？还是需要再大改？
2. **Path β vs α**: Codex 建议 β (SOTOPIA-RL warm-start + RC-OPD) 作 pragmatic mainline。师兄同意把本科生暑期主投 Path β 吗？
3. **5/05 hard pilot**: 20-50 ep 验证 hierarchical reflection signal 的量级能不能早 detect 失败模式。师兄同意这个 hard checkpoint 吗？
4. **Theory 降级**: 只保留 realizability gap 1 条严格证明，bias-variance 退草图。师兄觉得够 rigor 吗？
5. **ToMA 对 Path γ 冲击**: ToMA 已经在 SOTOPIA 做 ToM + same-model，v3 reframe 是 "per-token KL 替 filter-SFT"。师兄觉得 wedge 够硬吗？
6. **算力 + API 预算**: 2k H100h + $500-2000 GPT-4 judge，师兄组里能 cover 吗？
7. **Skill-SD 降级**: v3 AppWorld 移除 main table，只 appendix 引数字。师兄同意这个让步吗？

---

## 5. 下一步执行

1. ✅ v3 已出；**Codex PLAN REVIEW 二次打分 = 7.1 / Pass** (2026-04-22)
2. ✅ v3 已纳入 Codex v3 review 的 2 still_must_fix（5/05 量化 rule + pure-RL 路径分离）和 4 minor_polish（β headline 收窄等）
3. 🟡 Gemini 仍未挂；若之后可用，拿 §3 privileged context 候选过一遍看是否漏了社会心理学 angle
4. **带 v3 + Codex review 双打分 (6.1 → 7.1) 去给师兄讨论**
5. 4/29 立刻 action：给 Skill-SD / ToMA (eujhwang/toma) / iStar 作者发邮件要 code；启动 SOTOPIA 本地部署

---

## 6. ⚠️ v3 局限（诚实披露）

1. **Codex 打 v3 = 7.1 narrow pass**，Novelty 真实值 6.3（不是自估 6.5）。Scoop safety 6.0 仍是最低维。
2. **ToMA 精读未做**: 只读 abstract + 搜索摘要 + 确认 GitHub 存在（github.com/eujhwang/toma, 当前 minimal）；overlap 估 70% 可能偏低
3. **iStar 代码状态未完全确认**（OpenReview 上未见明确开源链接）；邮件求 code 或重实现
4. **5/05 pilot 结果是真未知数**: 如果 hierarchical reflection 在 SOTOPIA 上信号为 0，只能走 empirical study 路线；**不要包装成 "改做 pure RL 继续"**（Codex v3 明确指出这是不同 paper，不是同 mainline）
5. **Path β 换 novelty 换了 feasibility**: "OPD on top of RL" 本身不是硬 novelty claim；论文必须把 **"RC-OPD 在 strong RL warm-start 上能 squeeze 多少额外 gain, 且贡献 > inference-only reflection + ToMA-style training signal"** 做扎实
6. **Realizability gap proof**: Codex 确认 HDPO-style comparative bound 可行；不要试图证明 "hierarchy intrinsically better than any single-level" 这种强 claim

**v3 相对 v2 最大的诚实性改进**：不再自称 "blank territory"、不再承诺 3 条 formal proof、不再伪装 generic agentic OPD、5/05 checkpoint 量化、pure-RL 不伪装成延续路径。

---

## Sources

- [PreFlect](https://arxiv.org/html/2602.07187v1)
- [SaMuLe](https://arxiv.org/html/2509.20562v1)
- [ERL](https://arxiv.org/html/2603.24639v2)
- [ProductResearch](https://arxiv.org/html/2602.23716)
- [Synthetic Self-Reflected Trajectories](https://arxiv.org/html/2505.20023)
- [POCL](https://arxiv.org/html/2506.05695v1)
- [Scheduled Checkpoint Distillation](https://arxiv.org/html/2601.10114v1)
- [SOTOPIA-RL](https://arxiv.org/html/2508.03905v1)
- [ToMA (Infusing ToM into Social LLM Agents)](https://arxiv.org/abs/2509.22887) 【v3 新加】
- [CSQ](https://arxiv.org/html/2601.00885)
- [GATES](https://arxiv.org/html/2602.20574)
- [Progressive Privileged KD (TPAMI 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0031320322002229)
- [iStar (v3 新加 baseline)](https://arxiv.org/abs/2509.19199)

# OPD 方向时间线 (19 篇, 2025-04 → 2026-04)

> 按 arXiv v1 发布日期排序。每篇给一行 one-liner + 相对你论文的位置。
> ⭐ = 必须正面对标的 baseline；🔥 = 最近 3 个月内的工作（scoop 警戒线）

---

## 2025：OPD 前夜（RL 侧 + 蒸馏侧各自演化）

| 日期 | 论文 | arXiv | one-liner | 你论文里的位置 |
|------|------|-------|----------|--------------|
| 2025-04-21 | **LUFFY** | 2504.14945 | Off-policy 专家轨迹 + on-policy rollout 动态平衡 | T3 相关 RL 方法，related work 1 段 |
| 2025-04-22 | **TTRL** | 2504.16084 | 测试时 majority-vote 作 pseudo-reward 做 RL | T3 "无 GT 场景" 对照，related work |
| 2025-05-20 | **Structured Agent Distill** | 2505.13820 | Agent 轨迹拆 REASON / ACT 两 segment 分别蒸馏 | T3 agent 蒸馏相关工作 |
| 2025-06-24 | **SRFT** | 2506.19767 | 单阶段 SFT+RL，token entropy 加权 | T3 "SFT/RL 融合"相关 |
| 2025-09-12 | **SCoRe (Distill)** | 2509.14257 | Teacher 只纠正 student 最早错的那一步，7B 追平 72B | T3 agent 蒸馏相关（撞名警告：和 Google DM 2024 SCoRe 区分）|
| 2025-09-23 | **iStar** | 2509.19199 | Agentic RL + 隐式 step reward（不要 PRM 标注） | ⭐T2 agentic RL baseline，γ-pivot 邻居 |
| 2025-10-09 | **AdaSwitch** | 2510.07842 | On/off-policy 自适应切换（粒度待查） | T2 相关方法，ablation 对照组 |

**2025 阶段特征**：
- RL 侧（LUFFY / TTRL / SRFT）在解决 "reward 稀疏" 和 "SFT vs RL 融合"
- 蒸馏侧（Structured Agent Distill / SCoRe）仍在 off-policy
- iStar 已在 agentic 场景用 implicit reward，是 γ-pivot 的直接邻居（但不是 distillation，是 RL）

---

## 2026-01：OPD 开山月

| 日期 | 论文 | arXiv | one-liner | 你论文里的位置 |
|------|------|-------|----------|--------------|
| 2026-01-26 | **OPSD (Self-Distilled Reasoner)** | 2601.18734 | 同模型 + GT 作 privileged context，per-token KL | ⭐T1 开山之作，**必引必对标** |
| 2026-01-28 | **SDPO (RL via Self-Distill)** | 2601.20802 | Runtime feedback / error trace 作 privileged context | ⭐T1，feedback 分支的 SOTA，γ-pivot 正面对手 |

**启示**：OPSD 和 SDPO 几乎同时出现（隔 2 天），走的是不同 privileged context 设计 —— 这是 OPD 方法论的"**分叉起点**"。

---

## 2026-03：OPD 爆发期（每周都有新工作）

| 日期 | 论文 | arXiv | one-liner | 你论文里的位置 |
|------|------|-------|----------|--------------|
| 🔥 2026-03-05 | **CRISP** | 2603.05433 | "Be concise" + teacher refresh (M=50) 做推理压缩 | ⭐T1，迭代 teacher refresh 的第一个规范化工作 |
| 🔥 2026-03-17 | **OEL (OPCD)** | 2603.16856 | 从 agent 轨迹提取经验 → OPCD 内化进参数 | ⭐T1 agent-OPD 先驱，Skill-SD 的前身 |
| 🔥 2026-03-25 | **HDPO** | 2603.23871 | Cliff prompts（reward=0 的题）上触发 privileged 分支 | ⭐T1 failure-triggered 分支代表 |
| 🔥 2026-03-25 | **DGO** | 2603.24093 | 经验利用 + 内化双阶段循环，GRPO + MLE 混合 | T2 hybrid 对照 |
| 🔥 2026-03-26 | **Revisiting OPD** | 2603.25562 | 诊断 3 大失败模式 + 4 个 fix（truncated KL 等） | ⭐T1 方法论，实验设计必引 |

**启示**：3 月短短 3 周出了 **5 篇 OPD 工作**，**这就是为什么你的 v1 清单会漏**（清单大多建于 3 月之前）。方向拥挤度已经到了"月更"。

---

## 2026-04：OPD 成熟期（survey、范式级工作、agentic 扩张）

| 日期 | 论文 | arXiv | one-liner | 你论文里的位置 |
|------|------|-------|----------|--------------|
| 🔥 2026-04-01 | **OPD Survey** | 2604.00626 | Tencent LLM Dept 三轴分类 + 9 open problems | ⭐T1 related work 的分类学引用 |
| 🔥 2026-04-03 | **Self-Distilled RLVR (RLSD)** | 2604.03128 | OPSD 提供 token 级 update 幅度，RLVR 给方向 | T2 相关，HDPO 同宗 |
| 🔥 2026-04-09 | **Experience Replay for LLM** | 2604.08706 | Replay buffer 挑战 "on-policy 必要性" | ⭐T3 必做 ablation baseline |
| 🔥 2026-04-12 | **Skill-SD** | 2604.10674 | Agentic OPSD + 累积 skill 记忆（AppWorld +14/+42） | ⭐T1 **最大 scoop 源**，γ-pivot 主要对手 |
| 🔥 2026-04-14 | **Rethinking OPD** | 2604.13016 | JustRL-1.5B 反向蒸馏 pre-RL 基座，给 recipe | ⭐T1 方法论，recipe 选型引用 |

**启示**：4 月前半月就出了 **5 篇**（含 1 篇 survey + 1 篇方法论总结），**说明领域正在从"新方法爆发"过渡到"综述与 recipe 期"** —— 通常这是一个方向"最成熟窗口"的信号，后续 novelty 越来越细。

---

## 趋势观察（给你论文 intro 用的论点）

### 时间维度上的三个阶段

```
2025 Q2-Q3     │ 2026-01        │ 2026-03          │ 2026-04
——————————————│————————————————│—————————————————│——————————————
RL & 传统蒸馏   │ OPD 主干出现     │ 方法爆发期         │ 综述 + 方法论期
LUFFY/TTRL/    │ OPSD + SDPO     │ CRISP/OEL/HDPO/   │ OPD Survey +
SCoRe/SRFT/    │ 两条分叉         │ DGO/Revisiting    │ Skill-SD +
iStar          │                │                  │ Rethinking OPD
```

### 按 privileged context 的 lineage

```
OPSD (GT answer)  ─┬── HDPO (failure-triggered GT)
                   ├── DGO (structured experience + GRPO)
                   ├── Self-Distilled RLVR (GT + RLVR signal)
                   │
                   └── CRISP ("be concise" 指令) ← 摆脱 GT
                           │
                           └── OEL/OPCD (extracted knowledge) ← agent 场景
                                   │
                                   └── Skill-SD (累积 skill 记忆)

SDPO (runtime feedback)  ── 独立分支，但和 OEL 的"非 GT privileged context"路线融合趋势明显

iStar (implicit PRM)     ── RL 分支的对应物
AdaSwitch (context-aware)── "如何切换 on/off" 的元方法
Exp. Replay              ── 挑战 on-policy 假设本身
```

### 方法论演进方向

1. **Privileged context 从"静态 GT"→"动态提取经验"→"累积可重用 skill"**：越来越去 GT 化
2. **Loss 粒度从"token"→"segment"→"step"**：越来越结构化
3. **迭代从"不迭代（OPSD 原版）"→"teacher refresh（CRISP）"→"deploy 循环（OEL/Skill-SD）"**：越来越贴近持续学习
4. **适用域从"math"→"code"→"multi-turn agent"→（未来）→ "no-verifier"**：你的 γ-pivot 正好卡在最后这一段空白

### Scoop 风险地图（给师兄讨论用）

| 你的 pivot | 已被占据 | 剩余窄缝 |
|----------|---------|---------|
| α: Code-OPSD | SDPO + GATES + π-Distill | per-failing-test token-level credit assignment |
| β: Adaptive OPX | HDPO + AdaSwitch + OPD Survey | 几乎全被占，不推荐 |
| **γ: Verifier-Guided OPSD** | HDPO / iStar | **PRM-as-fallback-privileged-context + agent 场景**（SOTOPIA/WebShop 无 GT 线） |

**γ-pivot 的窗口期预估**：基于 3-4 月月更节奏，3-6 个月内极可能被 scoop，建议 **6 月底前要有初步 benchmark 数字**。

---

## 建议的阅读/精读顺序

**第一周**（建立坐标系）：OPSD → OPD Survey → Rethinking OPD → Revisiting OPD

**第二周**（搞清 privileged context 变体）：CRISP → OEL → Skill-SD → SDPO

**第三周**（研究 RL × OPD 混合和失败模式）：HDPO → Self-Distilled RLVR → DGO

**第四周**（邻近方法 + 挑战论）：iStar → AdaSwitch → LUFFY → Exp. Replay → TTRL → 其余

---

## 文件对照表（PDF 命名 ↔ arXiv ID）

| 编号 | 文件名 | arXiv | 速查 |
|------|-------|-------|------|
| 01 | OPSD_Self-Distilled-Reasoner | 2601.18734 | T1 开山 |
| 02 | OEL_Online-Experiential-Learning | 2603.16856 | T1 agent 先驱 |
| 03 | CRISP_Compressed-Reasoning | 2603.05433 | T1 迭代 refresh |
| 04 | DGO_Dual-Guidance | 2603.24093 | T2 |
| 05 | Skill-SD_Skill-Conditioned | 2604.10674 | ⭐ T1 最大 scoop 源 |
| 06 | HDPO_Hybrid-Distillation-PO | 2603.23871 | T1 failure-triggered |
| 07 | SDPO_RL-via-Self-Distillation | 2601.20802 | T1 feedback 分支 |
| 08 | Self-Distilled-RLVR | 2604.03128 | T2 |
| 09 | Revisiting-OPD | 2603.25562 | T1 方法论 |
| 10 | Rethinking-OPD | 2604.13016 | T1 方法论 |
| 11 | OPD-Survey | 2604.00626 | T1 分类学 |
| 12 | iStar_Agentic-RL-Implicit-PRM | 2509.19199 | T2 γ-pivot 邻居 |
| 13 | AdaSwitch | 2510.07842 | T2 |
| 14 | LUFFY_Off-Policy-Guidance | 2504.14945 | T3 |
| 15 | TTRL_Test-Time-RL | 2504.16084 | T3 |
| 16 | SRFT_Supervised-RL-FT | 2506.19767 | T3 |
| 17 | SCoRe_Reinforced-Distillation | 2509.14257 | T3（撞名警告）|
| 18 | Structured-Agent-Distillation | 2505.13820 | T3 |
| 19 | Experience-Replay-LLM | 2604.08706 | T3 必做 ablation |

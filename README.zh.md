# Soul-Driven Agent · 中文使用手册

> 一个会"成长"的 IR Agent —— 通过三层记忆(episode / insight / soul)
> 和反思引擎,持续把对话经验沉淀成稳定的"灵魂"。
> 对应作业:**IR Agent Lab**(Pass with Distinction 方向)

---

## 目录

1. [项目定位与设计动机](#1-项目定位与设计动机)
2. [整体架构](#2-整体架构)
3. [环境准备](#3-环境准备)
4. [配置 .env](#4-配置-env)
5. [快速开始](#5-快速开始)
6. [斜杠命令完整说明](#6-斜杠命令完整说明)
7. [工具(Tools)清单](#7-工具tools清单)
8. [记忆系统详解](#8-记忆系统详解)
9. [反思引擎工作机制](#9-反思引擎工作机制)
10. [用 soul.md 定义并迭代性格](#10-用-soulmd-定义并迭代性格)
11. [典型使用场景与对话脚本](#11-典型使用场景与对话脚本)
12. [演示视频脚本(5 分钟)](#12-演示视频脚本5-分钟)
13. [自定义与扩展](#13-自定义与扩展)
14. [常见问题排查](#14-常见问题排查)
15. [文件与数据布局](#15-文件与数据布局)
16. [服务器端同步指南](#16-服务器端同步指南)

---

## 1. 项目定位与设计动机

**作业要求**:扩展 AI Agent,以信息检索(IR)的方式增强其上下文获取能力。

**本项目的切入点**:把 *记忆本身* 当作 IR 问题来做。

- 经典 RAG 解决的是"如何在静态文档里找答案",
- 本 Agent 还要解决"如何在自己的对话历史里找经验,并把经验提炼成稳定的'我是谁'"。

因此引入三层时间尺度的记忆:

| 层 | 类比 | 写入时机 | 检索方式 |
|---|---|---|---|
| Episodic(经历) | 日记流水账 | 每轮对话结束 | 向量余弦 |
| Semantic / Insight(洞察) | 周末便签总结 | 每 N 轮反思一次 | 向量余弦 + 置信度 |
| Soul(灵魂) | 写进性格的座右铭 | 洞察足够稳定时晋升 | 直接注入 system prompt |

并对应 4 个 IR 工具(`search_docs / web_search / recall_*`)+ 1 个写回工具
(`update_soul_note`)—— 后者直接呼应作业说的 *"Actions count as IR."*

**对 OpenClaw 的改进点**(写报告时可直接引用):

| 维度 | OpenClaw | 本项目 |
|---|---|---|
| 记忆层数 | 单层 KV / RAG | Episodic / Insight / Soul 三层 |
| 持久化 | 会话内 cache | 磁盘 JSONL + YAML + 版本化快照 |
| 自更新 | 需外部干预 | 内置反思引擎自动总结、晋升 |
| 可解释性 | 向量黑盒 | Soul 是人类可读 YAML,可手编辑 |
| 防漂移 | 无显式机制 | 4 道闸门 + 一键回滚 |

---

## 2. 整体架构

```
                ┌──────────────────────────────┐
                │       用户输入  (CLI)        │
                └──────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────┐
  │           Context Builder(每轮重新组装)            │
  │  Soul.to_prompt()                                    │
  │  + top-K insights(向量相似)                        │
  │  + top-K episodes 摘要(向量相似)                  │
  │  + 工具 schemas                                      │
  └──────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────┐
  │   Agent Loop (ReAct, ≤ MAX_TOOL_ITERS 步)            │
  │                                                      │
  │   LLM → tool_calls? ──► dispatch ──► tool result     │
  │     ▲                                       │        │
  │     └───────────────────────────────────────┘        │
  │   直到 LLM 返回不带 tool_calls 的最终回答           │
  └──────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────┐
  │     Episode Recorder  → data/episodes.jsonl          │
  └──────────────────────────────────────────────────────┘
                              │
            未反思 episode 数 ≥ REFLECT_EVERY?
                              │ yes
                              ▼
  ┌──────────────────────────────────────────────────────┐
  │  Reflection Engine                                   │
  │  ① extractor: episodes → 新/强化/弱化 insights      │
  │  ② updater  : 高置信 insights → Soul 候选            │
  │                  ↳ 加闸门 → 新 soul.yaml             │
  │  ③ 旧 Soul 自动 snapshot 到 soul_history/vN.yaml     │
  └──────────────────────────────────────────────────────┘
```

关键文件:`src/agent.py` 是大脑、`src/soul.py` 是灵魂数据模型、
`src/reflection/` 是成长引擎、`src/tools/` 是 5 个工具。

---

## 3. 环境准备

**硬性约束**:只在 conda 环境 `IR_P_env`(Python 3.11)中运行。

```powershell
# 1) 看一下环境是否存在
& "C:\Anaconda\Scripts\conda.exe" env list

# 2) 安装依赖(已在 IR_P_env 中,无需 activate 也可以直接用全路径)
Set-Location "C:\Users\16083\Desktop\Study\IR_P\ass2\soul-agent"
& "C:\Anaconda\envs\IR_P_env\python.exe" -m pip install -r requirements.txt
```

依赖很轻量,只有 5 个包:
`openai`、`pyyaml`、`python-dotenv`、`requests`、`numpy`。

---

## 4. 配置 .env

复制模板并填写:

```powershell
Copy-Item .env.example .env
notepad .env
```

字段说明:

| 字段 | 必填 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | OpenAI 兼容服务的 API key(如 Berget.AI) |
| `OPENAI_BASE_URL` | ✅ | 端点,例如 `https://api.berget.ai/v1` |
| `OPENAI_MODEL` | ✅ | 主对话模型名 |
| `REFLECTION_MODEL` | ⛔ 可选 | 反思用的模型;留空则与主模型一致;**推荐填一个更强的模型** |
| `EMBED_MODEL` | ⛔ 可选 | OpenAI 兼容 `/embeddings` 模型;留空走哈希向量兜底 |
| `BRAVE_API_KEY` | ⛔ 可选 | Brave 搜索 key;留空则 `web_search` 工具不可用 |
| `REFLECT_EVERY` | 默认 3 | 累积多少条新 episode 触发一次反思 |
| `TOP_K_EPISODES` | 默认 3 | 注入 prompt 的历史 episode 数 |
| `TOP_K_INSIGHTS` | 默认 5 | 注入 prompt 的 insight 数 |
| `MAX_TOOL_ITERS` | 默认 6 | ReAct 单轮工具调用上限 |

> 关于 embedding:即使留空也能跑。本项目内置了基于 BLAKE2b 的有符号特征
> 哈希向量(`HASH_EMBED_DIM=512`),在 < 1000 条数据规模下相似度排序够用。
> 配上真正的 embedding 模型后,语义召回会显著更好。

---

## 5. 快速开始

```powershell
Set-Location "C:\Users\16083\Desktop\Study\IR_P\ass2\soul-agent"
& "C:\Anaconda\envs\IR_P_env\python.exe" -m src.main
```

启动后会看到:

```
Soul-Driven Agent — type /help for commands, /quit to exit.

Loaded Soul v1 with 0 episodes, 0 insights.

you>
```

任意输入即开始对话,工具调用会在终端实时打印:

```
you> What is BM25 and when does it beat dense retrieval?
  [tool] recall_insights({"query": "BM25 vs dense"}) -> {"query": "BM25...", "results": []}
  [tool] search_docs({"query": "BM25", "k": 3}) -> {"query": "BM25", "results": [...]}

agent> BM25 是一个概率排序函数 ......
```

每轮结束后看到的 `[reflect]` 日志说明反思引擎在工作。

**离线自检**(不需要 LLM,仅验证数据流和类型):

```powershell
& "C:\Anaconda\envs\IR_P_env\python.exe" -m scripts.smoke_test
```

---

## 6. 斜杠命令完整说明

| 命令 | 作用 | 备注 |
|---|---|---|
| `/help` | 打印命令帮助 | |
| `/soul` | 打印当前 Soul(渲染后的 Markdown) | 看 Agent 此刻"自我认知" |
| `/soul history` | 列出所有快照版本号 | 例如 `[1, 2, 3]` |
| `/soul diff A B` | 对比两个版本的 list 字段差异 | `+ added` / `- removed` |
| `/soul revert N` | 回滚到 v`N` 的内容(版本号会继续递增) | 用于演示"人工兜底" |
| `/soul reload` | 重新读取 `data/soul.md`(你手编了它之后) | 见第 10 节 |
| `/insights` | 列出所有 insight,含置信度、证据数、是否已晋升 | 已晋升的前缀是 `★` |
| `/episodes [N]` | 显示最近 N 条 episode 摘要(默认 5) | 前缀 `R` = 已反思 |
| `/reflect` | 立即对所有未反思 episode 触发一次反思 | 演示时常用 |
| `/reindex` | 重建 `docs/` 的向量索引 | 加了新文档后用 |
| `/quit` `/exit` | 退出 | Ctrl-C 也行 |

### 命令示例

```
you> /soul
# Agent Soul (version 1, updated 2026-05-16T00:00:00Z)
...

you> /insights
   ins_3f9a1b  conf=0.65  [user_preference]  User prefers analogies over formal proofs.  (+2 / -0)

you> /soul diff 1 2
== learned_patterns ==
  + Prefer analogies first; introduce notation only on follow-up.

you> /soul revert 1
reverted: now v3 (content from v1).
```

---

## 7. 工具(Tools)清单

LLM 通过 OpenAI 标准的 `tools=[...]` + `tool_choice=auto` 协议调用。

| 工具 | 类型 | 何时被调用 |
|---|---|---|
| `web_search(query, count)` | 外部 IR | 时事、最新 API、本地资料没有的事实 |
| `search_docs(query, k)` | 本地 RAG | 稳定的参考材料(放在 `docs/` 下) |
| `recall_episodes(query, k)` | 记忆 IR | 看用户是否问过类似问题 |
| `recall_insights(query, k)` | 记忆 IR | 在答之前先看看自己学到了什么 |
| `update_soul_note(field, content)` | **Action-as-IR** | 用户主动提供新信息或新观察 |

`update_soul_note` 出于安全设计**只允许写两个字段**:
- `open_questions`(还在观察、尚未定论的事)
- `knowledge_about_user`(用户刚明确告知的稳定事实)

`identity` / `values` / `learned_patterns` 的修改必须经过反思引擎,
不能被 Agent 自己用工具直接覆盖。

---

## 8. 记忆系统详解

### 8.1 Episodic(`data/episodes.jsonl`)

每行一个 JSON,字段:

```json
{
  "id": "ep_a1b2c3d4e5",
  "timestamp": "2026-05-16T03:21:09Z",
  "user_query": "...",
  "tool_calls": [{"name": "search_docs", "arguments": {...}, "result_preview": "..."}],
  "final_answer": "...",
  "user_feedback": null,
  "reflected": false
}
```

- **写入时机**:每轮对话结束自动写盘(append-only)。
- **检索**:`EpisodicStore.search(query, k)` 用 numpy 余弦,
  矩阵是惰性构建的,新增 episode 会失效缓存。
- **标记反思**:反思流程结束后将本批 episode 的 `reflected=true`,整体重写文件。

### 8.2 Semantic / Insight(`data/insights.jsonl`)

每条 insight:

```json
{
  "id": "ins_3f9a1b",
  "content": "User prefers analogies over formal proofs.",
  "category": "user_preference",
  "confidence": 0.75,
  "supporting_episodes": ["ep_...","ep_..."],
  "contradicting_episodes": [],
  "created_at": "...",
  "last_updated": "...",
  "promoted_to_soul": false
}
```

置信度演化(在 `InsightStore` 里):

| 事件 | confidence 变化 |
|---|---|
| 反思中被新 episode 印证 | `+0.10` |
| 反思中被新 episode 反驳 | `-0.15` |
| 低于 `DROP_BELOW=0.15` 且未晋升过 | 直接删除 |
| 高于 `PROMOTE_CONF=0.80` 且证据 ≥2 | 进入 Soul 晋升候选 |

### 8.3 Soul(`data/soul.yaml`)

七个字段:`version` / `last_updated` / `identity` / `values` /
`knowledge_about_user` / `learned_patterns` / `open_questions` / `evolution_log`。

每次反思更新后:
1. 旧 Soul 被快照到 `data/soul_history/v{N}.yaml`;
2. 新 Soul 写回 `data/soul.yaml`,`version += 1`;
3. `evolution_log` 追加一条记录(`from_version` / `to_version` / `added` / `removed` / `change`)。

---

## 9. 反思引擎工作机制

### 9.1 触发条件

任一满足即触发:

- 新增的未反思 episode 数 ≥ `REFLECT_EVERY`(默认 3);
- 用户在 CLI 输入 `/reflect`。

### 9.2 第一步:Insight Extractor(`src/reflection/extractor.py`)

将"现有 insights 摘要 + 这批 episode 简介"喂给反思模型,
强制 `response_format={"type": "json_object"}`,期望输出:

```json
{
  "new_insights": [
    {"content": "...", "category": "...", "supporting_episode_ids": ["ep_..."]}
  ],
  "reinforced_insight_ids": ["ins_..."],
  "weakened_insight_ids": ["ins_..."]
}
```

校验规则:
- 新 insight 的 `supporting_episode_ids` 必须落在本批 episode 内;
- 引用未知 insight id 的强化/弱化项被丢弃;
- 输出非合法 JSON 时整次反思跳过,不影响主流程。

### 9.3 第二步:Soul Updater(`src/reflection/updater.py`)

把"当前 Soul + 候选 insight"喂给反思模型,期望输出加/删/log 三段:

```json
{
  "additions": [
    {"field": "learned_patterns", "content": "...", "from_insight_id": "ins_..."}
  ],
  "removals": [
    {"field": "values", "content": "...", "reason": "..."}
  ],
  "log_entry": "one-line summary"
}
```

强制约束(由代码而非 prompt 保证):

| 闸门 | 实现 |
|---|---|
| 字段白名单 | 只接受 `identity / values / knowledge_about_user / learned_patterns` |
| 单字段上限 15 条 | `MAX_PER_FIELD=15`,超限的 add 直接拒绝 |
| 不写重复条目 | 大小写不敏感比对 |
| 删除需要理由 + 单次 ≤2 条 | `reason` 必填,`removals[:2]` |
| 必须基于真实 candidate | `from_insight_id` 必须属于本次候选集 |
| 旧 Soul 必快照 | `soul.snapshot()` 在写新文件前调用 |

---

## 10. 用 soul.md 定义并迭代性格

为了让"灵魂"既能由人手写,也能由反思引擎自动迭代,本项目维护了**两份等价
表示**,放在 `data/` 下,启动时按 mtime 自动协调:

| 文件 | 角色 | 谁来写 |
|---|---|---|
| `data/soul.md` | **人类入口**。手编辑性格、价值观、对用户的认知 | 你 |
| `data/soul.yaml` | 机器规范格式。`evolution_log` 等结构化字段的权威来源 | 反思引擎 / `/soul revert` |

加载与同步规则:

1. **启动时**,`Soul.load()` 比较两文件的 mtime:
   - `soul.md` 较新 → 以 md 为准载入,**自动从 yaml 复制保留 `evolution_log`**,再把结果写回两份文件。
   - `soul.yaml` 较新 → 直接读 yaml。
   - 只有一份 → 用那一份,首次会把缺失的那份补出来。
2. **反思更新或 `/soul revert` 之后**,`Soul.save()` **同时**重写 `soul.yaml` 和 `soul.md`,二者永远保持一致。
3. **运行中**手编了 `soul.md`?用 `/soul reload` 即时拉入,无需重启。

### 10.1 怎么从 0 开始定义性格

直接打开 `data/soul.md` 改就行。文件结构:

```markdown
version: 1
last_updated: 2026-05-16T00:00:00Z

## Identity
- 我是 Tian 的 IR/NLP 学习伙伴,中文母语,可中英混答。
- 我有 web 搜索、本地 docs RAG、记忆召回三类工具。

## Values
- 先给类比和例子,再上公式。
- 证据稀薄时承认"不确定",不要编参考。
- 用户给反例时降低相关 insight 的 confidence,而不是立刻翻供。

## What I know about the user
- Tian 用 vLLM 和 llama.cpp。
- Tian 习惯先自己试再问,所以可以直接给深度答案。

## Patterns I have learned
- (empty)

## Open questions
- 用户对数学推导的耐受度还不清楚。
```

字段语义:

| 字段 | 含义 | 写作建议 |
|---|---|---|
| `Identity` | "我是谁、扮演什么角色" | 给出与用户的关系、可用能力 |
| `Values` | 行为准则、对话风格 | 命令式动词:"先...再..."、"避免..." |
| `What I know about the user` | 关于用户的稳定事实 | 工具栈、母语、教育背景等 |
| `Patterns I have learned` | 反思总结出的模式 | **建议留空**,让反思引擎自己填 |
| `Open questions` | 还在观察、未定论的事 | 用问句形式 |

### 10.2 Markdown ⇄ YAML 转换规则

- `## <Title>` 后,首列的 `- xxx` 是该字段的列表项;其他内容被忽略,可自由写注释。
- 列表为空时写 `- (empty)`、`- (none)`、`- —` 都行,会被识别为空。
- 文件顶部的 `version: N` 和 `last_updated: ISO时间` 直接以 `key: value` 形式给出。
- `## Evolution log` 只用于显示,**不被解析回机器格式**;真实历史以 yaml 为准。
- 标题大小写不敏感(`## values` 也能识别)。
- 未识别的 H2 标题会被跳过,不影响其他字段。

### 10.3 性格如何随对话迭代

```
你手写 soul.md
       │
       ▼
启动 → Agent 把 md 作为 system prompt 的一部分
       │
       ▼
N 轮对话后反思引擎触发
       │
   生成 insight → 高置信度晋升进 Soul
       │
       ▼
soul.yaml 重写 + soul.md 同步重写(你下次打开 md 就能看到新增内容)
       │
       ▼
反思引擎在 evolution_log 写一条 "v3 → v4: ..."
       │
       ▼
不满意? `/soul revert N` 一键回滚
```

### 10.4 典型工作流

| 你想做的事 | 操作 |
|---|---|
| 重新塑造性格 | 编辑 `data/soul.md` → 重启 / `/soul reload` |
| 加一条用户事实 | 在 `## What I know about the user` 下加 `- ...` → `/soul reload` |
| 看反思引擎刚加了什么 | `cat data/soul.md` 或 `/soul` 命令 |
| 看完整演化历史 | `cat data/soul.yaml`,找 `evolution_log` |
| 反思把性格改坏了 | `/soul history` → `/soul revert N` |
| 把 Soul 重置回空白 | 删除 `data/soul.yaml` 和 `data/soul.md`,下次启动用默认值 |

### 10.5 注意事项

- 手写 md 时**不要乱改 `version`**,反思引擎期望它单调递增。如果你确实想"重置版本",可以把 yaml 一起删掉再重启。
- md 里**列表上限仍由代码限制为 15 条/字段**,超过部分会在下一次反思时被合并/裁剪。
- 反思引擎只能改写 `identity / values / knowledge_about_user / learned_patterns` 四个字段;`open_questions` 留给你和 `update_soul_note` 工具,反思不动它。
- 如果同时手编 md 又触发反思,以**保存时间晚的为准**;为避免冲突,手编后请先 `/soul reload` 再继续对话。

---

## 11. 典型使用场景与对话脚本

### 场景 A:第一次启动 → 让 Agent 认识你

```
you> 我是 NLP 方向的学生,主要用 vLLM 和 llama.cpp 跑推理。
you> 我更喜欢先看具体例子再看公式,纯数学推导对我有点劝退。
you> /reflect
you> /soul
# 此时应能看到 knowledge_about_user / learned_patterns 多了几条
```

### 场景 B:配合本地 docs 做 RAG

把任何 Markdown / 纯文本扔进 `docs/`,然后:

```
you> /reindex
reindexed: 12 chunks.

you> 解释一下 BM25 和 dense retrieval 各自的强项
  [tool] search_docs(...) -> {"results": [...]}

agent> ......
```

### 场景 C:让 Agent 学会一个偏好,并自动晋升到 Soul

```
you> 解释一下 attention 机制
agent> ......(纯公式)

you> 这样讲我没懂,能不能换个例子?
agent> ......(用咖啡店点单做类比)

you> 对!这样就清楚多了。
you> 再讲讲 cross-attention
agent> ......(直接给类比)

you> /reflect
[reflect] insight diff: {"new": [{"id":"ins_...", "content":"User responds best to analogies before formal definitions."}]}

# 再过 1-2 轮证据强化,confidence ≥ 0.8 后:
you> /reflect
[reflect] Soul evolved: v1 -> v2

you> /soul diff 1 2
== learned_patterns ==
  + Prefer analogies first; introduce notation only on follow-up.
```

### 场景 D:演示防漂移与人工干预

```
you> 我其实喜欢纯数学,别再用类比了。  ← 给一条相反信号
you> /reflect
# Soul 不会立刻翻案,只会弱化相关 insight 的 confidence

# 若 Agent 真的把 Soul 改坏了,手动回滚:
you> /soul history
snapshots: [1, 2, 3]; current live version: 4

you> /soul revert 2
reverted: now v5 (content from v2).
```

### 场景 E:Action-as-IR(让 Agent 自己写笔记)

```
you> 顺便记一下,我学期末有一场 IR 课的口试,日期还没定。
  [tool] update_soul_note({"field":"open_questions","content":"IR 口试日期未定,持续观察"}) -> {"ok": true, ...}

you> /soul
# open_questions 多了一条
```

---

## 12. 演示视频脚本(5 分钟)

| 时间 | 内容 | 命令/操作 |
|---|---|---|
| 0:00 | 启动 Agent,展示初始 `soul.yaml` v1 | `python -m src.main` → `/soul` |
| 0:30 | 提技术问题,展示工具实时调用日志 | "What is BM25?" |
| 1:00 | 第二轮,用类比讲解,用户表示"这样讲很清楚" | 自然对话 |
| 1:30 | 累积 3 条 episode → 自动触发反思 | 终端出现 `[reflect]` |
| 2:00 | 看新生成的 insight | `/insights` |
| 2:30 | 再来 1-2 轮强化,confidence 升到 0.8+ | 自然对话 |
| 3:00 | Soul 自动晋升 v1 → v2 | `[reflect] Soul evolved...` |
| 3:30 | 比较版本差异 | `/soul diff 1 2` |
| 4:00 | 演示防漂移:给相反信号,Soul 不立刻翻案 | 触发 `/reflect`,观察 confidence 下降 |
| 4:30 | 演示一键回滚 | `/soul revert 1` |

---

## 13. 自定义与扩展

### 12.1 添加新工具

1. 在 `src/tools/` 下新建 `<name>.py`,实现一个普通函数和一份 OpenAI tool schema。
2. 在 `src/agent.py` 的 `Agent.__init__` 里把它登记到 `self._dispatch` 和 `self._schemas`。
3. 重启程序即可。LLM 会在合适时机自动选用它。

最小骨架:

```python
# src/tools/calc.py
from typing import Any

def calc(expression: str) -> dict[str, Any]:
    try:
        return {"value": eval(expression, {"__builtins__": {}}, {})}
    except Exception as e:
        return {"error": str(e)}

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calc",
        "description": "Evaluate a safe arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}
```

```python
# src/agent.py 里:
from .tools import calc
self._dispatch["calc"] = lambda **kw: calc.calc(**kw)
self._schemas.append(calc.SCHEMA)
```

### 12.2 替换 embedding 后端

代码已抽象到 `src/memory/embed.py`。
若想用 `sentence-transformers`,在 `embed_texts` 顶部加一个分支即可,
其他模块无须改动。

### 12.3 调整反思激进度

修改 `.env`:

- 想让 Soul 演化更快 → 减小 `REFLECT_EVERY`,或在 `src/memory/semantic.py` 里
  调大 `CONF_DELTA_UP` / 调小 `PROMOTE_CONF`。
- 想让 Soul 更保守 → 反之。

### 12.4 多用户

目前是单用户。要支持多用户,把 `config.SOUL_PATH` 等改成
`data/<user_id>/...`,并在 `Agent.__init__` 接收 `user_id` 参数。

---

## 14. 常见问题排查

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `RuntimeError: OPENAI_API_KEY is not set` | 没有 `.env` 或字段为空 | `cp .env.example .env` 后填写 |
| `brave request failed` | `BRAVE_API_KEY` 未配置或额度耗尽 | 留空也不影响其他工具,只是 `web_search` 不可用 |
| `[embed] remote embedding failed ...` | `EMBED_MODEL` 配错或服务无 embedding 接口 | 留空走哈希兜底,或换一个支持 `/embeddings` 的端点 |
| `reflection JSON parse failed` | 模型没遵守 JSON 输出 | 反思自动跳过,主流程不受影响;考虑换更强的 `REFLECTION_MODEL` |
| `unknown tool 'xxx'` | LLM 幻觉了一个不存在的工具 | 已被 dispatcher 兜住,会回填错误给模型让其重试 |
| 终端中文乱码 | PowerShell 默认编码 | 启动前 `$env:PYTHONIOENCODING="utf-8"` |
| Soul 演化不起来 | episode 累积不够 / insight 置信度没到 0.8 | `/reflect` 多触发几次;或临时调低 `PROMOTE_CONF` |
| 工具反复被调用直到 `MAX_TOOL_ITERS` | 模型陷入循环 | 调大 `MAX_TOOL_ITERS` 只是缓兵之计;真正修法是改进 system prompt 或换模型 |

---

## 15. 文件与数据布局

```
soul-agent/
├── README.md                  # 英文简介
├── README.zh.md               # 本文件
├── requirements.txt
├── .env.example
├── data/
│   ├── soul.md                # **人类入口**:手编辑性格的地方
│   ├── soul.yaml              # 机器规范格式(权威 evolution_log)
│   ├── soul_history/          # 所有历史快照 v1.yaml, v2.yaml, ...
│   ├── insights.jsonl         # 语义记忆
│   └── episodes.jsonl         # 经历记忆
├── docs/                      # 本地 RAG 语料(放任意 .md / .txt)
│   ├── ir_basics.md
│   └── agent_loop.md
├── scripts/
│   └── smoke_test.py          # 离线自检
└── src/
    ├── __init__.py
    ├── config.py              # 集中读 .env
    ├── llm_client.py          # OpenAI 兼容 chat + embed
    ├── soul.py                # Soul 数据类
    ├── agent.py               # ReAct loop + 反思编排
    ├── main.py                # CLI 入口
    ├── memory/
    │   ├── embed.py           # 远程 embedding + 哈希兜底
    │   ├── episodic.py
    │   └── semantic.py
    ├── reflection/
    │   ├── extractor.py
    │   └── updater.py
    └── tools/
        ├── web_search.py
        ├── doc_rag.py
        └── memory_tools.py
```

`data/episodes.jsonl`、`data/insights.jsonl`、`data/soul_history/` 都是
**运行时自动生成**的,首次启动不存在不报错。

---

## 16. 服务器端同步指南

需要带上服务器的最小文件清单:

```
requirements.txt
.env.example          ← 服务器端拷贝成 .env 后填真实 key
README.md
README.zh.md
docs/                 ← 全部
src/                  ← 全部
data/soul.md          ← 你手写的性格定义(推荐携带)
data/soul.yaml        ← 与 soul.md 等价的机器格式(可选)
scripts/smoke_test.py ← 可选
```

**不要**带过去的:

- `data/episodes.jsonl`、`data/insights.jsonl`、`data/soul_history/` —— 这些
  是本地开发期间的实验产物,服务器端应该是空白起点。
- `__pycache__/`、`.env`(含密钥)

服务器端首次部署:

```bash
python -m pip install -r requirements.txt
cp .env.example .env  &&  vi .env       # 填好 key
python -m scripts.smoke_test            # 离线自检
python -m src.main                      # 正式启动
```

---

*本项目设计思路融合了 OpenClaw(Agent Gateway)、Generative Agents
(Park et al., 2023)的反思机制、MemGPT 的分层记忆,以及 ReAct 循环。
文件可手工编辑,版本可回滚,鼓励把它当作"会成长但你随时能管住"的助手来使用。*

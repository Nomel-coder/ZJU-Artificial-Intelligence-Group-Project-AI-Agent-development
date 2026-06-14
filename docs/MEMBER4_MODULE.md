# Member 4 模块文档 — 林放 (3250103069)

## 项目概览

浙江大学人工智能基础 A 第13组大作业：**选题一 — AI Agent 框架搭建**。

本项目的核心思路：基于 ReAct (Reasoning + Acting) 范式，让大语言模型学会"使用工具"——
模型不仅生成文字，还能自主调用计算器、搜索百科、读写文件，并根据执行结果修正推理。

---

## 团队分工与我的模块定位

```
用户输入
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  ReActAgent (agent.py)                              │
│  孙晋荣(原始框架) + 林放(集成增强)                    │
│                                                      │
│  ┌──────────────┐   ┌──────────────┐                │
│  │ system_prompt │   │  主循环控制   │  孙晋荣 + 姚文博 │
│  │  (姚文博)     │   │  (孙晋荣)    │                │
│  └──────┬───────┘   └──────┬───────┘                │
│         │                  │                         │
│  ┌──────┴──────────────────┴───────┐                │
│  │         林放的四个模块 ⭐         │                │
│  │  ┌──────────┐ ┌──────────────┐  │                │
│  │  │ parser   │ │  history     │  │                │
│  │  │ 3策略解析 │ │  manager     │  │                │
│  │  └────┬─────┘ └──────┬───────┘  │                │
│  │       │              │          │                │
│  │  ┌────┴──────────────┴───────┐  │                │
│  │  │  context_manager + memory │  │                │
│  │  └───────────────────────────┘  │                │
│  └────────────────┬────────────────┘                │
│                   │                                  │
│  ┌────────────────┴────────────────┐                │
│  │      ToolRegistry (杨中钦)      │                │
│  │  Calculator │ Wiki │ FileIO …  │                │
│  └─────────────────────────────────┘                │
└─────────────────────────────────────────────────────┘
  │
  ▼
最终答案 → 徐杨洋(Gradio界面 + 演示视频)
```

### 各成员贡献明细

| 成员 | 角色 | 具体交付 | 我的模块是否依赖 |
|------|------|----------|:---:|
| **孙晋荣** (组长) | Agent 主框架、API 接入、主循环控制 | `agent.py` 原始框架、DeepSeek API 封装、项目统筹 | ✅ 我的增强基于其原始框架 |
| **姚文博** | ReAct Prompt 设计、工具调用格式 | System prompt、`app.py`(Gradio UI)、`main.py`(CLI) | ✅ parser 兼容其 prompt 格式 |
| **杨中钦** | 工具模块开发 | `tools/` 全部 7 个文件（Calculator / Wikipedia / FileIO 等） | ✅ parser.route_tool 委托其执行 |
| **林放（我）** | 解析器、路由、历史、上下文、记忆 | 见下方模块清单 | — |
| **徐杨洋** | 系统测试、演示视频、答辩 PPT | 测试用例设计、Gradio 部署、视频录制 | ❌ 不依赖 |

---

## 我的模块清单

| # | 模块 | 文件 | 行数 | 核心职责 |
|---|------|------|------|----------|
| 1 | **多策略解析器** | `parser.py` | ~400 | 3 策略级联解析 LLM 输出为结构化数据 |
| 2 | **上下文管理器** | `context_manager.py` | ~280 | 中英混合 token 估算 + 窗口裁剪 |
| 3 | **历史管理器** | `history_manager.py` | ~350 | 结构化 ConversationTurn 追踪 |
| 4 | **短期记忆** | `memory.py` | ~250 | 旧轮抽取式压缩 + LLM 摘要接口 |
| 5 | **增强 Agent** | `agent.py` | ~300 | 集成以上 4 模块，保持向后兼容 |
| — | **测试** | `tests/test_member4.py` | ~500 | 38 个测试用例 |
| — | **文档** | `docs/MEMBER4_MODULE.md` | 本文 | 设计说明 + 依赖关系 |

---

## 依赖关系

| 依赖谁 | 依赖什么 | 我的处理 |
|--------|----------|----------|
| **杨中钦** | `tools/` — ToolRegistry、execute_react_tool | `parser.py` 和 `agent.py` 直接 import，将解析结果委托给工具模块执行 |
| **姚文博** | System prompt 定义的 ReAct 输出格式 | `parser.py` 以该格式为解析目标，同时扩展中英文双语、XML标签、大小写变体支持 |
| **孙晋荣** | `agent.py` 原始框架（API 客户端、主循环、生成器接口） | 保持原始公共 API 不变，内部集成我的 4 个模块，新增 4 个公共方法 |
| **徐杨洋** | 无代码级依赖 | — |

---

## 各模块详细设计

### 一、多策略解析器 (`parser.py`) ⭐ 核心

**设计动机**：姚文博的原版解析器只用一个正则表达式，遇到 LLM 输出稍有不规范就失败。我设计了 3 级降级策略：

```
策略 1: 结构化标签 (<thought>, <action>)
  ↓ 失败（最大置信度 0.95）
策略 2: 增强正则 (Thought:/思考：, 大小写不敏感, 多行)
  ↓ 失败（置信度 0.75-0.85）
策略 3: 模糊启发式 (关键词 + 位置推断)
  → 兜底（置信度 0.35-0.40）
```

**与原版对比**：

| 维度 | 姚文博原版 | 我的增强版 |
|------|-----------|-----------|
| 关键词语言 | 仅英文 | 中英文双语 |
| 大小写 | 仅匹配全小写 | 大小写不敏感 |
| Action Input | 单行 | 多行 + Markdown 代码块清理 |
| 工具名校验 | 无 | 自动匹配 ToolRegistry |
| 置信度评分 | 无 | 0.0-1.0 + 策略类型标注 |
| 失败处理 | 直接报错 | 三级降级 + 模糊回退 |

**新增公共 API**：

```python
from parser import parse_react_output_v2, ParseResult

result: ParseResult = parse_react_output_v2(llm_output, tool_registry=registry)
# result.type       → "thought" | "action" | "final_answer" | "error"
# result.confidence → 0.0-1.0
# result.strategy   → "structured_tags" | "regex" | "fuzzy" | "none"
```

**向后兼容**：保留姚文博原版的 `parse_react_response(text) → dict` 和 `route_tool(action, action_input) → str` 两个函数的签名和行为，`app.py` 和 `main.py` 无需任何改动。

### 二、上下文管理器 (`context_manager.py`)

**设计动机**：DeepSeek 模型不兼容 OpenAI 的 tiktoken 库，无法精确计数 token。我设计了字符级保守估算：

| 字符类型 | 估算系数 | 依据 |
|----------|---------|------|
| 中文字符 | ×1.5 | DeepSeek 中文约 1.2-2.0 token/字，取偏保守 |
| 英文/数字 | ×0.25 | 约 4 字符/token |
| 标点/空格 | ×0.3 | 一般偏少 |

**裁剪规则**：
1. system prompt 绝不删除
2. 从最早的消息开始丢弃
3. 最后一条 user 消息（当前提问）始终保留
4. 每轮 ReAct 循环后检查，超过阈值自动裁剪

### 三、历史管理器 (`history_manager.py`)

**设计动机**：孙晋荣原始框架用扁平 `messages` 列表管理对话，无法按轮次查询、无法生成摘要、无法统计工具使用情况。

**ConversationTurn** — 每轮完整对话的结构化记录：

```
用户提问 → Thought → Action → ActionInput → Observation → Final Answer
```

支持的能力：按 turn_id 查询、转为 OpenAI 消息列表格式、生成摘要、统计工具使用频率。

### 四、短期记忆 (`memory.py`)

**设计动机**：多轮对话中 context 不断增长，超过 token 限制就需要压缩。我采用两级策略：

**默认（抽取式）**：保留最近 3 轮完整对话，更早轮次压缩为一行：
```
[历史#1] 问: 计算圆的面积... → 答: 半径为5的圆面积为78.54
[历史#2] 问: Wikipedia 圆周率... → 答: 圆周率(π)是圆的周长与直径之比...
```

**进阶（LLM 摘要）**：当使用率 > 90% 时触发，调用 DeepSeek 做深度压缩（可能额外消耗 token，所以只在极度紧张时启用）。

---

## 测试覆盖

| 测试类 | 用例数 | 覆盖内容 |
|--------|--------|----------|
| `ParserStandardFormatTests` | 4 | 标准 Thought/Action/Final Answer |
| `ParserChineseTests` | 3 | 中文标点、中英混合 |
| `ParserEdgeCaseTests` | 5 | 大小写、空输入、多行、代码块、未知工具 |
| `ParserStructuredTagTests` | 2 | XML 标签格式 |
| `ParserBackwardCompatTests` | 1 | 旧 API 返回格式 |
| `ToolRouterTests` | 5 | 计算器/文件 IO/未知工具/无效输入/格式化 |
| `ContextManagerTests` | 5 | 中英 token 估算、裁剪、安全检测 |
| `HistoryManagerTests` | 5 | 轮次管理、上限、消息列表、摘要、统计 |
| `MemoryTests` | 3 | 上下文构建、压缩、触发判断 |
| `AgentIntegrationTests` | 3 | 模块集成、reset 兼容、API 测试 |
| `ToolRegistryCompatTests` | 2 | 杨中钦工具 + 姚文博兼容函数 |
| **合计** | **38** | |

```bash
# 运行我的测试
python -m unittest discover -s tests -v
# Ran 38 tests in 0.013s — OK (skipped=3, 需 DeepSeek API key)
```

---

## 运行环境

| 依赖 | 版本 | 用途 |
|------|------|------|
| `openai` | ≥1.0 | DeepSeek API 调用（孙晋荣模块需要） |
| `python-dotenv` | ≥1.0 | 读取 .env 环境变量 |
| `gradio` | ≥4.0 | Web 界面（姚文博模块需要） |
| `wikipedia` | ≥1.4 | Wikipedia 搜索（杨中钦模块需要） |

安装：`pip install -r requirements.txt`

# ZJU AI Agent Framework — ReAct 范式自主决策与工具调用系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-62%20passed-green.svg)](./tests/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

浙江大学 **人工智能基础 A**（2025-2026 春夏）第 13 组大作业。

**选题一：AI Agent 框架搭建 —— 从"会说话"到"能做事"**

---

## 📖 项目简介

基于 **ReAct (Reasoning + Acting)** 范式实现的轻量级 AI Agent 框架。Agent 能够：
- 理解用户自然语言指令
- 自主推理并选择合适的工具
- 调用工具获取结果
- 基于反馈进行多轮修正
- 最终给出准确答案

### 支持的工具

| 工具 | 功能 | 实现者 |
|------|------|--------|
| `Calculator` | 安全算术表达式计算（AST 白名单，非 eval） | 杨中钦 |
| `Wikipedia_Search` | 维基百科搜索（自动中英文） | 杨中钦 |
| `File_IO` | 本地文件读写/追加（沙盒安全） | 杨中钦 |
| `Read_File` | 读取工作目录内文本文件 | 杨中钦 |
| `List_Files` | 列出目录内容 | 杨中钦 |

---

## 🏗️ 架构概览

```
用户输入
  │
  ▼
┌─────────────────────────────────────────────┐
│              ReActAgent (agent.py)           │
│       孙晋荣(框架) · 林放(解析/记忆)         │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │  parser · history · context · memory │   │
│  │     解析执行与记忆模块（林放）         │   │
│  └────────────────┬─────────────────────┘   │
│                   │                          │
│  ┌────────────────┴─────────────────────┐   │
│  │          ToolRegistry (杨中钦)        │   │
│  │    Calculator · Wiki · FileIO · ...  │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
  │
  ▼
最终答案
```

---

## 👥 团队分工

| 成员 | 职责 | 核心文件 |
|------|------|----------|
| **孙晋荣** (组长) | Agent 主框架、API 接入、项目统筹 | `agent.py`(原始框架) |
| **姚文博** | ReAct Prompt 设计、工具调用格式 | `app.py`, `main.py`, `agent.py`(prompt) |
| **杨中钦** | 工具模块开发（计算器/Wiki/文件IO） | `tools/` |
| **林放** | 解析器、Tool Router、历史/上下文/记忆管理 | `parser.py`, `history_manager.py`, `context_manager.py`, `memory.py` |
| **徐杨洋** | 系统测试、Gradio 界面、演示视频 | 测试+演示 |

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/your-org/ZJU-Artificial-Intelligence-Group-Project-AI-Agent-development.git
cd ZJU-Artificial-Intelligence-Group-Project-AI-Agent-development
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

### 4. 运行

**命令行模式：**
```bash
python main.py
```

**Web 界面（Gradio）：**
```bash
python app.py
# 浏览器打开 http://127.0.0.1:7860
```

### 5. 运行测试

```bash
python -m unittest discover -s tests -v
# 62 tests passed ✓
```

---

## 📁 项目结构

```
.
├── agent.py              # ReAct Agent 主循环（孙晋荣 + 林放增强）
├── parser.py             # 3 策略级联解析器（林放）⭐
├── context_manager.py    # Token 估算 + 上下文窗口管理（林放）
├── history_manager.py    # 结构化对话历史追踪（林放）
├── memory.py             # 短期记忆 + 抽取式压缩（林放）
├── app.py                # Gradio Web 界面（姚文博）
├── main.py               # CLI 命令行界面（姚文博）
├── tools/                # 工具模块（杨中钦）
│   ├── __init__.py       # 公共 API + create_default_registry()
│   ├── base.py           # BaseTool / ToolSpec / ToolResult
│   ├── calculator.py     # AST 安全计算器
│   ├── wikipedia_search.py  # Wikipedia 搜索
│   ├── file_io.py        # 沙盒文件读写
│   ├── registry.py       # 工具注册与调度
│   └── react_compat.py   # Member2 兼容层
├── tests/
│   ├── test_tools.py     # 工具模块测试 24 个（杨中钦）
│   └── test_member4.py   # 林放模块测试 38 个
├── docs/
│   ├── plan.md           # 开题报告
│   └── MEMBER4_MODULE.md # 林放模块设计文档
├── .env.example          # 环境变量模板
├── .gitignore
├── requirements.txt
└── README.md
```

---

<details>
<summary><b>🔬 林放模块亮点</b>（点击展开）</summary>

### 多策略解析器 (`parser.py`)

采用 3 策略级联保证鲁棒性：

```
策略 1: 结构化标签 (<thought>, <action>)
  ↓ 失败
策略 2: 增强正则 (Thought:/思考：, 大小写不敏感, 多行)
  ↓ 失败
策略 3: 模糊启发式 (关键词 + 位置推断)
```

- 支持中英文双语关键词
- ParseResult 含置信度评分 (0-1)
- 完全向后兼容原有接口

### 上下文管理 (`context_manager.py`)

- 中英文混合 token 保守估算
- 滑动窗口裁剪，始终保留 system prompt
- 适应 DeepSeek 等非 OpenAI tokenizer

### 记忆模块 (`memory.py`)

- 抽取式压缩：最近 N 轮完整保留，旧轮压缩为一行摘要
- 可选 LLM 深度摘要（上下文使用率 > 90% 时触发）

</details>

---

## 📊 测试

```bash
$ python -m unittest discover -s tests -v
----------------------------------------------------------------------
Ran 62 tests in 1.619s
OK (skipped=1)
```

---

## 📝 实验报告与交付物

- 源代码：本仓库
- 开题报告：`docs/plan.md`
- 模块文档：`docs/MEMBER4_MODULE.md`
- 演示视频：（徐杨洋负责）
- 答辩 PPT：（徐杨洋负责）

---

## 📄 License

MIT License — 仅供课程学习使用

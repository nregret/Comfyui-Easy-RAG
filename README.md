<div align="center">

# 🧠 ComfyUI Easy RAG

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Extension-blue.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![i18n: EN/ZH](https://img.shields.io/badge/i18n-EN%20%7C%20ZH-orange.svg)](#)

*为 ComfyUI 带来极其简单、高效且强大的本地 RAG（检索增强生成）与大语言模型工作流体验。*

</div>

---

## 🌟 核心特性 (Key Features)

*   **📄 多格式文档解析**：一键无缝加载并解析 `TXT`, `MD`, `JSON`, `PDF` 等常见文档格式。
*   **🧠 高效本地向量引擎**：基于 `FAISS` 的强大向量检索，完全本地化运行，百分百保护您的数据隐私。
*   **🤖 双驱动大模型接入**：
    *   **本地私有部署 (LM Studio)**：无缝对接本地多模态大模型，支持完全离线运行与视觉能力。
    *   **云端强大算力 (External API)**：原生支持 OpenAI 兼容格式的外部 API（如 DeepSeek, GPT-4o, Claude 等），享受顶级逻辑推理能力。
*   **⚡ 极致的显存优化 (VRAM Optimization)**：专为 ComfyUI 环境定制的显存调度机制。在检索和文本生成结束后，**自动卸载**语言模型与 Embedding 模型，确保您后续的图像生成工作流（如 SDXL, FLUX）拥有充足显存。
*   **🗂️ 预制语料系统**：支持从特定系统目录极速加载海量预制文本、设定集与数据集。
*   **🌍 原生国际化支持**：严格遵循 ComfyUI 官方 i18n 标准，内置完美的中文（简体）与英文界面双语切换。

---

## 🧩 节点一览 (Nodes Overview)

本插件提供了一套完整的从“数据输入”到“检索生成”的节点链：

| 节点名称 (Node Name) | 功能描述 (Description) |
| :--- | :--- |
| 📄 `EasyRAG - Document Loader` | 核心文档加载器。支持上传和解析 input 目录下的多种格式文档。 |
| 📚 `Rag 预制文档加载` | 从系统配置的原始语料库（Original Corpus）中批量加载数据。 |
| 🏗️ `EasyRAG - Vector Store Builder (FAISS)` | 将文档切片（Chunking）并使用 Embedding 模型构建、持久化本地向量数据库。 |
| 💬 `EasyRAG - LM Studio API (Advanced)` | 调用本地 LM Studio 服务的高级问答节点，支持 RAG 检索、多模态图片输入与精细参数调节。 |
| 💬 `EasyRAG - LM Studio API (Simple)` | 极简版的本地对话节点，适合快速搭建轻量级工作流。 |
| ☁️ `EasyRAG - 外部 API (高级)` | **[NEW]** 通过 API Key 调用云端大模型（DeepSeek等），将本地向量库的知识与云端最强 AI 结合。 |

---

## 🛠️ 安装指南 (Installation)

### 1. 克隆项目
将本仓库克隆到你的 ComfyUI 自定义节点目录中：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/Comfyui-Easy-RAG.git
```

### 2. 安装依赖
进入插件目录并安装所需的 Python 依赖：
```bash
cd Comfyui-Easy-RAG
pip install -r requirements.txt
```
*(核心依赖包含：`faiss-cpu`, `sentence-transformers`, `requests`, `pypdf`)*

### 3. 重新启动
重启你的 ComfyUI 即可在节点菜单 `RagPrompt` 分类下找到所有功能。

---

## 📂 目录与模型准备 (Setup & Directories)

为了让插件发挥最大威力，请按照以下结构放置你的模型和数据：

### 1. 向量模型 (Embedding Models)
文本转换为向量必需的核心模型。请将其放置在 ComfyUI 的 `embeddings` 目录下（推荐使用 `bge-small-zh-v1.5` 等优秀的轻量级模型）。
```text
ComfyUI/models/embeddings/
 └── bge-small-zh-v1.5/
     ├── config.json
     ├── pytorch_model.bin
     └── ...
```

### 2. 向量数据库 (Vector Database)
此目录由插件自动生成和管理，无需手动干预。你构建的索引将永久保存在：
```text
ComfyUI/models/RAG/VectorDB/<你的索引名称>/
```

### 3. 预制语料库 (Original Corpus)
供 `Rag 预制文档加载` 节点使用，方便你统一管理小说、设定集、提示词库等：
```text
ComfyUI/models/RAG/Original/
```

### 4. 系统提示词预设 (System Prompts)
你可以将常用的系统提示词（`txt` 或 `md` 格式）放入以下目录，即可在对话节点的下拉菜单中一键调用：
```text
ComfyUI/custom_nodes/Comfyui-Easy-RAG/systemprompt/
```

---

## 🚀 快速上手 (Quick Start)

构建一个完整的 RAG 对话工作流仅需 4 步：

1. **加载文档**：使用 `Document Loader` 节点选择你的 PDF 或 TXT 文件。
2. **构建向量库**：将文档连接至 `Vector Store Builder`，选择本地 Embedding 模型并为索引命名。
3. **连接大模型**：将生成的 `RAG 索引` 输出连接到 `外部 API (高级)` 或 `LM Studio API` 节点。
4. **开始对话**：在对话节点中输入你的问题（如：“总结这份文档的核心观点”），点击 Queue Prompt 运行！

### 💡 搭配 LM Studio 使用的提示：
*   请确保 LM Studio 已经在本地运行。
*   在 LM Studio 左侧菜单开启 `Local Server`（默认监听 `http://127.0.0.1:1234`）。
*   加载任意你想使用的 LLM 模型即可。

---

## 🤝 贡献与支持 (Contributing)

欢迎提交 Issues 和 Pull Requests！如果你觉得这个项目对你有帮助，请在 GitHub 上为它点亮一颗 ⭐！

**License:** MIT License

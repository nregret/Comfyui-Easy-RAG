# Comfyui-Easy-RAG

本插件为 ComfyUI 提供本地 RAG 工作流节点，支持文档构建向量库、检索增强问答，以及 LM Studio 本地多模态对话。

## 功能特性

- 文档加载与上传（`txt / md / json / pdf`）
- FAISS 向量库构建与复用
- RAG 检索 + LM Studio 对话生成
- 高级节点支持图片输入（多模态）
- 支持中文向量库名（Windows 路径兼容修复）
- 预制文档双来源加载：
  - 插件目录 `rag/`
  - `ComfyUI/models/RAG/Original/`
- i18n 多语言支持（`locales/`）

## 节点列表

- `EasyRAG - Document Loader`（`RagPromptDocumentLoader`）
- `Rag 预制文档加载`（`RagPromptPrebuiltLoader`）
- `EasyRAG - Vector Store Builder (FAISS)`（`RagPromptVectorStoreBuilder`）
- `EasyRAG - LM Studio API (Advanced)`（`RagPromptLMStudioChatAdvanced`）
- `EasyRAG - LM Studio API (Simple)`（`RagPromptLMStudioChatSimple`）

## 默认参数（当前版本）

### 向量库构建

- `chunk_size`: `4000`
- `chunk_overlap`: `0`
- `unload_embedding_model_after_build`: `true`

### LM Studio API（高级）

- `max_tokens`: `2048`
- `stream`: `true`

## 安装

1. 放到 ComfyUI 自定义节点目录：

```bash
ComfyUI/custom_nodes/Comfyui-Easy-RAG
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 重启 ComfyUI。

## 使用说明

### 方式 A：标准 RAG 流程

1. `Document Loader` 或 `预制文档加载` 输出 `documents`
2. `Vector Store Builder` 构建并输出 `rag_index`
3. 将 `rag_index` 连接到 `LM Studio API (Advanced/Simple)`
4. 输入 `question`，运行即可

### 方式 B：预制语料快速构建

- 在 `Rag 预制文档加载` 中直接选择文档或目录
- 数据来源会自动从以下路径合并显示（不显示来源前缀）：
  - 插件 `rag/`
  - `ComfyUI/models/RAG/Original/`

## LM Studio 配置

- 本地服务地址默认：

```text
http://127.0.0.1:1234
```

- 确保 LM Studio 已启动并可访问模型列表接口。

## 嵌入模型放置位置

请将本地 embedding 模型放在：

```text
ComfyUI/models/embeddings/
```

例如：

```text
ComfyUI/models/embeddings/harrier-oss-v1-0.6b/
ComfyUI/models/embeddings/bge-small-zh-v1.5/
```

## 向量库存储位置

默认存储在：

```text
ComfyUI/models/RAG/VectorDB/<index_name>/
```

目录内包含：

- `index.faiss`
- `chunks.json`
- `meta.json`

## 多语言（i18n）

本项目按 ComfyUI 官方 i18n 结构组织：

```text
locales/
  en/
    main.json
    nodeDefs.json
  zh/
    main.json
    nodeDefs.json
```

语言切换跟随 `Comfy.Locale`。

## 依赖

- `faiss-cpu>=1.8.0`
- `sentence-transformers>=3.0.0`
- `requests>=2.31.0`
- `pypdf>=4.0.0`


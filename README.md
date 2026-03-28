# ComfyUI Easy RAG (Local)

本插件为 ComfyUI 提供本地 RAG 节点，能力类似 LM Studio `rag-v1`：

- 文档上传/加载（`txt`, `json`, `md`, `pdf`）
- 自动文本切分 + `sentence-transformers` embedding
- 使用 `FAISS` 构建向量数据库
- 查询执行 `top-k` 语义检索
- 自动拼接 context 到 prompt
- 通过 LM Studio 本地 OpenAI 兼容 API 生成回答

## 节点列表

1. `EasyRAG - 文档加载`
2. `EasyRAG - 向量库构建(FAISS)`
3. `EasyRAG - LM Studio API (高级)`
4. `EasyRAG - LM Studio API (简约)`

## 安装

在当前插件目录安装依赖：

```bash
pip install -r requirements.txt
```

将本目录放入：

```text
ComfyUI/custom_nodes/comfyui-easy-rag
```

重启 ComfyUI。

## 工作流示例

### 方式 A：分步 RAG

1. 文档加载节点：
   - `document`：选择 `input` 目录中的文档（仅支持 `txt/json/md/pdf`）
   - `上传文档` 按钮：直接弹出文件选择框上传文档到 `ComfyUI/input`
2. 向量库构建节点接收 `documents`，生成 `rag_index`
   - `chunk_size` 默认 `400`
   - `chunk_overlap` 默认 `80`
   - `show_retrieval_log` 可开启检索日志，后续问答时自动输出命中与分数
   - `unload_embedding_model_after_build`（与 API 节点同风格开关）：
     - 关：保留 embedding 模型（更快）
     - 开：构建后卸载 embedding 模型（更省显存）
3. LM Studio API（高级）节点输入 `question`，并连接 `rag_index` 生成回答（无 `rag_context` 输入）
   - 可选连接 `image` 输入，走多模态问答
   - `model` 下拉会自动获取 LM Studio 可用模型列表
   - 统一使用 OpenAI 兼容接口，默认走 `/v1/responses`，不可用时自动回退 `/v1/chat/completions`
   - 输出内容字段自动适配（`content` / `reasoning_content`）
   - `stream` 支持流式返回并在控制台实时输出增量文本
   - `unload_model_after_response` 可在回答后请求卸载模型以节省显存
4. LM Studio API（简约）节点
   - 输入更少：去掉 `temperature` / `max_tokens` / `top_k` / `stream`
   - 输出只有 `answer`

### 方式 B：一体化问答

1. 文档加载 -> 向量库构建
2. 将 `rag_index` 直接连接到 `LM Studio API (高级)` 或 `LM Studio API (简约)` 节点
3. 节点会自动执行检索并把 context 拼接到 prompt 后请求 LM Studio

## LM Studio 配置

确保 LM Studio 已启动本地服务（OpenAI 兼容 API），通常地址：

```text
http://127.0.0.1:1234
```

如果 `model` 留空，节点会自动从 `/v1/models` 读取第一个可用模型。

当前实现同时支持：

1. OpenAI 兼容推理接口：`/v1/chat/completions`
2. OpenAI 兼容 Responses 接口：`/v1/responses`
3. Native v1 模型管理接口：
   - `GET /api/v1/models`（模型列表）
   - `POST /api/v1/models/unload`（卸载模型）

## 向量模型放置位置（推荐）

本插件只从 `ComfyUI/models/embeddings` 读取本地 `sentence-transformers` 模型目录，不支持其他路径或在线模型名。

示例目录：

```text
ComfyUI/models/embeddings/
  all-MiniLM-L6-v2/
    config.json
    modules.json
    ...
  bge-small-zh-v1.5/
    config.json
    modules.json
    ...
```

`向量库构建` 节点中的 `embedding_model` 下拉会自动列出这些本地目录。
如果目录为空，节点会提示你先放入本地 embedding 模型。

推荐模型（按常见用途）：

1. `BAAI/bge-small-zh-v1.5`：中文语义检索，速度快，显存/内存占用低
2. `BAAI/bge-base-zh-v1.5`：中文效果更好，资源占用高于 small
3. `sentence-transformers/all-MiniLM-L6-v2`：英文与通用场景，轻量稳定
4. `intfloat/multilingual-e5-small`：多语言场景，体积较小
5. `intfloat/multilingual-e5-base`：多语言效果更好，资源占用更高

## 向量库存储位置

默认保存在：

```text
data/faiss_indexes/<index_name>/
```

包含：

- `index.faiss`
- `chunks.json`
- `meta.json`

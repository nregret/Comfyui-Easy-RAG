# ComfyUI RAG Prompt (Local)

本插件为 ComfyUI 提供本地 RAG 节点，基于原作者 nregret/Comfyui-Easy-RAG 二次开发，优化了图像反推、显存管理、多模态交互与多语言嵌入模型支持。

功能完整兼容 LM Studio 本地 RAG 工作流，无需联网、零成本构建 AI 绘画提示词知识库。

## 优化亮点
- 修复图像反推 astype 错误，支持图片输入多模态问答
- 优化显存自动清理，模型可热卸载，不占显存
- 支持热拔插切换不同向量库
- 兼容最新多语言嵌入模型（harrier-oss 等）
- 节点命名重构，更清晰、更易用
- 保留原项目所有稳定性与兼容性

## 支持功能
- 文档上传/加载（txt, json, md, pdf）
- 自动文本切分 + sentence-transformers embedding
- 使用 FAISS 构建向量数据库
- 查询执行 top-k 语义检索
- 自动拼接 context 到 prompt
- 通过 LM Studio 本地 OpenAI 兼容 API 生成回答
- 图像反推 + 多模态对话
- 显存优化、模型热卸载

## 节点列表
1. RAG Prompt - 文档加载
2. RAG Prompt - 向量库构建(FAISS)
3. RAG Prompt - LM Studio API (高级)
4. RAG Prompt - LM Studio API (简约)

## 安装
在当前插件目录安装依赖：

```bash
pip install -r requirements.txt
将本目录放入：
text
ComfyUI/custom_nodes/ComfyUI-RAG-Prompt
重启 ComfyUI。
工作流示例
方式 A：分步 RAG
文档加载节点：
document：选择 input 目录中的文档
上传文档按钮：直接上传到 ComfyUI/input
向量库构建节点接收 documents，生成 rag_index
chunk_size 默认 400
chunk_overlap 默认 80
show_retrieval_log 可开启检索日志
构建后自动卸载模型，大幅节省显存
LM Studio API (高级) 节点输入 question，连接 rag_index 生成回答
可连接 image 输入，支持多模态问答
自动获取 LM Studio 模型列表
支持 stream 流式输出
回答后自动卸载模型，释放显存
LM Studio API (简约) 节点
极简输入，一键生成答案
方式 B：一体化问答
文档加载 → 向量库构建
将 rag_index 连接到 LM Studio API 节点
自动检索 → 自动拼接上下文 → 生成回答
LM Studio 配置
确保 LM Studio 已启动本地服务：
text
http://127.0.0.1:1234
向量模型放置位置
模型必须放在：
text
ComfyUI/models/embeddings/
例如：
plaintext
ComfyUI/models/embeddings/harrier-oss-v1-0.6b/
ComfyUI/models/embeddings/bge-small-zh-v1.5/
向量库存储位置
默认保存在：
text
data/faiss_indexes/<index_name>/
包含：
index.faiss
chunks.json
meta.json
开源声明
本项目基于 nregret/Comfyui-Easy-RAG 二次开发，感谢原作者的开源贡献。
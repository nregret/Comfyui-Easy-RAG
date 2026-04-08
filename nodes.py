import base64
import gc
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import folder_paths
import numpy as np
from PIL import Image
import torch
import comfy.model_management as model_management

from .rag_core import (
    build_faiss_index,
    default_index_root,
    extract_answer_between_newlines,
    list_lmstudio_models,
    lmstudio_chat,
    load_single_document,
    search_index,
    unload_embedding_model,
    unload_lmstudio_model,
)


SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}

# ====================== 【1】加在最上方：原作者核心显存清理函数 ======================
def _clear_vram_before_run(enabled: bool):
    if not enabled:
        return
    gc.collect()
    try:
        if hasattr(model_management, "unload_all_models"):
            model_management.unload_all_models()
        if hasattr(model_management, "cleanup_models"):
            model_management.cleanup_models(True)
        if hasattr(model_management, "soft_empty_cache"):
            model_management.soft_empty_cache()
    except:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

def _is_supported_doc_file(path: str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def _list_input_docs_for_combo() -> List[str]:
    input_dir = folder_paths.get_input_directory()
    if not os.path.isdir(input_dir):
        return [""]
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    docs = [f for f in files if _is_supported_doc_file(f)]
    docs = sorted(docs)
    return docs if docs else [""]


def _list_existing_indexes() -> List[str]:
    root = default_index_root()
    if not root.exists():
        return ["default_index"]
    indexes = []
    for item in root.iterdir():
        if item.is_dir() and (item / "index.faiss").exists():
            indexes.append(item.name)
    indexes = sorted(set(indexes))
    return indexes if indexes else ["default_index"]


def _list_local_embedding_models() -> List[str]:
    model_paths: List[str] = []
    for emb_root in folder_paths.get_folder_paths("embeddings"):
        root = Path(emb_root)
        if not root.exists():
            continue
        for p in root.iterdir():
            if not p.is_dir():
                continue
            if (p / "config.json").exists() or (p / "modules.json").exists():
                model_paths.append(str(p.resolve()))
    model_paths = sorted(set(model_paths))
    return model_paths if model_paths else [""]


def _list_lmstudio_models_for_ui() -> List[str]:
    models = list_lmstudio_models("http://127.0.0.1:1234", timeout=2)
    return models if models else [""]


def _image_tensor_to_data_url(image) -> str:
    if image is None:
        return ""
    arr = image
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


_LAST_MODEL_BY_BASE_URL: Dict[str, str] = {}


# ==============================================
# 文档加载节点（不动）
# ==============================================
class DocumentLoaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "document": (
                    _list_input_docs_for_combo(),
                    {"tooltip": "选择txt/json/md/pdf文档"}
                ),
            }
        }

    RETURN_TYPES = ("RAG_DOCUMENTS", "STRING")
    RETURN_NAMES = ("documents", "summary")
    FUNCTION = "load_documents"
    CATEGORY = "RagPrompt"

    @classmethod
    def VALIDATE_INPUTS(cls, document):
        if document:
            if not folder_paths.exists_annotated_filepath(document):
                return f"无效文档: {document}"
            if not _is_supported_doc_file(document):
                return "不支持的文件类型"
        return True

    def load_documents(self, document: str):
        # 【2】每个节点第一行：清显存
        _clear_vram_before_run(True)
        
        if not document:
            return ([], "请选择文档")
        file_path = Path(folder_paths.get_annotated_filepath(document)).resolve()
        documents = []
        errors = []
        try:
            doc = load_single_document(file_path)
            if doc.get("text"):
                documents.append(doc)
        except Exception as e:
            errors.append(str(e))
        summary = f"加载完成：{len(documents)} 个文档" if not errors else f"错误：{errors[0]}"
        
        # 末尾清理（保留）
        gc.collect()
        model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        return (documents, summary)


# ==============================================
# 向量库构建节点
# ==============================================
class VectorStoreBuilderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "documents": ("RAG_DOCUMENTS",),
                "index_list": (_list_existing_indexes(), {
                    "default": "default_index",
                    "tooltip": "选择已存在的向量库"
                }),
                "index_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "留空=使用上方选择；输入名称=新建库"
                }),
                "embedding_model": (_list_local_embedding_models(), {
                    "tooltip": "选择本地embedding模型"
                }),
                "chunk_size": ("INT", {"default": 400, "min": 100, "max": 4000, "step": 10}),
                "chunk_overlap": ("INT", {"default": 80, "min": 0, "max": 2000, "step": 10}),
                "show_retrieval_log": ("BOOLEAN", {"default": True}),
                "unload_embedding_model_after_build": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("RAG_INDEX", "STRING")
    RETURN_NAMES = ("rag_index", "summary")
    FUNCTION = "build_vector_store"
    CATEGORY = "RagPrompt"

    def build_vector_store(
        self,
        documents: List[Dict],
        index_list: str,
        index_name: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        show_retrieval_log: bool,
        unload_embedding_model_after_build: bool,
    ):
        # 【2】每个节点第一行：清显存
        _clear_vram_before_run(True)
        
        selected_model = str(embedding_model or "").strip()
        if not selected_model:
            print("❌ [RAG错误] 未选择有效的embedding模型！")
            raise ValueError("请选择有效的embedding模型")

        final_name = index_name.strip() or index_list.strip()
        index_dir = default_index_root() / final_name

        print("=" * 60)
        print(f"[RAG向量库] 处理中 | 库名称: {final_name}")
        print(f"[RAG向量库] 库路径: {index_dir}")

        if (index_dir / "index.faiss").exists():
            try:
                chunks = json.loads((index_dir / "chunks.json").read_text("utf-8"))
                cnt = len(chunks)
            except:
                cnt = 0
            info = {
                "index_name": final_name,
                "index_dir": str(index_dir),
                "embedding_model": selected_model,
                "show_retrieval_log": show_retrieval_log
            }
            summary = f"✅ 库已存在，跳过构建：{final_name}"
            print(f"✅ [RAG日志] 向量库已存在，跳过构建 | 块数: {cnt}")
        else:
            info = build_faiss_index(
                documents=documents,
                embedding_model=selected_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                index_name=final_name
            )
            cnt = info.get("chunks_count", 0)
            summary = f"✅ 构建完成：{info['index_name']}"
            print(f"🆕 [RAG日志] 新建向量库成功 | 块数: {cnt}")

        if unload_embedding_model_after_build:
            print(f"♻️ [RAG日志] 已卸载embedding模型")

        print("=" * 60)
        unload_info = unload_embedding_model(selected_model) if unload_embedding_model_after_build else None
        info["unload_embedding_model_after_build"] = bool(unload_embedding_model_after_build)
        info["embedding_unload_info"] = unload_info

        # 【5】保留原有末尾显存清理
        gc.collect()
        model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return (info, summary)


# ==============================================
# 高级对话节点
# ==============================================
class LMStudioRAGChatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "question": ("STRING", {"multiline": True}),
                "base_url": ("STRING", {"default": "http://127.0.0.1:1234"}),
                "model": (_list_lmstudio_models_for_ui(),),
                "system_prompt": ("STRING", {"multiline": True}),
                "temperature": ("FLOAT", {"default": 0.2}),
                "max_tokens": ("INT", {"default": 512}),
                "top_k": ("INT", {"default": 5}),
                "stream": ("BOOLEAN", {"default": False}),
                "unload_model_after_response": ("BOOLEAN", {"default": True}),
            },
            "optional": {"rag_index": ("RAG_INDEX",), "image": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("answer", "context_used", "raw_response")
    FUNCTION = "chat_with_rag"
    CATEGORY = "RagPrompt"

    def chat_with_rag(self, question, base_url, model, system_prompt, temperature, max_tokens, top_k, stream, unload_model_after_response, rag_index=None, image=None):
        # 【2】每个节点第一行：清显存
        _clear_vram_before_run(True)
        
        base = base_url.strip()
        models = list_lmstudio_models(base)
        chosen = model.strip() or (models[0] if models else "")

        if _LAST_MODEL_BY_BASE_URL.get(base) and _LAST_MODEL_BY_BASE_URL[base] != chosen:
            try:
                unload_lmstudio_model(base, _LAST_MODEL_BY_BASE_URL[base])
            except:
                pass

        ctx = ""
        if rag_index:
            ref = rag_index.get("index_dir") or rag_index.get("index_name")
            # 【3】加 device="cpu"
            res = search_index(ref, question, top_k=top_k, device="cpu")
            ctx = res["context"]
            # 【4】检索完强制卸载embedding模型
            try:
                unload_embedding_model(rag_index["embedding_model"])
                print("♻️ [RAG] 检索完成，已卸载embedding模型")
            except:
                pass

        img = _image_tensor_to_data_url(image)
        resp = lmstudio_chat(
            base_url=base, model=chosen,
            question=question, context=ctx, image_data_url=img,
            system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens, stream=stream
        )

        _LAST_MODEL_BY_BASE_URL[base] = chosen
        if unload_model_after_response and chosen:
            try:
                unload_lmstudio_model(base, chosen)
                _LAST_MODEL_BY_BASE_URL.pop(base, None)
            except:
                pass

        # 【5】保留原有末尾显存清理
        gc.collect()
        model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        ans = extract_answer_between_newlines(resp["answer"])
        return (ans, ctx, json.dumps(resp, ensure_ascii=False))


# ==============================================
# 简约对话节点
# ==============================================
class LMStudioRAGChatSimpleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "question": ("STRING", {"multiline": True}),
                "base_url": ("STRING", {"default": "http://127.0.0.1:1234"}),
                "model": (_list_lmstudio_models_for_ui(),),
                "system_prompt": ("STRING", {"multiline": True}),
                "unload_model_after_response": ("BOOLEAN", {"default": True}),
            },
            "optional": {"rag_index": ("RAG_INDEX",), "image": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("answer")
    FUNCTION = "chat_simple"
    CATEGORY = "RagPrompt"

    def chat_simple(self, question, base_url, model, system_prompt, unload_model_after_response, rag_index=None, image=None):
        # 【2】每个节点第一行：清显存
        _clear_vram_before_run(True)
        
        base = base_url.strip()
        models = list_lmstudio_models(base)
        chosen = model.strip() or (models[0] if models else "")
        if _LAST_MODEL_BY_BASE_URL.get(base) and _LAST_MODEL_BY_BASE_URL[base] != chosen:
            try:
                unload_lmstudio_model(base, _LAST_MODEL_BY_BASE_URL[base])
            except:
                pass

        ctx = ""
        if rag_index:
            # 【3】加 device="cpu"
            res = search_index(rag_index.get("index_dir") or rag_index.get("index_name"), question, device="cpu")
            ctx = res["context"]
            # 【4】检索完强制卸载embedding模型
            try:
                unload_embedding_model(rag_index["embedding_model"])
                print("♻️ [RAG] 检索完成，已卸载embedding模型")
            except:
                pass

        resp = lmstudio_chat(
            base_url=base, model=chosen,
            question=question, context=ctx, image_data_url=_image_tensor_to_data_url(image),
            system_prompt=system_prompt, stream=True
        )

        _LAST_MODEL_BY_BASE_URL[base] = chosen
        if unload_model_after_response and chosen:
            try:
                unload_lmstudio_model(base, chosen)
                _LAST_MODEL_BY_BASE_URL.pop(base, None)
            except:
                pass

        # 【5】保留原有末尾显存清理
        gc.collect()
        model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return extract_answer_between_newlines(resp["answer"])


# ==============================================
# 节点注册
# ==============================================
NODE_CLASS_MAPPINGS = {
    "RagPromptDocumentLoader": DocumentLoaderNode,
    "RagPromptVectorStoreBuilder": VectorStoreBuilderNode,
    "RagPromptLMStudioChatAdvanced": LMStudioRAGChatNode,
    "RagPromptLMStudioChatSimple": LMStudioRAGChatSimpleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RagPromptDocumentLoader": "RagPrompt 文档加载",
    "RagPromptVectorStoreBuilder": "RagPrompt 向量库构建(FAISS)",
    "RagPromptLMStudioChatAdvanced": "RagPrompt LM Studio 高级对话",
    "RagPromptLMStudioChatSimple": "RagPrompt LM Studio 简约对话",
}

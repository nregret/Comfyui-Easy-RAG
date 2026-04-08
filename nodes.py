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
try:
    from server import PromptServer  # type: ignore
    from aiohttp import web  # type: ignore
except Exception:
    PromptServer = None
    web = None

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
from .i18n import t


SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}

PREBUILT_SOURCE_PLUGIN = "plugin"
PREBUILT_SOURCE_ORIGINAL = "original"

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


def _list_prebuilt_docs_for_combo() -> List[str]:
    try:
        source_roots = _get_prebuilt_source_roots()
        items: List[str] = []
        seen = set()
        for source, root in source_roots.items():
            if not root.exists():
                continue
            for item in root.iterdir():
                name = item.name + "/" if item.is_dir() else item.name
                if item.is_file() and not _is_supported_doc_file(str(item)):
                    continue
                if name in seen:
                    continue
                seen.add(name)
                if item.is_dir():
                    items.append(name)
                else:
                    items.append(name)
        return sorted(items) if items else [""]
    except:
        return [""]


def _get_prebuilt_source_roots() -> Dict[str, Path]:
    plugin_rag_root = Path(__file__).resolve().parent / "rag"
    plugin_rag_root.mkdir(parents=True, exist_ok=True)

    models_dir = getattr(folder_paths, "models_dir", None)
    if models_dir:
        comfy_models_root = Path(models_dir)
    else:
        comfy_models_root = Path(__file__).resolve().parents[2] / "models"

    original_corpus_root = comfy_models_root / "RAG" / "Original"
    return {
        PREBUILT_SOURCE_PLUGIN: plugin_rag_root,
        PREBUILT_SOURCE_ORIGINAL: original_corpus_root,
    }


def _resolve_prebuilt_target(document: str) -> Path:
    raw = (document or "").strip()
    if not raw:
        raise ValueError(t("Please select a prebuilt document source."))

    source_roots = _get_prebuilt_source_roots()
    candidates: List[Path] = []
    relative = raw

    # Backward compatibility: support values like "plugin:xxx" and "original:xxx".
    if ":" in raw:
        maybe_source, maybe_relative = raw.split(":", 1)
        if maybe_source in source_roots:
            relative = maybe_relative
            root = source_roots[maybe_source]
            candidates = [root]

    # Default resolution order: plugin rag first, then models/RAG/Original.
    if not candidates:
        candidates = [source_roots[PREBUILT_SOURCE_PLUGIN], source_roots[PREBUILT_SOURCE_ORIGINAL]]

    relative = relative.lstrip("/\\")
    normalized = relative.rstrip("/\\")
    for root in candidates:
        if not root.exists():
            continue
        target = (root / normalized).resolve()
        root_resolved = root.resolve()
        if root_resolved not in target.parents and target != root_resolved:
            continue
        if target.exists():
            return target

    # If both roots are missing, give an explicit source-folder hint.
    if not source_roots[PREBUILT_SOURCE_PLUGIN].exists() and not source_roots[PREBUILT_SOURCE_ORIGINAL].exists():
        raise FileNotFoundError(
            t("Prebuilt source folder not found: {folder}", folder=str(source_roots[PREBUILT_SOURCE_ORIGINAL]))
        )

    raise FileNotFoundError(t("Invalid prebuilt path: {path}", path=raw))


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


def _register_index_list_route() -> None:
    if PromptServer is None or web is None:
        return
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return
    if getattr(instance, "_easyrag_index_route_registered", False):
        return

    @instance.routes.get("/easyrag/indexes")
    async def easyrag_list_indexes(request):
        return web.json_response({"items": _list_existing_indexes()})

    instance._easyrag_index_route_registered = True


_register_index_list_route()


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
                    {"tooltip": t("Select a document (txt/json/md/pdf). Use the Upload Document button below to put a file into the input folder first."), "label": t("document")}
                ),
            }
        }

    RETURN_TYPES = ("RAG_DOCUMENTS", "STRING")
    RETURN_NAMES = (t("documents"), t("summary"))
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
            return ([], t("Please select or upload a document in the document field (txt/json/md/pdf)."))
        file_path = Path(folder_paths.get_annotated_filepath(document)).resolve()
        documents = []
        errors = []
        try:
            doc = load_single_document(file_path)
            if doc.get("text"):
                documents.append(doc)
        except Exception as e:
            errors.append(str(e))
        summary = t("Document load complete. Total files: {total}, succeeded: {success}, failed: {failed}", total=len(documents), success=len(documents), failed=len(errors))
        
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
                "documents": ("RAG_DOCUMENTS", {"label": t("documents")}),
                "index_list": (_list_existing_indexes(), {
                    "default": "default_index",
                    "tooltip": t("Select an existing vector store"),
                    "label": t("index_list")
                }),
                "index_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": t("Leave empty to use selection above; enter a name to create a new index"),
                    "label": t("index_name")
                }),
                "embedding_model": (_list_local_embedding_models(), {
                    "tooltip": t("Select a local embedding model"),
                    "label": t("embedding_model")
                }),
                "chunk_size": ("INT", {"default": 4000, "min": 100, "max": 4000, "step": 10, "label": t("chunk_size")}),
                "chunk_overlap": ("INT", {"default": 0, "min": 0, "max": 2000, "step": 10, "label": t("chunk_overlap")}),
                "show_retrieval_log": ("BOOLEAN", {"default": True, "label": t("show_retrieval_log")}),
                "unload_embedding_model_after_build": ("BOOLEAN", {"default": True, "label": t("unload_embedding_model_after_build")}),
            }
        }

    RETURN_TYPES = ("RAG_INDEX", "STRING")
    RETURN_NAMES = (t("rag_index"), t("summary"))
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
            try:
                meta = json.loads((index_dir / "meta.json").read_text("utf-8"))
                cnt_docs = meta.get("documents_count", 0)
            except:
                cnt_docs = 0
            summary = t("Vector store built: {index_name}, documents: {documents_count}, chunks: {chunks_count}, model: {selected_model}, path: {index_dir}", index_name=final_name, documents_count=cnt_docs, chunks_count=cnt, selected_model=selected_model, index_dir=str(index_dir))
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
            summary = t("Vector store built: {index_name}, documents: {documents_count}, chunks: {chunks_count}, model: {selected_model}, path: {index_dir}", index_name=info["index_name"], documents_count=info["documents_count"], chunks_count=info["chunks_count"], selected_model=selected_model, index_dir=info["index_dir"])
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
                "question": ("STRING", {"multiline": True, "label": t("question")}),
                "base_url": ("STRING", {"default": "http://127.0.0.1:1234", "label": t("base_url")}),
                "model": (_list_lmstudio_models_for_ui(), {"label": t("model")}),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": t("You are a rigorous local RAG assistant. Prefer answering from the provided context."),
                    "label": t("system_prompt")
                }),
                "temperature": ("FLOAT", {"default": 0.2, "label": t("temperature")}),
                "max_tokens": ("INT", {"default": 2048, "label": t("max_tokens")}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "label": t("seed")}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 100, "label": t("top_k")}),
                "stream": ("BOOLEAN", {"default": True, "label": t("stream")}),
                "unload_model_after_response": ("BOOLEAN", {"default": True, "label": t("unload_model_after_response")}),
            },
            "optional": {
                "rag_index": ("RAG_INDEX", {"label": t("rag_index")}),
                "image": ("IMAGE", {"label": t("image")})
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = (t("answer"), t("context_used"), t("raw_response"))
    FUNCTION = "chat_with_rag"
    CATEGORY = "RagPrompt"

    def chat_with_rag(self, question, base_url, model, system_prompt, temperature, max_tokens, seed, top_k, stream, unload_model_after_response, rag_index=None, image=None):
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
            system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens,
            seed=seed, stream=stream
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
                "question": ("STRING", {"multiline": True, "label": t("question")}),
                "base_url": ("STRING", {"default": "http://127.0.0.1:1234", "label": t("base_url")}),
                "model": (_list_lmstudio_models_for_ui(), {"label": t("model")}),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": t("You are a rigorous local RAG assistant. Prefer answering from the provided context."),
                    "label": t("system_prompt")
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "label": t("seed")}),
                "unload_model_after_response": ("BOOLEAN", {"default": True, "label": t("unload_model_after_response")}),
            },
            "optional": {
                "rag_index": ("RAG_INDEX", {"label": t("rag_index")}),
                "image": ("IMAGE", {"label": t("image")})
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = (t("answer"),)
    FUNCTION = "chat_simple"
    CATEGORY = "RagPrompt"

    def chat_simple(self, question, base_url, model, system_prompt, seed, unload_model_after_response, rag_index=None, image=None):
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
            system_prompt=system_prompt, seed=seed, stream=True
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
# 预制文档加载节点
# ==============================================
class PrebuiltLoaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "document": (
                    _list_prebuilt_docs_for_combo(),
                    {"tooltip": t("Select a prebuilt document or folder from rag and models/RAG/Original"), "label": t("document")}
                ),
            }
        }

    RETURN_TYPES = ("RAG_DOCUMENTS", "STRING")
    RETURN_NAMES = (t("documents"), t("summary"))
    FUNCTION = "load_prebuilt"
    CATEGORY = "RagPrompt"

    def load_prebuilt(self, document: str):
        _clear_vram_before_run(True)
        if not document:
            return ([], t("Please select or upload a document in the document field (txt/json/md/pdf)."))

        try:
            target_path = _resolve_prebuilt_target(document)
        except Exception as e:
            return ([], str(e))
        
        documents = []
        errors = []
        
        files_to_load = []
        if target_path.is_dir():
            for ext in SUPPORTED_EXTENSIONS:
                files_to_load.extend(target_path.glob(f"**/*{ext}"))
        else:
            files_to_load.append(target_path)
            
        for f in files_to_load:
            try:
                doc = load_single_document(f)
                if doc.get("text"):
                    documents.append(doc)
            except Exception as e:
                errors.append(f"{f.name}: {str(e)}")
        
        summary = t("Document load complete. Total files: {total}, succeeded: {success}, failed: {failed}", total=len(documents), success=len(documents), failed=len(errors))
        if errors:
            summary += t(" (failed: {count})", count=len(errors))
            
        gc.collect()
        model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return (documents, summary)


# ==============================================
# 节点注册
# ==============================================
NODE_CLASS_MAPPINGS = {
    "RagPromptDocumentLoader": DocumentLoaderNode,
    "RagPromptPrebuiltLoader": PrebuiltLoaderNode,
    "RagPromptVectorStoreBuilder": VectorStoreBuilderNode,
    "RagPromptLMStudioChatAdvanced": LMStudioRAGChatNode,
    "RagPromptLMStudioChatSimple": LMStudioRAGChatSimpleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RagPromptDocumentLoader": t("EasyRAG - Document Loader"),
    "RagPromptPrebuiltLoader": t("Rag 预制文档加载"),
    "RagPromptVectorStoreBuilder": t("EasyRAG - Vector Store Builder (FAISS)"),
    "RagPromptLMStudioChatAdvanced": t("EasyRAG - LM Studio API (Advanced)"),
    "RagPromptLMStudioChatSimple": t("EasyRAG - LM Studio API (Simple)"),
}

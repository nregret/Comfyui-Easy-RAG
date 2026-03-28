import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import folder_paths
import numpy as np
from PIL import Image

from .rag_core import (
    build_faiss_index,
    extract_answer_between_newlines,
    list_lmstudio_models,
    lmstudio_chat,
    load_single_document,
    search_index,
    unload_embedding_model,
    unload_lmstudio_model,
)


SUPPORTED_DOC_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}


def _is_supported_doc_file(path: str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_DOC_EXTENSIONS


def _list_input_docs_for_combo() -> List[str]:
    input_dir = folder_paths.get_input_directory()
    if not os.path.isdir(input_dir):
        return [""]
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    docs = [f for f in files if _is_supported_doc_file(f)]
    docs = sorted(docs)
    return docs if docs else [""]


def _list_local_embedding_models() -> List[str]:
    # 仅从 ComfyUI/models/embeddings 读取本地向量模型
    model_paths: List[str] = []
    for emb_root in folder_paths.get_folder_paths("embeddings"):
        root = Path(emb_root)
        if not root.exists():
            continue
        for p in root.iterdir():
            if not p.is_dir():
                continue
            # sentence-transformers 常见模型目录特征文件
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


class DocumentLoaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "document": (
                    _list_input_docs_for_combo(),
                    {
                        "tooltip": "选择文档（txt/json/md/pdf）。可用下方“上传文档”按钮上传到 input 目录后再选择。"
                    },
                ),
            }
        }

    RETURN_TYPES = ("RAG_DOCUMENTS", "STRING")
    RETURN_NAMES = ("documents", "summary")
    FUNCTION = "load_documents"
    CATEGORY = "EasyRAG"

    @classmethod
    def VALIDATE_INPUTS(cls, document):
        if document:
            if not folder_paths.exists_annotated_filepath(document):
                return f"无效文档: {document}"
            if not _is_supported_doc_file(document):
                return f"不支持的文档类型: {document}"
        return True

    def load_documents(self, document: str):
        if not document:
            return ([], "请在 document 中选择或上传一个文档（txt/json/md/pdf）。")

        file_path = Path(folder_paths.get_annotated_filepath(document)).resolve()
        if not _is_supported_doc_file(str(file_path)):
            return ([], f"不支持的文档类型: {file_path.suffix}，仅支持 txt/json/md/pdf。")

        documents: List[Dict] = []
        errors: List[str] = []
        try:
            doc = load_single_document(file_path, encoding="utf-8")
            if doc.get("text"):
                documents.append(doc)
        except Exception as e:
            errors.append(f"{file_path}: {e}")

        summary = f"文档加载完成。总文件: 1, 成功: {len(documents)}, 失败: {len(errors)}"
        if errors:
            summary += "\n" + "\n".join(errors[:5])
        return (documents, summary)


class VectorStoreBuilderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "documents": ("RAG_DOCUMENTS",),
                "index_name": ("STRING", {"default": "default_index"}),
                "embedding_model": (
                    _list_local_embedding_models(),
                    {"tooltip": "仅允许使用 ComfyUI/models/embeddings 下的本地模型"},
                ),
                "chunk_size": ("INT", {"default": 400, "min": 100, "max": 4000, "step": 10}),
                "chunk_overlap": ("INT", {"default": 80, "min": 0, "max": 2000, "step": 10}),
                "show_retrieval_log": ("BOOLEAN", {"default": True, "label_on": "日志开", "label_off": "日志关"}),
                "unload_embedding_model_after_build": (
                    "BOOLEAN",
                    {"default": True, "label_on": "卸载模型", "label_off": "保留模型"},
                ),
            }
        }

    RETURN_TYPES = ("RAG_INDEX", "STRING")
    RETURN_NAMES = ("rag_index", "summary")
    FUNCTION = "build_vector_store"
    CATEGORY = "EasyRAG"

    def build_vector_store(
        self,
        documents: List[Dict],
        index_name: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        show_retrieval_log: bool,
        unload_embedding_model_after_build: bool,
    ):
        selected_model = str(embedding_model or "").strip()
        if not selected_model:
            raise ValueError(
                "未检测到可用 embedding 模型。请先把 sentence-transformers 模型放到 ComfyUI/models/embeddings。"
            )
        info = build_faiss_index(
            documents=documents,
            embedding_model=selected_model,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            index_name=index_name.strip(),
        )
        unload_info = None
        if unload_embedding_model_after_build:
            unload_info = unload_embedding_model(selected_model)
        info["show_retrieval_log"] = bool(show_retrieval_log)
        info["unload_embedding_model_after_build"] = bool(unload_embedding_model_after_build)
        info["embedding_unload_info"] = unload_info
        summary = (
            f"向量库构建完成: {info['index_name']}, 文档数: {info['documents_count']}, "
            f"chunk数: {info['chunks_count']}, 模型: {selected_model}, 目录: {info['index_dir']}, "
            f"卸载策略: {'unload' if unload_embedding_model_after_build else 'keep'}"
        )
        return (info, summary)


class LMStudioRAGChatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "question": ("STRING", {"multiline": True, "default": ""}),
                "base_url": ("STRING", {"default": "http://127.0.0.1:1234"}),
                "model": (
                    _list_lmstudio_models_for_ui(),
                    {"tooltip": "模型选项来自 LM Studio API（默认地址: 127.0.0.1:1234）"},
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "你是一个严谨的本地RAG助手，优先根据给定上下文回答。",
                    },
                ),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 16}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "stream": ("BOOLEAN", {"default": False, "label_on": "流式开", "label_off": "流式关"}),
                "unload_model_after_response": (
                    "BOOLEAN",
                    {"default": True, "label_on": "卸载模型", "label_off": "保留模型"},
                ),
            },
            "optional": {
                "rag_index": ("RAG_INDEX",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("answer", "context_used", "raw_response")
    FUNCTION = "chat_with_rag"
    CATEGORY = "EasyRAG"

    def chat_with_rag(
        self,
        question: str,
        base_url: str,
        model: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        top_k: int,
        stream: bool,
        unload_model_after_response: bool,
        rag_index=None,
        image=None,
    ):
        base = base_url.strip()
        available_models = list_lmstudio_models(base, timeout=4)
        chosen_model = (model or "").strip()
        if (not chosen_model) and available_models:
            chosen_model = available_models[0]
        elif chosen_model and available_models and chosen_model not in available_models:
            # 若节点缓存了旧模型名，自动回退到当前可用模型
            chosen_model = available_models[0]

        # 自动模型切换卸载：同一 LM Studio 地址下，模型变化时先卸载上一个
        auto_unload_before_switch = None
        prev_model = _LAST_MODEL_BY_BASE_URL.get(base, "")
        if prev_model and chosen_model and prev_model != chosen_model:
            try:
                auto_unload_before_switch = unload_lmstudio_model(base, prev_model)
            except Exception as e:
                auto_unload_before_switch = {"ok": False, "error": str(e), "instance_id": prev_model}

        context_used = ""
        show_retrieval_log = bool((rag_index or {}).get("show_retrieval_log", False))
        if rag_index is not None:
            index_ref = rag_index.get("index_dir") or rag_index.get("index_name")
            result = search_index(index_ref, query=question, top_k=int(top_k))
            context_used = result["context"]
            if show_retrieval_log:
                print(f"[EasyRAG][问答检索] question={question!r} top_k={top_k}")
                print(
                    f"[EasyRAG][问答检索] rag_hit={result.get('rag_hit')} best_score={result.get('best_score', 0.0):.4f}"
                )
                for i, item in enumerate(result.get("items", []), start=1):
                    snippet = (item.get("text", "") or "").replace("\n", " ")[:120]
                    print(
                        f"[EasyRAG][问答检索#{i}] score={item.get('score', 0.0):.4f} "
                        f"source={item.get('source', '')} text={snippet}"
                    )

        image_data_url = _image_tensor_to_data_url(image) if image is not None else ""
        response = lmstudio_chat(
            base_url=base,
            model=chosen_model,
            question=question,
            context=context_used,
            image_data_url=image_data_url,
            system_prompt=system_prompt,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            api_mode="responses",
            stream=bool(stream),
            emit_stream_log=bool(stream or show_retrieval_log),
        )
        _LAST_MODEL_BY_BASE_URL[base] = chosen_model or response.get("model", "")
        unload_status = None
        selected_model = response.get("model") or chosen_model
        if unload_model_after_response and selected_model:
            try:
                unload_status = unload_lmstudio_model(base, selected_model)
                if _LAST_MODEL_BY_BASE_URL.get(base) == selected_model:
                    _LAST_MODEL_BY_BASE_URL.pop(base, None)
            except Exception as e:
                unload_status = {"ok": False, "error": str(e)}
        extracted_answer = extract_answer_between_newlines(response["answer"])
        raw_out = {
            "chat": response["raw"],
            "selected_model": selected_model,
            "available_models": available_models,
            "auto_unload_before_switch": auto_unload_before_switch,
            "unload_requested": bool(unload_model_after_response),
            "unload_status": unload_status,
        }
        return (
            extracted_answer,
            context_used,
            json.dumps(raw_out, ensure_ascii=False, indent=2),
        )


class LMStudioRAGChatSimpleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "question": ("STRING", {"multiline": True, "default": ""}),
                "base_url": ("STRING", {"default": "http://127.0.0.1:1234"}),
                "model": (
                    _list_lmstudio_models_for_ui(),
                    {"tooltip": "模型选项来自 LM Studio API（默认地址: 127.0.0.1:1234）"},
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "你是一个严谨的本地RAG助手，优先根据给定上下文回答。",
                    },
                ),
                "unload_model_after_response": (
                    "BOOLEAN",
                    {"default": True, "label_on": "卸载模型", "label_off": "保留模型"},
                ),
            },
            "optional": {
                "rag_index": ("RAG_INDEX",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("answer",)
    FUNCTION = "chat_simple"
    CATEGORY = "EasyRAG"

    def chat_simple(
        self,
        question: str,
        base_url: str,
        model: str,
        system_prompt: str,
        unload_model_after_response: bool,
        rag_index=None,
        image=None,
    ):
        base = base_url.strip()
        available_models = list_lmstudio_models(base, timeout=4)
        chosen_model = (model or "").strip()
        if (not chosen_model) and available_models:
            chosen_model = available_models[0]
        elif chosen_model and available_models and chosen_model not in available_models:
            chosen_model = available_models[0]

        prev_model = _LAST_MODEL_BY_BASE_URL.get(base, "")
        if prev_model and chosen_model and prev_model != chosen_model:
            try:
                unload_lmstudio_model(base, prev_model)
            except Exception:
                pass

        context_used = ""
        show_retrieval_log = bool((rag_index or {}).get("show_retrieval_log", False))
        if rag_index is not None:
            index_ref = rag_index.get("index_dir") or rag_index.get("index_name")
            result = search_index(index_ref, query=question)
            context_used = result["context"]
            if show_retrieval_log:
                print(f"[EasyRAG][简约检索] question={question!r} top_k=5(default)")
                print(
                    f"[EasyRAG][简约检索] rag_hit={result.get('rag_hit')} best_score={result.get('best_score', 0.0):.4f}"
                )
                for i, item in enumerate(result.get("items", []), start=1):
                    snippet = (item.get("text", "") or "").replace("\n", " ")[:120]
                    print(
                        f"[EasyRAG][简约检索#{i}] score={item.get('score', 0.0):.4f} "
                        f"source={item.get('source', '')} text={snippet}"
                    )
        elif show_retrieval_log:
            print("[EasyRAG][简约检索] 未连接 rag_index，跳过检索。")

        image_data_url = _image_tensor_to_data_url(image) if image is not None else ""
        response = lmstudio_chat(
            base_url=base,
            model=chosen_model,
            question=question,
            context=context_used,
            image_data_url=image_data_url,
            system_prompt=system_prompt,
            api_mode="responses",
            stream=True,
            emit_stream_log=True,
        )
        _LAST_MODEL_BY_BASE_URL[base] = chosen_model or response.get("model", "")
        selected_model = response.get("model") or chosen_model
        if unload_model_after_response and selected_model:
            try:
                unload_lmstudio_model(base, selected_model)
                if _LAST_MODEL_BY_BASE_URL.get(base) == selected_model:
                    _LAST_MODEL_BY_BASE_URL.pop(base, None)
            except Exception:
                pass

        extracted_answer = extract_answer_between_newlines(response["answer"])
        return (extracted_answer,)

NODE_CLASS_MAPPINGS = {
    "EasyRAGDocumentLoader": DocumentLoaderNode,
    "EasyRAGVectorStoreBuilder": VectorStoreBuilderNode,
    "EasyRAGLMStudioChatAdvanced": LMStudioRAGChatNode,
    "EasyRAGLMStudioChatSimple": LMStudioRAGChatSimpleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyRAGDocumentLoader": "EasyRAG - 文档加载",
    "EasyRAGVectorStoreBuilder": "EasyRAG - 向量库构建(FAISS)",
    "EasyRAGLMStudioChatAdvanced": "EasyRAG - LM Studio API (高级)",
    "EasyRAGLMStudioChatSimple": "EasyRAG - LM Studio API (简约)",
}

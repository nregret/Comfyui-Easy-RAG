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

from .i18n import t
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
    entries = sorted(os.listdir(input_dir))
    docs = []
    for name in entries:
        full = os.path.join(input_dir, name)
        if os.path.isdir(full):
            # 检查目录下是否有支持的文档文件
            has_docs = any(
                f.lower().endswith(ext)
                for f in os.listdir(full)
                for ext in SUPPORTED_DOC_EXTENSIONS
            )
            if has_docs:
                docs.append(name + os.sep)
        elif _is_supported_doc_file(name):
            docs.append(name)
    return docs if docs else [""]


def _collect_supported_docs_from_dir(directory: Path) -> List[Path]:
    files: List[Path] = []
    for ext in SUPPORTED_DOC_EXTENSIONS:
        files.extend(sorted(directory.rglob(f"*{ext}")))
    return files


def _resolve_input_token_to_paths(token: str) -> List[Path]:
    raw = (token or "").strip().strip('"').strip("'")
    if not raw:
        return []
    normalized = raw.rstrip("/\\")
    if not normalized:
        return []

    input_dir = Path(folder_paths.get_input_directory()).resolve()
    candidates: List[Path] = []

    # 优先按 input 目录内相对路径解析（ComfyUI 常见用法）
    rel = (input_dir / normalized).resolve()
    candidates.append(rel)

    # 再尝试 annotated filepath 解析（兼容 ComfyUI 特殊路径格式）
    try:
        annotated = Path(folder_paths.get_annotated_filepath(normalized)).resolve()
        candidates.append(annotated)
    except Exception:
        pass

    # 最后尝试绝对路径
    p = Path(normalized)
    if p.is_absolute():
        candidates.append(p.resolve())

    out: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file() and _is_supported_doc_file(str(candidate)):
            out.append(candidate)
        elif candidate.is_dir():
            out.extend(_collect_supported_docs_from_dir(candidate))
    return out


def _resolve_document_targets(document: str, documents_text: str) -> List[Path]:
    tokens: List[str] = []
    if (document or "").strip():
        tokens.append(document.strip())
    if (documents_text or "").strip():
        normalized = documents_text.replace("\r", "\n")
        for part in normalized.replace(";", "\n").replace(",", "\n").split("\n"):
            p = part.strip()
            if p:
                tokens.append(p)

    files: List[Path] = []
    seen = set()
    for token in tokens:
        for p in _resolve_input_token_to_paths(token):
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            files.append(p.resolve())
    return files


def _infer_role(filename: str) -> str:
    """根据文件名推断文档角色，用于两阶段检索路由。"""
    fname = filename.lower()
    if "paradigm" in fname:
        return "paradigm"
    if "vocabulary" in fname or "vocab" in fname:
        return "vocabulary"
    if "example" in fname:
        return "example"
    if "system_prompt" in fname or "system-prompt" in fname:
        return "system"
    return "general"


def _load_documents_from_targets(targets: List[Path]) -> tuple[List[Dict], List[str]]:
    documents: List[Dict] = []
    errors: List[str] = []

    for p in targets:
        if not _is_supported_doc_file(str(p)):
            continue
        try:
            doc = load_single_document(p, encoding="utf-8")
            if doc.get("text") or doc.get("chunks"):
                doc["role"] = _infer_role(p.name)
                documents.append(doc)
        except Exception as e:
            errors.append(f"{p}: {e}")

    return documents, errors


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


def _clear_vram_before_run(enabled: bool) -> Dict:
    if not enabled:
        return {"requested": False, "ok": True, "steps": []}

    steps: List[str] = []
    errors: List[str] = []

    try:
        gc.collect()
        steps.append(t("gc.collect"))
    except Exception as e:
        errors.append(t("gc.collect failed: {error}", error=e))

    try:
        import comfy.model_management as model_management  # type: ignore

        # 优先使用 ComfyUI 官方卸载路径，避免仅清 cache 但模型仍常驻显存。
        if hasattr(model_management, "unload_all_models"):
            model_management.unload_all_models()
            steps.append(t("comfy.model_management.unload_all_models"))

        if hasattr(model_management, "cleanup_models"):
            try:
                model_management.cleanup_models()
                steps.append(t("comfy.model_management.cleanup_models"))
            except TypeError:
                # 兼容部分版本 cleanup_models 需要参数。
                model_management.cleanup_models(True)
                steps.append(t("comfy.model_management.cleanup_models(True)"))

        if hasattr(model_management, "soft_empty_cache"):
            model_management.soft_empty_cache()
            steps.append(t("comfy.model_management.soft_empty_cache"))
        elif hasattr(model_management, "empty_cache"):
            model_management.empty_cache()
            steps.append(t("comfy.model_management.empty_cache"))
    except Exception as e:
        errors.append(t("comfy model_management clear failed: {error}", error=e))

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            steps.append(t("torch.cuda.empty_cache"))
    except Exception as e:
        errors.append(t("torch cuda clear failed: {error}", error=e))

    return {
        "requested": True,
        "ok": len(errors) == 0,
        "steps": steps,
        "errors": errors,
    }


class DocumentLoaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "document": (
                    _list_input_docs_for_combo(),
                    {
                        "tooltip": t(
                            "选择一个文档或文件夹（txt/json/md/pdf）。可用下方“上传文档”按钮上传后再选择。"
                        )
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
        if not (document or "").strip():
            return True
        resolved = _resolve_document_targets(document or "", "")
        if not resolved:
            return t("无效文档: {document}", document=document)
        return True

    def load_documents(self, document: str):
        targets = _resolve_document_targets(document or "", "")
        if not targets:
            return ([], t("请在 document 中选择或上传一个文档（txt/json/md/pdf）或文件夹。"))
        documents, errors = _load_documents_from_targets(targets)

        summary = t(
            "文档加载完成。总文件: {total}, 成功: {success}, 失败: {failed}",
            total=len(documents) + len(errors),
            success=len(documents),
            failed=len(errors),
        )
        if errors:
            summary += "\n" + "\n".join(errors[:5])
        return (documents, summary)


RAG_DIR = os.path.join(os.path.dirname(__file__), "rag")

class EasyRAGPrebuiltLibraryNode:
    @classmethod
    def INPUT_TYPES(cls):
        libraries = []
        if os.path.isdir(RAG_DIR):
            libraries = [
                d for d in os.listdir(RAG_DIR)
                if os.path.isdir(os.path.join(RAG_DIR, d)) and not d.startswith(".")
            ]
        if not libraries:
            libraries = [""]  # 占位符
            
        return {
            "required": {
                "library_name": (libraries,),  # 自动生成下拉菜单
            }
        }

    RETURN_TYPES = ("RAG_DOCUMENTS", "STRING")
    RETURN_NAMES = ("documents", "summary")
    FUNCTION = "load_from_library"
    CATEGORY = "EasyRAG"

    @classmethod
    def VALIDATE_INPUTS(cls, library_name):
        return True

    def load_from_library(self, library_name: str):
        if not library_name:
            return ([], t("未选择任何预制库，或者 rag 文件夹内没有子文件夹。"))
            
        target_dir = os.path.join(RAG_DIR, library_name)
        if not os.path.isdir(target_dir):
            return ([], t("预制库文件夹不存在: {name}", name=library_name))
            
        targets = _resolve_document_targets(target_dir, "")
        documents, errors = _load_documents_from_targets(targets)
        
        summary = t(
            "预制库 [{name}] 加载完成。总文件: {total}, 成功: {success}, 失败: {failed}",
            name=library_name,
            total=len(documents) + len(errors),
            success=len(documents),
            failed=len(errors),
        )
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
                    {"tooltip": t("仅允许使用 ComfyUI/models/embeddings 下的本地模型")},
                ),
                "chunk_size": ("INT", {"default": 400, "min": 100, "max": 4000, "step": 10}),
                "chunk_overlap": ("INT", {"default": 80, "min": 0, "max": 2000, "step": 10}),
                "show_retrieval_log": (
                    "BOOLEAN",
                    {"default": True, "label_on": t("日志开"), "label_off": t("日志关")},
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
    ):
        selected_model = str(embedding_model or "").strip()
        if not selected_model:
            raise ValueError(
                t(
                    "未检测到可用 embedding 模型。请先把 sentence-transformers 模型放到 ComfyUI/models/embeddings。"
                )
            )
        info = build_faiss_index(
            documents=documents,
            embedding_model=selected_model,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            index_name=index_name.strip(),
        )
        unload_info = unload_embedding_model(selected_model)
        info["show_retrieval_log"] = bool(show_retrieval_log)
        info["embedding_unload_info"] = unload_info
        summary = (
            t(
                "向量库构建完成: {index_name}, 文档数: {documents_count}, chunk数: {chunks_count}, 模型: {selected_model}, 目录: {index_dir}",
                index_name=info["index_name"],
                documents_count=info["documents_count"],
                chunks_count=info["chunks_count"],
                selected_model=selected_model,
                index_dir=info["index_dir"],
            )
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
                    {"tooltip": t("模型选项来自 LM Studio API（默认地址: 127.0.0.1:1234）")},
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": t("你是一个严谨的本地RAG助手，优先根据给定上下文回答。"),
                    },
                ),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 16}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1, "control_after_generate": True},
                ),
                "stream": ("BOOLEAN", {"default": False, "label_on": t("流式开"), "label_off": t("流式关")}),
                "clear_vram_before_run": (
                    "BOOLEAN",
                    {"default": True, "label_on": t("运行前清理"), "label_off": t("直接运行")},
                ),
                "unload_model_after_response": (
                    "BOOLEAN",
                    {"default": True, "label_on": t("卸载模型"), "label_off": t("保留模型")},
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
        seed: int,
        stream: bool,
        clear_vram_before_run: bool,
        unload_model_after_response: bool,
        rag_index=None,
        image=None,
    ):
        base = base_url.strip()
        vram_cleanup = _clear_vram_before_run(bool(clear_vram_before_run))
        if not vram_cleanup.get("ok", True):
            print(t("[EasyRAG][显存清理] 运行前清理失败: {errors}", errors=vram_cleanup.get("errors", [])))

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
        embedding_unload_after_retrieval = None
        if rag_index is not None:
            index_ref = rag_index.get("index_dir") or rag_index.get("index_name")
            result = search_index(index_ref, query=question, top_k=int(top_k), device="cpu")
            context_used = result["context"]
            if show_retrieval_log:
                print(t("[EasyRAG][问答检索] question={question!r} top_k={top_k}", question=question, top_k=top_k))
                print(
                    t(
                        "[EasyRAG][问答检索] rag_hit={rag_hit} best_score={best_score:.4f}",
                        rag_hit=result.get("rag_hit"),
                        best_score=result.get("best_score", 0.0),
                    )
                )
                for i, item in enumerate(result.get("items", []), start=1):
                    snippet = (item.get("text", "") or "").replace("\n", " ")[:120]
                    print(
                        t(
                            "[EasyRAG][问答检索#{i}] score={score:.4f} source={source} chunk_text={chunk_text}",
                            i=i,
                            score=item.get("score", 0.0),
                            source=item.get("source", ""),
                            chunk_text=snippet,
                        )
                    )

            retrieval_embedding_model = str((rag_index or {}).get("embedding_model", "")).strip()
            embedding_unload_after_retrieval = unload_embedding_model(
                retrieval_embedding_model if retrieval_embedding_model else None
            )
            if show_retrieval_log:
                print(
                    t(
                        "[EasyRAG][问答检索] embedding检索后卸载: count={count} ok={ok}",
                        count=int(embedding_unload_after_retrieval.get("count", 0)),
                        ok=bool(embedding_unload_after_retrieval.get("ok", True)),
                    )
                )
                unload_errors = embedding_unload_after_retrieval.get("errors") or []
                if unload_errors:
                    print(
                        t(
                            "[EasyRAG][问答检索] embedding检索后卸载错误: {errors}",
                            errors=unload_errors,
                        )
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
            seed=int(seed),
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
            "vram_cleanup_before_run": vram_cleanup,
            "embedding_unload_after_retrieval": embedding_unload_after_retrieval,
            "auto_unload_before_switch": auto_unload_before_switch,
            "seed": int(seed),
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
                    {"tooltip": t("模型选项来自 LM Studio API（默认地址: 127.0.0.1:1234）")},
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": t("你是一个严谨的本地RAG助手，优先根据给定上下文回答。"),
                    },
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1, "control_after_generate": True},
                ),
                "clear_vram_before_run": (
                    "BOOLEAN",
                    {"default": True, "label_on": t("运行前清理"), "label_off": t("直接运行")},
                ),
                "unload_model_after_response": (
                    "BOOLEAN",
                    {"default": True, "label_on": t("卸载模型"), "label_off": t("保留模型")},
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
        seed: int,
        clear_vram_before_run: bool,
        unload_model_after_response: bool,
        rag_index=None,
        image=None,
    ):
        base = base_url.strip()
        vram_cleanup = _clear_vram_before_run(bool(clear_vram_before_run))
        if not vram_cleanup.get("ok", True):
            print(t("[EasyRAG][显存清理] 运行前清理失败: {errors}", errors=vram_cleanup.get("errors", [])))

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
            result = search_index(index_ref, query=question, device="cpu")
            context_used = result["context"]
            if show_retrieval_log:
                print(t("[EasyRAG][简约检索] question={question!r} top_k=5(default)", question=question))
                print(
                    t(
                        "[EasyRAG][简约检索] rag_hit={rag_hit} best_score={best_score:.4f}",
                        rag_hit=result.get("rag_hit"),
                        best_score=result.get("best_score", 0.0),
                    )
                )
                for i, item in enumerate(result.get("items", []), start=1):
                    snippet = (item.get("text", "") or "").replace("\n", " ")[:120]
                    print(
                        t(
                            "[EasyRAG][简约检索#{i}] score={score:.4f} source={source} chunk_text={chunk_text}",
                            i=i,
                            score=item.get("score", 0.0),
                            source=item.get("source", ""),
                            chunk_text=snippet,
                        )
                    )
            retrieval_embedding_model = str((rag_index or {}).get("embedding_model", "")).strip()
            embedding_unload_after_retrieval = unload_embedding_model(
                retrieval_embedding_model if retrieval_embedding_model else None
            )
            if show_retrieval_log:
                print(
                    t(
                        "[EasyRAG][简约检索] embedding检索后卸载: count={count} ok={ok}",
                        count=int(embedding_unload_after_retrieval.get("count", 0)),
                        ok=bool(embedding_unload_after_retrieval.get("ok", True)),
                    )
                )
                unload_errors = embedding_unload_after_retrieval.get("errors") or []
                if unload_errors:
                    print(
                        t(
                            "[EasyRAG][简约检索] embedding检索后卸载错误: {errors}",
                            errors=unload_errors,
                        )
                    )
        elif show_retrieval_log:
            print(t("[EasyRAG][简约检索] 未连接 rag_index，跳过检索。"))

        image_data_url = _image_tensor_to_data_url(image) if image is not None else ""
        response = lmstudio_chat(
            base_url=base,
            model=chosen_model,
            question=question,
            context=context_used,
            image_data_url=image_data_url,
            system_prompt=system_prompt,
            seed=int(seed),
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


# ---------------------------------------------------------------------------
# Qwen3-Embedding 系列默认 query instruction（仅 instruction-aware 模型生效）
# ---------------------------------------------------------------------------
_QWEN3_DEFAULT_INSTRUCTION = (
    "Retrieve relevant knowledge for AI portrait prompt engineering, "
    "including prompt paradigms, vocabulary, and high-quality examples."
)


class TwoStageRAGNode:
    """两阶段 RAG 检索节点：第一阶段匹配范式，第二阶段检索词汇与范例。"""

    @classmethod
    def INPUT_TYPES(cls):
        s1_default = (
            "你是写实人像提示词结构设计师，服务于 Z-Image（万相）模型。\n\n"
            "## 知识库\n\n"
            "你将收到检索自「01_prompt_paradigms.md」的内容，包含 4 种结构范式和 7 层通用骨架。\n\n"
            "## 任务\n\n"
            "根据用户主题，选择范式，搭建骨架。只输出骨架，不输出提示词。\n\n"
            "## 步骤\n\n"
            "1. 从用户主题中提取：人物、风格、光影、场景\n"
            "2. 从知识库 4 种范式中选最合适的（默认混合式）\n"
            "3. 按 7 层骨架写出每层的方向，每层一句话，不要展开细节\n\n"
            "## 输出格式（严格遵守，不要输出其他任何内容）\n\n"
            "主题：[复述用户主题]\n"
            "范式：[范式名称]\n"
            "1. [画质方向，一句话]\n"
            "2. [主体方向，一句话]\n"
            "3. [外貌方向，一句话]\n"
            "4. [服饰方向，一句话]\n"
            "5. [姿态方向，一句话]\n"
            "6. [场景方向，一句话]\n"
            "7. [光影方向，一句话]"
        )

        s2_default = (
            "你是中文写实人像提示词生成师，服务于 Z-Image（万相）模型。\n\n"
            "## 输入\n\n"
            "1. 用户主题\n"
            "2. 第一阶段输出的 7 层骨架\n"
            "3. 检索自「02_dimension_vocabulary.md」的词汇和搭配模式\n"
            "4. 检索自「03_portrait_examples.md」的优质范例\n\n"
            "## 任务\n\n"
            "根据骨架，用词汇库填充，参考范例写法，生成一段完整的中文提示词。\n\n"
            "## 核心原则\n\n"
            "**用户主题是唯一的创作依据。** 每一层内容都必须服务于用户描述的画面，不能自行添加用户没提到的元素。\n\n"
            "## 写法\n\n"
            "根据骨架指定的范式来写：\n"
            "- 氛围描述式：中文长句，像描述一幅画面\n"
            "- 词组堆叠式：逗号分隔词组\n"
            "- 权重精控式：描述 +（描述：权重值）\n"
            "- 混合式：自由组合\n\n"
            "## 词汇使用\n\n"
            "- 从词汇库选词，优先高频词\n"
            "- 用高频搭配模式组织词汇\n"
            "- 最终选什么词由你根据主题判断，不是机械拼凑\n\n"
            "## 质量底线\n\n"
            "- 光影描述不能少\n"
            "- 服饰写具体款式和材质\n"
            "- 权重语法谨慎使用\n"
            "- 氛围修饰词收尾\n\n"
            "## 长度要求\n\n"
            "基于 167 条真实高赞提示词的统计：\n"
            "- 平均长度：426 字符\n"
            "- 中位数：308 字符\n"
            "- 多数集中在 200~400 字符之间\n\n"
            "你的输出长度应控制在 **200~400 字符**。如果骨架信息量大，可以写到 400 字符上限；如果主题简单，200~300 字符就足够。不要为了凑长度而堆砌无关词汇，也不要过度简略导致信息缺失。\n\n"
            "## 输出\n\n"
            "只输出一段中文提示词。不要输出任何其他内容。不要输出反向提示词。不要输出结构化格式、不要加粗、不要标题、不要编号、不要分隔线。就是一段纯文本提示词，可以直接复制到 Z-Image 使用。"
        )

        return {
            "required": {
                "question": ("STRING", {"multiline": True, "default": ""}),
                "base_url": ("STRING", {"default": "http://127.0.0.1:1234"}),
                "model": (
                    _list_lmstudio_models_for_ui(),
                    {"tooltip": t("模型选项来自 LM Studio API（默认地址: 127.0.0.1:1234）")},
                ),
                "stage1_system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": s1_default,
                    },
                ),
                "stage2_system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": s2_default,
                    },
                ),

                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 32, "max": 8192, "step": 16}),
                "top_k_stage1": ("INT", {"default": 6, "min": 1, "max": 20, "step": 1}),
                "top_k_stage2": ("INT", {"default": 8, "min": 1, "max": 20, "step": 1}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1, "control_after_generate": True},
                ),
                "stream": ("BOOLEAN", {"default": False, "label_on": t("流式开"), "label_off": t("流式关")}),
                "clear_vram_before_run": (
                    "BOOLEAN",
                    {"default": True, "label_on": t("运行前清理"), "label_off": t("直接运行")},
                ),
                "unload_model_after_response": (
                    "BOOLEAN",
                    {"default": True, "label_on": t("卸载模型"), "label_off": t("保留模型")},
                ),
                "show_retrieval_log": (
                    "BOOLEAN",
                    {"default": True, "label_on": t("日志开"), "label_off": t("日志关")},
                ),
            },
            "optional": {
                "rag_index": ("RAG_INDEX",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("answer", "retrieval_debug")
    FUNCTION = "two_stage_rag"
    CATEGORY = "EasyRAG"

    def two_stage_rag(
        self,
        question: str,
        base_url: str,
        model: str,
        stage1_system_prompt: str,
        stage2_system_prompt: str,

        temperature: float,
        max_tokens: int,
        top_k_stage1: int,
        top_k_stage2: int,
        seed: int,
        stream: bool,
        clear_vram_before_run: bool,
        unload_model_after_response: bool,
        show_retrieval_log: bool,
        rag_index=None,
        image=None,
    ):
        def _sanitize_final_output(text: str) -> str:
            cleaned = extract_answer_between_newlines(text or "").strip()
            if not cleaned:
                return ""

            lines = cleaned.splitlines()
            # 去掉包裹代码块，保留内部正文
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
                while lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]

            # 仅移除首部 markdown 标题行，保留正文结构与内容
            while lines and lines[0].strip().startswith("#"):
                lines = lines[1:]
            return "\n".join(lines).strip()

        base = base_url.strip()
        vram_cleanup = _clear_vram_before_run(bool(clear_vram_before_run))
        if not vram_cleanup.get("ok", True):
            print(t("[EasyRAG][两阶段] 运行前清理失败: {errors}", errors=vram_cleanup.get("errors", [])))

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
        stage1_outline = ""
        debug_lines: List[str] = []

        def _log(message: str):
            if not show_retrieval_log:
                return
            debug_lines.append(message)
            print(f"[EasyRAG][两阶段] {message}")

        if rag_index is not None:
            index_ref = rag_index.get("index_dir") or rag_index.get("index_name")

            # ===== 第一阶段：从范式模板中检索 =====
            _log(
                f"[Stage1] retrieval_start top_k={int(top_k_stage1)} role_filter=paradigm,system,general"
            )
            stage1 = search_index(
                index_ref,
                query=question,
                top_k=int(top_k_stage1),
                device="cpu",
                role_filter=["paradigm", "system", "general"],
                query_instruction=_QWEN3_DEFAULT_INSTRUCTION,
            )
            _log(
                f"[Stage1] hits={len(stage1.get('items', []))} best_score={stage1.get('best_score', 0):.4f}"
            )
            for i, item in enumerate(stage1.get("items", []), start=1):
                snippet = (item.get("text", "") or "").replace("\n", " ")[:100]
                _log(
                    f"  [S1#{i}] score={item.get('score', 0):.4f} role={item.get('doc_role', '?')} {snippet}"
                )

            # 第一阶段骨架生成，具体步骤和输出格式由 stage1_system_prompt 控制
            stage1_context = stage1["context"] if stage1["rag_hit"] else ""
            stage1_task = (
                "请严格按照系统提示词的要求与输出格式，根据以下用户主题结合知识库上下文进行推理。\n"
                "【⚠️重要注意】：必须生成具体的方向指导，并将系统提示词模板中的占位符（如 '[画质方向，一句话]' 等）替换为你实际生成的内容，绝不能原样照抄占位符模板！\n\n"
                f"用户主题：\n{question}"
            )
            _log(f"[Stage1] inference_start max_tokens={int(max_tokens)}")
            stage1_response = lmstudio_chat(
                base_url=base,
                model=chosen_model,
                question=stage1_task,
                context=stage1_context,
                image_data_url="",
                system_prompt=stage1_system_prompt,
                temperature=max(0.0, float(temperature) * 0.5),
                max_tokens=int(max_tokens),
                seed=int(seed),
                api_mode="responses",
                stream=False,
                emit_stream_log=False,
            )
            stage1_outline = extract_answer_between_newlines(stage1_response["answer"]).strip()
            if not stage1_outline:
                stage1_outline = t(
                    "结构化草案为空。请基于用户需求直接补齐结构槽位。"
                )
            _log("[Stage1] inference_done")
            _log(f"[Stage1Draft] {stage1_outline[:300].replace(chr(10), ' ')}")

            # ===== 第二阶段：根据“结构化草案”检索词汇与示例 =====
            stage2_query = (
                f"用户需求：{question}\n\n"
                f"结构化草案：\n{stage1_outline}\n\n"
                "请为以上结构检索可填充的词汇和示例。"
            )
            _log(
                f"[Stage2] retrieval_start top_k={int(top_k_stage2)} role_filter=vocabulary,example"
            )
            stage2 = search_index(
                index_ref,
                query=stage2_query,
                top_k=int(top_k_stage2),
                device="cpu",
                role_filter=["vocabulary", "example"],
                query_instruction=_QWEN3_DEFAULT_INSTRUCTION,
            )
            _log(
                f"[Stage2] hits={len(stage2.get('items', []))} best_score={stage2.get('best_score', 0):.4f}"
            )
            for i, item in enumerate(stage2.get("items", []), start=1):
                snippet = (item.get("text", "") or "").replace("\n", " ")[:100]
                _log(
                    f"  [S2#{i}] score={item.get('score', 0):.4f} role={item.get('doc_role', '?')} {snippet}"
                )

            # 仅将第二阶段作为填词上下文（第一阶段草案单独传入 question）
            stage2_context = stage2["context"] if stage2["rag_hit"] else ""

            context_parts = []
            if stage1_context:
                context_parts.append(
                    f"=== {t('第一阶段：范式匹配')} ===\n{stage1_context}"
                )
            if stage2_context:
                context_parts.append(
                    f"=== {t('第二阶段：词汇与范例')} ===\n{stage2_context}"
                )
            context_used = "\n\n".join(context_parts)

            # 卸载 embedding 模型
            retrieval_embedding_model = str((rag_index or {}).get("embedding_model", "")).strip()
            unload_embedding_model(
                retrieval_embedding_model if retrieval_embedding_model else None
            )
        elif show_retrieval_log:
            print(t("[EasyRAG][两阶段] 未连接 rag_index，跳过检索。"))

        image_data_url = _image_tensor_to_data_url(image) if image is not None else ""
        final_task = question
        final_context = context_used
        if stage1_outline:
            # 第二阶段填词生成：将草案长句转化为高质量的AI绘画提示词
            final_task = (
                "请严格按照系统提示词的填词要求，阅读【用户主题】与【第一阶段骨架】。你需要将骨架中的常规描述，转化为更专业、更精炼的 AI 绘画提示词。\n"
                "【⚠️检索库使用要求】：从知识库提供的【词汇与范例】上下文中，挑选**最贴合主题**的高质量词汇来替换普通词汇。**必须克制！绝不能无脑堆砌大词！**\n"
                "【⚠️排版风格注意】：最终输出必须是精炼的专业提示词短语组合。切忌啰嗦叙事，剔除所有的废话主谓宾。\n"
                "【⚠️去重与防崩坏审查（极度重要）】：绝对不能出现语义重复的视觉元素（如同一部位有多种描述、重复描写两遍服饰或手部动作）！合并所有同类项，确保物理空间逻辑严密，重复描述会导致 AI 绘画产生畸变（多重肢体）。\n"
                "【⚠️字数与密度】：极其克制！合并去重后，总词组/短语数量强制控制在 15~25 个！\n"
                "【⚠️最终输出限制】：直接输出最后结果正文，不留任何解释、标题和括号标签。\n\n"
                f"用户主题：\n{question}\n\n"
                f"第一阶段骨架：\n{stage1_outline}"
            )
            # 第二阶段仅需要词汇+示例上下文
            final_context = ""
            if rag_index is not None:
                # 从合并 context 中抽取第二阶段段落，避免把第一阶段再混入填词素材
                marker = f"=== {t('第二阶段：词汇与范例')} ==="
                if marker in context_used:
                    final_context = context_used.split(marker, 1)[1].strip()
                else:
                    final_context = context_used
        _log(f"[Stage2] inference_start max_tokens={int(max_tokens)}")

        response = lmstudio_chat(
            base_url=base,
            model=chosen_model,
            question=final_task,
            context=final_context,
            image_data_url=image_data_url,
            system_prompt=stage2_system_prompt,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            seed=int(seed) + 1,
            api_mode="responses",
            stream=bool(stream),
            emit_stream_log=bool(stream or show_retrieval_log),
        )
        _log("[Stage2] inference_done")

        _LAST_MODEL_BY_BASE_URL[base] = chosen_model or response.get("model", "")
        selected_model = response.get("model") or chosen_model
        if unload_model_after_response and selected_model:
            try:
                unload_lmstudio_model(base, selected_model)
                if _LAST_MODEL_BY_BASE_URL.get(base) == selected_model:
                    _LAST_MODEL_BY_BASE_URL.pop(base, None)
            except Exception:
                pass

        _log(f"[Stage2] inference_output_len={len(response['answer'])}")
        extracted_answer = _sanitize_final_output(response["answer"])
        if not extracted_answer:
            extracted_answer = t(
                "第二阶段生成为空。请适当提高 max_tokens，或检查 Stage2 检索命中后重试。"
            )
            _log("[Stage2] empty_response")
        debug_out = "\n".join(debug_lines) if debug_lines else ""
        return (extracted_answer, debug_out)


def _log(msg: str):
    print(f"[EasyRAG] {msg}")


def _sanitize_final_output(text: str) -> str:
    # 提取核心内容并清理常见的标签后缀
    raw = extract_answer_between_newlines(text)
    # 针对 Z-Image 的特定清理逻辑：移除可能出现的引导性前缀
    prefixes = [
        "提示词：", "正文：", "输出：", "Prompt:", "Output:", 
        "生成的提示词如下：", "最终提示词如下：", "【最终结果】",
        "Generated Prompt:", "Final Prompt:"
    ]
    for p in prefixes:
        if raw.startswith(p):
            raw = raw[len(p):].strip()
            break
    return raw.strip()


class EasyRAGRetrieverNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "question": ("STRING", {"multiline": True, "default": ""}),
                "rag_index": ("RAG_INDEX",),
                "top_k": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
                "rewrite_query": (
                    "BOOLEAN",
                    {"default": False, "label_on": "开启重写", "label_off": "关闭重写"},
                ),
                "base_url": ("STRING", {"default": "http://127.0.0.1:1234"}),
                "model": (
                    _list_lmstudio_models_for_ui(),
                    {"tooltip": "LLM model for rewriting the query"},
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are a query rewriter. Rewrite the user's input into a high-quality search query for a vector database.",
                    },
                ),
                # 新增：卸载开关
                "unload_model_after_response": (
                    "BOOLEAN",
                    {"default": False, "label_on": t("Unload Model"), "label_off": t("Keep Model")},
                ),
            }
            
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("retrieved_context", "count")
    FUNCTION = "retrieve"
    CATEGORY = "EasyRAG"

    def retrieve(
        self, question, rag_index, top_k, rewrite_query, base_url, model, system_prompt, unload_model_after_response
    ):
        if not (question or "").strip():
            return ("", 0)

        search_query = question.strip()
        selected_model = model

        if rewrite_query:
            try:
                response = lmstudio_chat(
                    base_url=base_url,
                    model=model,
                    question=question,
                    system_prompt=system_prompt,
                    temperature=0.1,
                    max_tokens=200,
                    stream=False,
                )
                search_query = extract_answer_between_newlines(response["answer"])
                selected_model = response.get("model") or model
                print(f"[EasyRAG][Retriever] Rewritten query: {search_query}")
            except Exception as e:
                print(f"[EasyRAG][Retriever] Query rewrite failed: {e}")

        index_ref = rag_index.get("index_dir") or rag_index.get("index_name")
        result = search_index(
            index_ref, query=search_query, top_k=int(top_k), device="cpu"
        )

                # 运行后卸载逻辑
        if unload_model_after_response and rewrite_query and selected_model:
            try:
                from .rag_core import unload_lmstudio_model
                unload_lmstudio_model(base_url, selected_model)
            except Exception as e:
                print(f"[EasyRAG][Retriever] Unload model failed: {e}")

        items = result.get("items", [])
        context_parts = [item["text"] for item in items]
        context = "\n\n".join(context_parts)

        return (context, len(items))


NODE_CLASS_MAPPINGS = {
    "EasyRAGDocumentLoader": DocumentLoaderNode,
    "EasyRAGPrebuiltLibrary": EasyRAGPrebuiltLibraryNode,
    "EasyRAGVectorStoreBuilder": VectorStoreBuilderNode,
    "EasyRAGLMStudioChatAdvanced": LMStudioRAGChatNode,
    "EasyRAGLMStudioChatSimple": LMStudioRAGChatSimpleNode,
    "EasyRAGTwoStageRAG": TwoStageRAGNode,
    "EasyRAGRetriever": EasyRAGRetrieverNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyRAGDocumentLoader": "EasyRAG - Document Loader",
    "EasyRAGPrebuiltLibrary": "EasyRAG-预制库",
    "EasyRAGVectorStoreBuilder": "EasyRAG - Vector Store Builder (FAISS)",
    "EasyRAGLMStudioChatAdvanced": "EasyRAG - LM Studio API (Advanced)",
    "EasyRAGLMStudioChatSimple": "EasyRAG - LM Studio API (Simple)",
    "EasyRAGTwoStageRAG": "EasyRag-Zimage-2步检索推理",
    "EasyRAGRetriever": "EasyRAG - Vector Library Retriever",
}

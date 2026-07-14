import base64
import gc
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import folder_paths

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
    external_api_chat,
)
from .i18n import t


SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}

PREBUILT_SOURCE_PLUGIN = "plugin"
PREBUILT_SOURCE_ORIGINAL = "original"

_TORCH_MODULE = None
_TORCH_IMPORT_ATTEMPTED = False
_MODEL_MANAGEMENT_MODULE = None
_MODEL_MANAGEMENT_IMPORT_ATTEMPTED = False
_NUMPY_MODULE = None
_PIL_IMAGE_CLASS = None


def _get_torch():
    global _TORCH_MODULE, _TORCH_IMPORT_ATTEMPTED
    if not _TORCH_IMPORT_ATTEMPTED:
        _TORCH_IMPORT_ATTEMPTED = True
        try:
            import torch  # type: ignore
            _TORCH_MODULE = torch
        except Exception:
            _TORCH_MODULE = None
    return _TORCH_MODULE


def _get_model_management():
    global _MODEL_MANAGEMENT_MODULE, _MODEL_MANAGEMENT_IMPORT_ATTEMPTED
    if not _MODEL_MANAGEMENT_IMPORT_ATTEMPTED:
        _MODEL_MANAGEMENT_IMPORT_ATTEMPTED = True
        try:
            import comfy.model_management as model_management  # type: ignore
            _MODEL_MANAGEMENT_MODULE = model_management
        except Exception:
            _MODEL_MANAGEMENT_MODULE = None
    return _MODEL_MANAGEMENT_MODULE


def _get_numpy():
    global _NUMPY_MODULE
    if _NUMPY_MODULE is None:
        import numpy as np  # type: ignore
        _NUMPY_MODULE = np
    return _NUMPY_MODULE


def _get_pil_image_class():
    global _PIL_IMAGE_CLASS
    if _PIL_IMAGE_CLASS is None:
        from PIL import Image  # type: ignore
        _PIL_IMAGE_CLASS = Image
    return _PIL_IMAGE_CLASS


def _soft_empty_cache(ipc_collect: bool = False):
    model_management = _get_model_management()
    if model_management is not None:
        try:
            if hasattr(model_management, "soft_empty_cache"):
                model_management.soft_empty_cache()
        except Exception:
            pass

    torch = _get_torch()
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if ipc_collect and hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


# ====================== 【1】加在最上方：原作者核心显存清理函数 ======================
def _clear_vram_before_run(enabled: bool):
    if not enabled:
        return
    gc.collect()
    model_management = _get_model_management()
    try:
        if model_management is not None and hasattr(model_management, "unload_all_models"):
            model_management.unload_all_models()
        if model_management is not None and hasattr(model_management, "cleanup_models"):
            model_management.cleanup_models(True)
        if model_management is not None and hasattr(model_management, "soft_empty_cache"):
            model_management.soft_empty_cache()
    except:
        pass
    torch = _get_torch()
    try:
        if torch is not None and torch.cuda.is_available():
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
        for root in source_roots.values():
            if not root.exists():
                continue
            for item in root.iterdir():
                name = f"📂 {item.name}" if item.is_dir() else f"📄 {item.name}"
                if item.is_file() and not _is_supported_doc_file(str(item)):
                    continue
                if name in seen:
                    continue
                seen.add(name)
                items.append(name)
        return sorted(items) if items else [""]
    except:
        return [""]


def _list_system_prompt_files_for_combo() -> List[str]:
    """列出 systemprompt 文件夹中的文件，第一个选项为'自定义'"""
    systemprompt_root = Path(__file__).resolve().parent / "systemprompt"
    items = ["🛠️ 自定义"]  # 第一个选项
    
    try:
        if systemprompt_root.exists():
            for item in systemprompt_root.iterdir():
                if item.is_file() and item.suffix.lower() in {".txt", ".md"}:
                    items.append(f"📄 {item.name}")
    except:
        pass
    
    return items if len(items) > 1 else ["🛠️ 自定义"]


def _resolve_system_prompt_file(selection: str) -> str:
    """根据选择返回系统提示词内容，如果选择'自定义'或文件不存在返回空字符串"""
    if not selection or "🛠️" in selection or "自定义" in selection:
        return ""  # 返回空字符串，表示使用输入框内容
    
    # 去除图标前缀
    normalized_selection = selection
    for prefix in ("📄", "📜"):
        if normalized_selection.startswith(prefix):
            normalized_selection = normalized_selection[len(prefix):].strip()
            break
    
    systemprompt_root = Path(__file__).resolve().parent / "systemprompt"
    file_path = systemprompt_root / normalized_selection
    
    try:
        if file_path.exists():
            return file_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"⚠️ [EasyRAG] 读取系统提示词文件失败: {e}")
    
    return ""  # 文件不存在或读取失败时回退到自定义


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

    # Support display labels such as "📂 folder" / "📄 file" from combo values.
    normalized_raw = raw
    for prefix in ("📂", "📄"):
        if normalized_raw.startswith(prefix):
            normalized_raw = normalized_raw[len(prefix):].strip()
            break

    source_roots = _get_prebuilt_source_roots()
    candidates: List[Path] = []
    relative = normalized_raw

    # Backward compatibility: support values like "plugin:xxx" and "original:xxx".
    if ":" in normalized_raw:
        maybe_source, maybe_relative = normalized_raw.split(":", 1)
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
    models = list_lmstudio_models("http://127.0.0.1:1234", timeout=0.25)
    return models if models else [""]


def _dynamic_image_index(name: str) -> int:
    if name == "image":
        return 1
    if name.startswith("image_"):
        tail = name[len("image_"):]
        if tail.isdigit():
            return int(tail)
    return 10**9


def _is_dynamic_image_input(name: str) -> bool:
    return name.startswith("image_") and name[len("image_"):].isdigit()


def _sorted_dynamic_image_values(kwargs: Dict[str, Any]) -> List[Any]:
    items: List[Tuple[int, str, Any]] = []
    for key, value in kwargs.items():
        if _is_dynamic_image_input(key):
            items.append((_dynamic_image_index(key), key, value))
    items.sort(key=lambda item: (item[0], item[1]))
    return [value for _, _, value in items]


class DynamicApiOptionalInputs(dict):
    def __contains__(self, key):
        return dict.__contains__(self, key) or _is_dynamic_image_input(str(key))

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        if _is_dynamic_image_input(str(key)):
            return ("IMAGE", {"label": t("image")})
        raise KeyError(key)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default


def _image_tensor_to_data_urls(image) -> List[str]:
    if image is None:
        return []
    if isinstance(image, (list, tuple)):
        urls: List[str] = []
        for item in image:
            urls.extend(_image_tensor_to_data_urls(item))
        return urls

    np = _get_numpy()
    Image = _get_pil_image_class()
    arr = image
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    frames = [arr[i] for i in range(arr.shape[0])] if arr.ndim == 4 else [arr]
    urls: List[str] = []
    for frame in frames:
        frame = np.clip(frame, 0.0, 1.0)
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        if frame.ndim == 3 and frame.shape[-1] > 3:
            frame = frame[..., :3]
        frame = (frame * 255.0).astype(np.uint8)
        img = Image.fromarray(frame)
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        urls.append(f"data:image/png;base64,{b64}")

    return urls


def _image_tensor_to_data_url(image) -> str:
    urls = _image_tensor_to_data_urls(image)
    return urls[0] if urls else ""


def _collect_image_data_urls(image=None, **kwargs) -> List[str]:
    urls = _image_tensor_to_data_urls(image)
    for value in _sorted_dynamic_image_values(kwargs):
        urls.extend(_image_tensor_to_data_urls(value))
    return urls


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
        _soft_empty_cache(ipc_collect=True)
            
        return (documents, summary)


# ==============================================
# 向量库构建节点
# ==============================================
class VectorStoreBuilderNode:
    @staticmethod
    def _build_mode_values() -> List[str]:
        # Keep all options for backend validation compatibility. 
        # The UI will be filtered to only 2 items by our JS extension.
        return [
            "create_new",
            "use_existing",
            "Create New",
            "Use Existing",
            "新建向量库",
            "使用已有向量库",
        ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "documents": ("RAG_DOCUMENTS", {"label": t("documents")}),
                "build_mode": (cls._build_mode_values(), {
                    "default": "create_new",
                    "label": t("build_mode"),
                    "tooltip": t("Choose whether to create a new vector store or use an existing one")
                }),
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
        build_mode: str,
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
        mode_raw = str(build_mode or "create_new").strip().lower()
        use_existing = (
            mode_raw == "use_existing"
            or "use existing" in mode_raw
            or "已有" in mode_raw
        )
        existing_name = str(index_list or "").strip()
        new_name = str(index_name or "").strip()

        if use_existing:
            final_name = existing_name
            if not final_name:
                raise ValueError(t("Please select an existing vector store"))
        else:
            final_name = new_name
            if not final_name:
                raise ValueError(t("Please enter a new vector store name"))
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
        _soft_empty_cache(ipc_collect=True)

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
                "system_prompt_source": (_list_system_prompt_files_for_combo(), {
                    "default": "🛠️ 自定义",
                    "label": t("system_prompt_source")
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": t("You are a rigorous local RAG assistant. Prefer answering from the provided context."),
                    "label": t("system_prompt")
                }),
                "temperature": ("FLOAT", {"default": 0.2, "label": t("temperature")}),
                "max_tokens": ("INT", {"default": 2048, "min": 0, "max": 8192, "step": 512, "label": t("max_tokens")}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "label": t("seed")}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 100, "label": t("top_k")}),
                "stream": ("BOOLEAN", {"default": True, "label": t("stream")}),
                "unload_model_after_response": ("BOOLEAN", {"default": True, "label": t("unload_model_after_response")}),
            },
            "optional": DynamicApiOptionalInputs({
                "rag_index": ("RAG_INDEX", {"label": t("rag_index")}),
                "image": ("IMAGE", {"label": t("image")})
            }),
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = (t("answer"), t("context_used"), t("raw_response"))
    FUNCTION = "chat_with_rag"
    CATEGORY = "RagPrompt"

    def chat_with_rag(self, question, base_url, model, system_prompt, system_prompt_source, temperature, max_tokens, seed, top_k, stream, unload_model_after_response, rag_index=None, image=None, **kwargs):
        # 【2】每个节点第一行：清显存
        _clear_vram_before_run(True)

        # 判断使用哪个系统提示词
        if system_prompt_source and "🛠️" not in system_prompt_source and "自定义" not in system_prompt_source:
            file_content = _resolve_system_prompt_file(system_prompt_source)
            if file_content:
                print(f"📝 [高级API] 使用文件系统提示词: {system_prompt_source}")
                system_prompt = file_content
            else:
                print(f"📝 [高级API] 未找到提示词文件或内容为空，使用输入框提示词")
        else:
            print(f"📝 [高级API] 使用自定义输入框提示词")

        base = base_url.strip()
        models = list_lmstudio_models(base)
        chosen = model.strip() or (models[0] if models else "")

        print(f"🤖 [高级API] 选择模型: {chosen}")
        print(f"🔗 [高级API] 连接地址: {base}")
        print(f"📝 [高级API] 流式输出: {stream}")

        if _LAST_MODEL_BY_BASE_URL.get(base) and _LAST_MODEL_BY_BASE_URL[base] != chosen:
            try:
                print(f"🔄 [高级API] 切换模型: {_LAST_MODEL_BY_BASE_URL[base]} -> {chosen}")
                unload_lmstudio_model(base, _LAST_MODEL_BY_BASE_URL[base])
            except:
                pass

        ctx = ""
        if rag_index:
            ref = rag_index.get("index_dir") or rag_index.get("index_name")
            # 【3】加 device="cpu"
            print(f"🔍 [高级API] 开始RAG检索 (top_k={top_k})")
            res = search_index(ref, question, top_k=top_k, device="cpu")
            ctx = res["context"]
            print(f"✅ [高级API] RAG检索完成，检索到 {len(res['items'])} 个相关片段")
            # 【4】检索完强制卸载embedding模型
            try:
                unload_embedding_model(rag_index["embedding_model"])
                print("♻️ [RAG] 检索完成，已卸载embedding模型")
            except:
                pass

        image_urls = _collect_image_data_urls(image, **kwargs)
        print(f"🚀 [高级API] 开始生成回答...")
        resp = lmstudio_chat(
            base_url=base, model=chosen,
            question=question, context=ctx, image_data_urls=image_urls,
            system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens,
            seed=seed, stream=stream, emit_stream_log=True
        )
        print(f"✅ [高级API] 生成完成，回答长度: {len(resp['answer'])} 字符")

        _LAST_MODEL_BY_BASE_URL[base] = chosen
        if unload_model_after_response and chosen:
            try:
                print(f"♻️ [高级API] 卸载模型: {chosen}")
                unload_lmstudio_model(base, chosen)
                _LAST_MODEL_BY_BASE_URL.pop(base, None)
                print(f"✅ [高级API] 模型卸载完成")
            except:
                pass

        # 【5】保留原有末尾显存清理
        gc.collect()
        _soft_empty_cache(ipc_collect=True)

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
                "system_prompt_source": (_list_system_prompt_files_for_combo(), {
                    "default": "🛠️ 自定义",
                    "label": t("system_prompt_source")
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": t("You are a rigorous local RAG assistant. Prefer answering from the provided context."),
                    "label": t("system_prompt")
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "label": t("seed")}),
                "unload_model_after_response": ("BOOLEAN", {"default": True, "label": t("unload_model_after_response")}),
            },
            "optional": DynamicApiOptionalInputs({
                "rag_index": ("RAG_INDEX", {"label": t("rag_index")}),
                "image": ("IMAGE", {"label": t("image")})
            }),
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = (t("answer"),)
    FUNCTION = "chat_simple"
    CATEGORY = "RagPrompt"

    def chat_simple(self, question, base_url, model, system_prompt, system_prompt_source, seed, unload_model_after_response, rag_index=None, image=None, **kwargs):
        # 【2】每个节点第一行：清显存
        _clear_vram_before_run(True)
        
        # 判断使用哪个系统提示词
        if system_prompt_source and "🛠️" not in system_prompt_source and "自定义" not in system_prompt_source:
            file_content = _resolve_system_prompt_file(system_prompt_source)
            if file_content:
                print(f"📝 [简约API] 使用文件系统提示词: {system_prompt_source}")
                system_prompt = file_content
            else:
                print(f"📝 [简约API] 未找到提示词文件或内容为空，使用输入框提示词")
        else:
            print(f"📝 [简约API] 使用自定义输入框提示词")
        
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
            question=question, context=ctx, image_data_urls=_collect_image_data_urls(image, **kwargs),
            system_prompt=system_prompt, temperature=0.2, max_tokens=4096, seed=seed, stream=False, api_mode="chat_completions"
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
        _soft_empty_cache(ipc_collect=True)

        return (extract_answer_between_newlines(resp["answer"]),)


# ==============================================
# 外部 API 对话节点 (高级)
# ==============================================
class ExternalRAGChatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "question": ("STRING", {"multiline": True, "label": t("question")}),
                "base_url": ("STRING", {"default": "https://api.deepseek.com", "label": t("base_url")}),
                "api_key": ("STRING", {"default": "", "label": t("api_key")}),
                "model": ("STRING", {"default": "deepseek-chat", "label": t("model")}),
                "system_prompt_source": (_list_system_prompt_files_for_combo(), {
                    "default": "🛠️ 自定义",
                    "label": t("system_prompt_source")
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": t("You are a rigorous local RAG assistant. Prefer answering from the provided context."),
                    "label": t("system_prompt")
                }),
                "temperature": ("FLOAT", {"default": 0.7, "label": t("temperature")}),
                "max_tokens": ("INT", {"default": 2048, "min": 0, "max": 8192, "step": 512, "label": t("max_tokens")}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "label": t("seed")}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 100, "label": t("top_k")}),
                "stream": ("BOOLEAN", {"default": True, "label": t("stream")}),
            },
            "optional": DynamicApiOptionalInputs({
                "rag_index": ("RAG_INDEX", {"label": t("rag_index")}),
                "image": ("IMAGE", {"label": t("image")})
            }),
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = (t("answer"), t("context_used"), t("raw_response"))
    FUNCTION = "chat_with_external_rag"
    CATEGORY = "RagPrompt"

    def chat_with_external_rag(self, question, base_url, api_key, model, system_prompt, system_prompt_source, temperature, max_tokens, seed, top_k, stream, rag_index=None, image=None, **kwargs):
        _clear_vram_before_run(True)

        # 判断使用哪个系统提示词
        if system_prompt_source and "🛠️" not in system_prompt_source and "自定义" not in system_prompt_source:
            file_content = _resolve_system_prompt_file(system_prompt_source)
            if file_content:
                print(f"📝 [外部API] 使用文件系统提示词: {system_prompt_source}")
                system_prompt = file_content
            else:
                print(f"📝 [外部API] 未找到提示词文件或内容为空，使用输入框提示词")
        else:
            print(f"📝 [外部API] 使用自定义输入框提示词")

        base = base_url.strip()
        chosen = model.strip()

        print(f"🌐 [外部API] 使用模型: {chosen}")
        print(f"🔗 [外部API] 请求地址: {base}")

        ctx = ""
        if rag_index:
            ref = rag_index.get("index_dir") or rag_index.get("index_name")
            print(f"🔍 [外部API] 开始RAG检索 (top_k={top_k})")
            res = search_index(ref, question, top_k=top_k, device="cpu")
            ctx = res["context"]
            print(f"✅ [外部API] RAG检索完成")
            try:
                unload_embedding_model(rag_index["embedding_model"])
            except:
                pass

        image_urls = _collect_image_data_urls(image, **kwargs)
        print(f"🚀 [外部API] 开始调用云端生成...")
        
        resp = external_api_chat(
            base_url=base,
            api_key=api_key,
            model=chosen,
            question=question,
            context=ctx,
            image_data_urls=image_urls,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            stream=stream,
            emit_stream_log=True
        )
        
        print(f"✅ [外部API] 生成完成")

        gc.collect()
        _soft_empty_cache()

        ans = extract_answer_between_newlines(resp["answer"])
        return (ans, ctx, json.dumps(resp, ensure_ascii=False))


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
        _soft_empty_cache()
            
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
    "RagPromptExternalChatAdvanced": ExternalRAGChatNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RagPromptDocumentLoader": t("EasyRAG - Document Loader"),
    "RagPromptPrebuiltLoader": t("Rag 预制文档加载"),
    "RagPromptVectorStoreBuilder": t("EasyRAG - Vector Store Builder (FAISS)"),
    "RagPromptLMStudioChatAdvanced": t("EasyRAG - LM Studio API (Advanced)"),
    "RagPromptLMStudioChatSimple": t("EasyRAG - LM Studio API (Simple)"),
    "RagPromptExternalChatAdvanced": t("EasyRAG - External API (Advanced)"),
}

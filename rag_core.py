from __future__ import annotations

import json
import re
import threading
import io
import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import requests
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None

try:
    from transformers.utils import logging as hf_logging  # type: ignore
except Exception:  # pragma: no cover
    hf_logging = None


SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}


def _safe_read_text(path: Path, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding, errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        from PyPDF2 import PdfReader  # fallback
    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def parse_json_to_text(raw: str) -> str:
    try:
        obj = json.loads(raw)
    except Exception:
        return raw
    return json.dumps(obj, ensure_ascii=False, indent=2)


def load_single_document(path: Path, encoding: str = "utf-8") -> Dict:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    if ext in {".txt", ".md"}:
        text = _safe_read_text(path, encoding=encoding)
    elif ext == ".json":
        text = parse_json_to_text(_safe_read_text(path, encoding=encoding))
    elif ext == ".pdf":
        text = _read_pdf(path)
    else:
        text = ""

    return {
        "source": str(path),
        "extension": ext,
        "text": text.strip(),
        "title": path.name,
    }


def expand_paths(path_text: str) -> List[Path]:
    if not path_text.strip():
        return []

    parts = re.split(r"[\n,;]+", path_text.strip())
    files: List[Path] = []
    for p in parts:
        raw = p.strip().strip('"').strip("'")
        if not raw:
            continue
        path = Path(raw)
        if path.is_file():
            files.append(path)
            continue
        if path.is_dir():
            for ext in SUPPORTED_EXTENSIONS:
                files.extend(path.rglob(f"*{ext}"))
            continue
        # allow glob patterns
        for hit in Path(".").glob(raw):
            if hit.is_file() and hit.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(hit.resolve())

    # deduplicate + stable
    seen = set()
    out = []
    for f in files:
        k = str(f.resolve())
        if k not in seen:
            seen.add(k)
            out.append(f.resolve())
    return out


def split_text(text: str, chunk_size: int = 700, chunk_overlap: int = 120) -> List[str]:
    text = re.sub(r"\r\n?", "\n", text or "").strip()
    if not text:
        return []

    # paragraph first
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: List[str] = []

    def _emit_buffer(buffer: str):
        if not buffer.strip():
            return
        if len(buffer) <= chunk_size:
            chunks.append(buffer.strip())
            return
        start = 0
        length = len(buffer)
        step = max(1, chunk_size - chunk_overlap)
        while start < length:
            end = min(length, start + chunk_size)
            cut = buffer[start:end].strip()
            if cut:
                chunks.append(cut)
            if end >= length:
                break
            start += step

    current = ""
    for para in paras:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        _emit_buffer(current)
        if len(para) > chunk_size:
            # sentence fallback
            sents = re.split(r"(?<=[.!?。！？])\s+", para)
            sentence_buf = ""
            for sent in sents:
                sent = sent.strip()
                if not sent:
                    continue
                candidate_sent = f"{sentence_buf} {sent}".strip() if sentence_buf else sent
                if len(candidate_sent) <= chunk_size:
                    sentence_buf = candidate_sent
                else:
                    _emit_buffer(sentence_buf)
                    sentence_buf = sent
            _emit_buffer(sentence_buf)
            current = ""
        else:
            current = para

    _emit_buffer(current)
    return chunks


@dataclass
class EmbeddingBackend:
    model_name: str
    _model: Optional[SentenceTransformer] = None
    _MODEL_CACHE: ClassVar[Optional[Dict[str, Any]]] = None  # class-level lazy init
    _MODEL_CACHE_LOCK: ClassVar[threading.Lock] = threading.Lock()

    @property
    def model(self) -> SentenceTransformer:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers 未安装。请执行: pip install -r requirements.txt"
            )
        if EmbeddingBackend._MODEL_CACHE is None:
            EmbeddingBackend._MODEL_CACHE = {}
        if self._model is None:
            key = str(self.model_name).strip()
            with EmbeddingBackend._MODEL_CACHE_LOCK:
                cached = EmbeddingBackend._MODEL_CACHE.get(key)
                if cached is None:
                    # 静默加载，避免向用户暴露易引起误解的 checkpoint 兼容提示
                    out_buf = io.StringIO()
                    err_buf = io.StringIO()
                    st_logger = logging.getLogger("sentence_transformers")
                    tf_logger = logging.getLogger("transformers")
                    tfmu_logger = logging.getLogger("transformers.modeling_utils")
                    old_st_level = st_logger.level
                    old_tf_level = tf_logger.level
                    old_tfmu_level = tfmu_logger.level
                    old_hf_verbosity = None
                    try:
                        st_logger.setLevel(logging.ERROR)
                        tf_logger.setLevel(logging.ERROR)
                        tfmu_logger.setLevel(logging.ERROR)
                        if hf_logging is not None:
                            try:
                                old_hf_verbosity = hf_logging.get_verbosity()
                            except Exception:
                                old_hf_verbosity = None
                            hf_logging.set_verbosity_error()
                        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                            cached = SentenceTransformer(key)
                    except Exception:
                        # 出错时不吞掉真实异常
                        raise
                    finally:
                        st_logger.setLevel(old_st_level)
                        tf_logger.setLevel(old_tf_level)
                        tfmu_logger.setLevel(old_tfmu_level)
                        if hf_logging is not None and old_hf_verbosity is not None:
                            try:
                                hf_logging.set_verbosity(old_hf_verbosity)
                            except Exception:
                                pass
                    EmbeddingBackend._MODEL_CACHE[key] = cached
                self._model = cached
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)


def unload_embedding_model(model_name: Optional[str] = None) -> Dict:
    """
    卸载缓存中的 embedding 模型。
    - model_name 为 None: 卸载全部
    - model_name 非空: 仅卸载指定模型
    """
    unloaded: List[str] = []
    with EmbeddingBackend._MODEL_CACHE_LOCK:
        cache = EmbeddingBackend._MODEL_CACHE or {}
        if model_name is None:
            unloaded = list(cache.keys())
            cache.clear()
        else:
            key = str(model_name).strip()
            if key in cache:
                cache.pop(key, None)
                unloaded.append(key)
        EmbeddingBackend._MODEL_CACHE = cache

    # 尝试释放显存
    try:
        import gc
        gc.collect()
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        pass

    return {"unloaded": unloaded, "count": len(unloaded)}


def default_index_root() -> Path:
    root = Path(__file__).resolve().parent / "data" / "faiss_indexes"
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_faiss_index(
    documents: List[Dict],
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    index_name: str,
) -> Dict:
    if faiss is None:
        raise ImportError("faiss 未安装。请执行: pip install -r requirements.txt")
    if not documents:
        raise ValueError("No documents provided.")
    if not index_name.strip():
        raise ValueError("index_name must not be empty.")

    embedder = EmbeddingBackend(embedding_model)
    chunks: List[Dict] = []
    for doc_id, doc in enumerate(documents):
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        split_chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, chunk in enumerate(split_chunks):
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "doc_id": doc_id,
                    "source": doc.get("source", ""),
                    "title": doc.get("title", ""),
                    "text": chunk,
                    "position": i,
                }
            )

    if not chunks:
        raise ValueError("No chunks generated from documents.")

    chunk_texts = [x["text"] for x in chunks]
    vectors = embedder.encode(chunk_texts)
    dim = vectors.shape[1]

    # cosine with normalized vectors -> inner product
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    root = default_index_root()
    index_dir = root / index_name
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (index_dir / "meta.json").write_text(
        json.dumps(
            {
                "index_name": index_name,
                "embedding_model": embedding_model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "dim": dim,
                "documents_count": len(documents),
                "chunks_count": len(chunks),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "index_name": index_name,
        "index_dir": str(index_dir),
        "embedding_model": embedding_model,
        "chunks_count": len(chunks),
        "documents_count": len(documents),
    }


def load_index(index_name_or_path: str) -> Tuple[Any, List[Dict], Dict]:
    if faiss is None:
        raise ImportError("faiss 未安装。请执行: pip install -r requirements.txt")
    path = Path(index_name_or_path)
    if path.is_dir():
        index_dir = path
    else:
        index_dir = default_index_root() / index_name_or_path
    if not index_dir.exists():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")

    index = faiss.read_index(str(index_dir / "index.faiss"))
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    return index, chunks, meta


def search_index(
    index_name_or_path: str,
    query: str,
    top_k: int = 5,
) -> Dict:
    if not query.strip():
        raise ValueError("query must not be empty.")

    index, chunks, meta = load_index(index_name_or_path)
    embedder = EmbeddingBackend(meta["embedding_model"])
    qvec = embedder.encode([query])
    scores, indices = index.search(qvec, top_k)

    items: List[Dict] = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        items.append(
            {
                "score": float(score),
                "text": chunk["text"],
                "source": chunk.get("source", ""),
                "title": chunk.get("title", ""),
                "position": chunk.get("position", 0),
            }
        )

    context_lines: List[str] = []
    for i, item in enumerate(items, start=1):
        context_lines.append(
            f"[{i}] source={item['source']} score={item['score']:.4f}\n{item['text']}"
        )
    best_score = items[0]["score"] if items else 0.0

    return {
        "query": query,
        "top_k": top_k,
        "items": items,
        "rag_hit": len(items) > 0,
        "best_score": float(best_score),
        "context": "\n\n".join(context_lines).strip(),
    }


def resolve_lmstudio_model(base_url: str, timeout: int = 20) -> str:
    models = list_lmstudio_models(base_url=base_url, timeout=timeout)
    if not models:
        raise RuntimeError("LM Studio API 返回空模型列表。")
    return models[0]


def list_lmstudio_models(base_url: str, timeout: int = 10) -> List[str]:
    base = base_url.rstrip("/")
    out: List[str] = []

    # 1) LM Studio native v1 REST
    native_url = base + "/api/v1/models"
    try:
        resp = requests.get(native_url, timeout=timeout)
        if resp.ok:
            data = resp.json()
            models = data.get("models", [])
            for m in models:
                key = m.get("key") or m.get("id")
                if key:
                    out.append(str(key))
    except requests.RequestException:
        pass

    # 2) OpenAI-compatible fallback
    if not out:
        openai_url = base + "/v1/models"
        try:
            resp = requests.get(openai_url, timeout=timeout)
            if resp.ok:
                data = resp.json()
                models = data.get("data", [])
                for m in models:
                    model_id = m.get("id")
                    if model_id:
                        out.append(str(model_id))
        except requests.RequestException:
            pass

    # deduplicate
    seen = set()
    uniq = []
    for m in out:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


def unload_lmstudio_model(base_url: str, instance_id: str, timeout: int = 20) -> Dict:
    endpoint = base_url.rstrip("/") + "/api/v1/models/unload"
    payload = {"instance_id": instance_id}
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _normalize_text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item.get("text", ""))
                elif isinstance(item.get("content"), str):
                    parts.append(item.get("content", ""))
        return "\n".join([p for p in parts if p]).strip()
    if isinstance(value, dict):
        return _normalize_text_content(value.get("text") or value.get("content"))
    return str(value).strip()


def _pick_answer(content_text: str, reasoning_text: str) -> str:
    # 自动策略：优先内容，其次 reasoning
    return content_text or reasoning_text


def _extract_answer_from_chat_payload(data: Dict) -> Dict:
    message = data.get("choices", [{}])[0].get("message", {}) if isinstance(data, dict) else {}
    content_text = _normalize_text_content(message.get("content"))
    reasoning_text = _normalize_text_content(
        message.get("reasoning_content") or message.get("reasoning")
    )
    answer = _pick_answer(content_text, reasoning_text)
    return {
        "answer": answer.strip(),
        "content_text": content_text,
        "reasoning_text": reasoning_text,
    }


def _extract_answer_from_responses_payload(data: Dict) -> Dict:
    content_parts: List[str] = []
    reasoning_parts: List[str] = []

    output_text = _normalize_text_content(data.get("output_text") if isinstance(data, dict) else "")
    if output_text:
        content_parts.append(output_text)

    output = data.get("output", []) if isinstance(data, dict) else []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).lower()
            if item_type == "message":
                for c in item.get("content", []) or []:
                    if not isinstance(c, dict):
                        continue
                    ctype = str(c.get("type", "")).lower()
                    if ctype in {"output_text", "text"}:
                        text = _normalize_text_content(c.get("text"))
                        if text:
                            content_parts.append(text)
                    elif "reasoning" in ctype:
                        text = _normalize_text_content(c.get("text") or c.get("content"))
                        if text:
                            reasoning_parts.append(text)
            elif "reasoning" in item_type:
                text = _normalize_text_content(
                    item.get("reasoning_content")
                    or item.get("summary")
                    or item.get("content")
                    or item.get("text")
                )
                if text:
                    reasoning_parts.append(text)

    top_reasoning = _normalize_text_content(data.get("reasoning_content") if isinstance(data, dict) else "")
    if top_reasoning:
        reasoning_parts.append(top_reasoning)

    content_text = "\n".join([p for p in content_parts if p]).strip()
    reasoning_text = "\n".join([p for p in reasoning_parts if p]).strip()
    answer = _pick_answer(content_text, reasoning_text)
    return {
        "answer": answer.strip(),
        "content_text": content_text,
        "reasoning_text": reasoning_text,
    }


def _stream_chat_completions(
    endpoint: str,
    payload: Dict,
    timeout: int,
    emit_stream_log: bool,
) -> Dict:
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    stream_parts: List[str] = []
    with requests.post(endpoint, json=payload, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
            except Exception:
                continue
            delta = event.get("choices", [{}])[0].get("delta", {})
            c = _normalize_text_content(delta.get("content"))
            r = _normalize_text_content(delta.get("reasoning_content") or delta.get("reasoning"))
            if c:
                content_parts.append(c)
                stream_parts.append(c)
                if emit_stream_log:
                    print(c, end="", flush=True)
            if r:
                reasoning_parts.append(r)
                stream_parts.append(r)
                if emit_stream_log:
                    print(r, end="", flush=True)
    if emit_stream_log and (content_parts or reasoning_parts):
        print("")
    content_text = "".join(content_parts).strip()
    reasoning_text = "".join(reasoning_parts).strip()
    answer = _pick_answer(content_text, reasoning_text)
    return {
        "answer": answer,
        "content_text": content_text,
        "reasoning_text": reasoning_text,
        "stream_text": "".join(stream_parts).strip(),
        "raw": {
            "stream": True,
            "api_mode": "chat_completions",
            "content_text": content_text,
            "reasoning_text": reasoning_text,
        },
    }


def _stream_responses(
    endpoint: str,
    payload: Dict,
    timeout: int,
    emit_stream_log: bool,
) -> Dict:
    current_event = ""
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    stream_parts: List[str] = []
    final_raw: Dict[str, Any] = {}

    with requests.post(endpoint, json=payload, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=False):
            if raw_line is None:
                continue
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("event:"):
                current_event = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                event_data = json.loads(data_str)
            except Exception:
                continue

            if current_event.endswith(".completed"):
                if isinstance(event_data, dict):
                    final_raw = event_data.get("response", event_data)

            delta_text = _normalize_text_content(event_data.get("delta") if isinstance(event_data, dict) else "")
            if delta_text:
                if "reasoning" in current_event:
                    reasoning_parts.append(delta_text)
                else:
                    content_parts.append(delta_text)
                stream_parts.append(delta_text)
                if emit_stream_log:
                    print(delta_text, end="", flush=True)

    if emit_stream_log and (content_parts or reasoning_parts):
        print("")

    content_text = "".join(content_parts).strip()
    reasoning_text = "".join(reasoning_parts).strip()
    if final_raw:
        extracted = _extract_answer_from_responses_payload(final_raw)
        content_text = extracted.get("content_text") or content_text
        reasoning_text = extracted.get("reasoning_text") or reasoning_text
    answer = _pick_answer(content_text, reasoning_text)
    return {
        "answer": answer,
        "content_text": content_text,
        "reasoning_text": reasoning_text,
        "stream_text": "".join(stream_parts).strip(),
        "raw": final_raw or {
            "stream": True,
            "api_mode": "responses",
            "content_text": content_text,
            "reasoning_text": reasoning_text,
        },
    }


def lmstudio_chat(
    base_url: str,
    model: str,
    question: str,
    context: str = "",
    image_data_url: str = "",
    system_prompt: str = "You are a helpful assistant.",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    api_mode: str = "responses",
    stream: bool = False,
    emit_stream_log: bool = False,
    timeout: int = 120,
) -> Dict:
    if not model.strip():
        model = resolve_lmstudio_model(base_url, timeout=20)

    final_user_prompt = question.strip()
    if context.strip():
        final_user_prompt = (
            "请基于以下检索到的上下文回答问题。如果上下文不足，请明确说明。\n\n"
            f"【上下文】\n{context.strip()}\n\n"
            f"【问题】\n{question.strip()}"
        )

    user_content: Any = final_user_prompt
    if image_data_url.strip():
        # OpenAI-compatible multimodal format
        user_content = [
            {"type": "text", "text": final_user_prompt},
            {"type": "image_url", "image_url": {"url": image_data_url.strip()}},
        ]

    mode = (api_mode or "chat_completions").strip().lower()
    base = base_url.rstrip("/")
    if mode == "responses":
        endpoint = base + "/v1/responses"
        user_input_content: List[Dict[str, Any]] = [{"type": "input_text", "text": final_user_prompt}]
        if image_data_url.strip():
            user_input_content.append({"type": "input_image", "image_url": image_data_url.strip()})
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": user_input_content},
            ],
            "stream": bool(stream),
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_output_tokens"] = int(max_tokens)
    else:
        endpoint = base + "/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": bool(stream),
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

    try:
        if stream:
            if mode == "responses":
                result = _stream_responses(
                    endpoint=endpoint,
                    payload=payload,
                    timeout=timeout,
                    emit_stream_log=emit_stream_log,
                )
            else:
                result = _stream_chat_completions(
                    endpoint=endpoint,
                    payload=payload,
                    timeout=timeout,
                    emit_stream_log=emit_stream_log,
                )
            return {
                "answer": result["answer"],
                "raw": result["raw"],
                "model": model,
                "stream_text": result.get("stream_text", ""),
            }

        resp = requests.post(endpoint, json=payload, timeout=timeout)
    except requests.RequestException as e:
        # responses 端点不可用时，自动回退到 chat/completions（仍是 OpenAI 兼容）
        if mode == "responses":
            return lmstudio_chat(
                base_url=base_url,
                model=model,
                question=question,
                context=context,
                image_data_url=image_data_url,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                api_mode="chat_completions",
                stream=stream,
                emit_stream_log=emit_stream_log,
                timeout=timeout,
            )
        raise RuntimeError(
            f"无法连接 LM Studio API: {endpoint}。请确认 LM Studio 已启动本地服务并监听该地址。原始错误: {e}"
        ) from e
    resp.raise_for_status()
    data = resp.json()
    if mode == "responses":
        extracted = _extract_answer_from_responses_payload(data)
    else:
        extracted = _extract_answer_from_chat_payload(data)
    answer = extracted.get("answer", "").strip()
    return {"answer": answer, "raw": data, "model": model, "stream_text": answer}


def extract_answer_between_newlines(content: str) -> str:
    """
    返回完整 content；仅在明显“外壳包裹”场景下提取中间正文，避免误截断。
    """
    text = (content or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    # 优先处理 markdown fenced code block 包裹（如 ```text ... ```）
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 2 and lines[-1].strip() == "```":
            inner = "\n".join(lines[1:-1]).strip()
            if inner:
                return inner

    lines = text.split("\n")
    if len(lines) >= 3:
        first_line = lines[0].strip()
        last_line = lines[-1].strip()
        # 仅在首尾是明显包裹符号时提取中间
        wrappers = {"```", "<<<", ">>>", "---", "***"}
        if first_line in wrappers and last_line in wrappers:
            middle = "\n".join(lines[1:-1]).strip()
            if middle:
                return middle

    return text

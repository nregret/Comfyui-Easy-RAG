import { app } from "/scripts/app.js";

const TARGET_NODE = "RagPromptVectorStoreBuilder";
const MODE_CREATE_NEW = "create_new";
const MODE_USE_EXISTING = "use_existing";
const BUILD_MODE_NAMES = ["build_mode", "操作模式", "Build Mode"];
const INDEX_LIST_NAMES = ["index_list", "已有向量库", "Existing Index"];
const INDEX_NAME_NAMES = ["index_name", "创建向量库名", "Create Vector Store Name"];
const EMBEDDING_MODEL_NAMES = ["embedding_model", "向量模型", "Embedding Model"];

function getComboValues(widget) {
  if (!widget) return [];
  if (Array.isArray(widget.options?.values)) return widget.options.values;
  if (Array.isArray(widget.options)) return widget.options;
  return [];
}

function setComboValues(widget, values) {
  if (!widget) return;
  if (widget.options && Array.isArray(widget.options.values)) {
    widget.options.values = values;
    return;
  }
  if (widget.options && !Array.isArray(widget.options.values)) {
    widget.options.values = values;
    return;
  }
  widget.options = values;
}

function ensureWidgetValueInOptions(widget) {
  if (!widget) return;
  const values = getComboValues(widget);
  if (!Array.isArray(values) || values.length === 0) return;
  const current = String(widget.value ?? "");
  if (!values.includes(current)) {
    widget.value = values[0];
  }
}

function getWidgetByNames(node, names) {
  const set = new Set(names.map((x) => String(x || "").trim().toLowerCase()));
  return (node.widgets || []).find((w) => set.has(String(w?.name || "").trim().toLowerCase()));
}

function setWidgetVisible(widget, visible) {
  if (!widget) return;
  if (!widget.__easyrag_wrapped_compute_size) {
    const originalComputeSize = widget.computeSize;
    widget.__easyrag_original_computeSize = originalComputeSize;
    widget.computeSize = function (...args) {
      if (this.hidden) return [0, -4];
      if (typeof originalComputeSize === "function") {
        return originalComputeSize.apply(this, args);
      }
      return [this.width || 0, 20];
    };
    widget.__easyrag_wrapped_compute_size = true;
  }
  widget.hidden = !visible;
}

function normalizeMode(value) {
  const v = String(value || "").trim().toLowerCase();
  if (v.includes("use_existing") || v.includes("use existing") || v.includes("已有")) return MODE_USE_EXISTING;
  if (v.includes("create_new") || v.includes("create new") || v.includes("新建")) return MODE_CREATE_NEW;
  if (v === MODE_USE_EXISTING) return MODE_USE_EXISTING;
  return MODE_CREATE_NEW;
}

function preferChineseLocale() {
  const normalize = (value) => String(value || "").toLowerCase();
  try {
    const v = normalize(app?.ui?.settings?.getSettingValue?.("Comfy.Locale"));
    if (v.includes("zh")) return true;
    if (v.includes("en")) return false;
  } catch (_) {}
  try {
    const v = normalize(app?.extensionManager?.settingStore?.get?.("Comfy.Locale"));
    if (v.includes("zh")) return true;
    if (v.includes("en")) return false;
  } catch (_) {}
  try {
    const v = normalize(globalThis?.document?.documentElement?.lang);
    if (v.includes("zh")) return true;
    if (v.includes("en")) return false;
  } catch (_) {}
  try {
    const v = normalize(globalThis?.localStorage?.getItem?.("Comfy.Settings.Comfy.Locale"));
    if (v.includes("zh")) return true;
    if (v.includes("en")) return false;
  } catch (_) {}
  try {
    const v = normalize(globalThis?.navigator?.language);
    if (v.includes("zh")) return true;
  } catch (_) {}
  return false;
}

function localizeModeOptions(node, modeWidget) {
  // We no longer manually rewrite options here. 
  // Native nodeDefs.json is sufficient and more stable for modern ComfyUI.
  return;
}

function coerceModeWidget(node) {
  const modeWidget = getWidgetByNames(node, BUILD_MODE_NAMES);
  if (!modeWidget) return;
  
  const zh = preferChineseLocale();
  const val = String(modeWidget.value || "").toLowerCase();
  
  // Backward compatibility: map any incoming value to the 2 current options
  if (val.includes("use") || val.includes("已有")) {
    modeWidget.value = zh ? "使用已有向量库" : "Use Existing";
  } else if (val.includes("create") || val.includes("新建")) {
    modeWidget.value = zh ? "新建向量库" : "Create New";
  }
}

function applyModeUI(node) {
  coerceModeWidget(node);
  ensureWidgetValueInOptions(getWidgetByNames(node, EMBEDDING_MODEL_NAMES));
  ensureWidgetValueInOptions(getWidgetByNames(node, INDEX_LIST_NAMES));
  const modeWidget = getWidgetByNames(node, BUILD_MODE_NAMES);
  const indexListWidget = getWidgetByNames(node, INDEX_LIST_NAMES);
  const indexNameWidget = getWidgetByNames(node, INDEX_NAME_NAMES);
  if (!modeWidget || !indexListWidget || !indexNameWidget) return;

  const mode = normalizeMode(modeWidget.value);
  const usingExisting = mode === MODE_USE_EXISTING;
  setWidgetVisible(indexListWidget, usingExisting);
  setWidgetVisible(indexNameWidget, !usingExisting);
}

app.registerExtension({
  name: "rag.index-mode-ui",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TARGET_NODE) return;

    // Fix: Prune the list of options at registration level to prevent UI clutter (6 options bug).
    // This happens once per node type and ensures the widget's 'source of truth' is clean.
    const buildModeInput = nodeData.input?.required?.build_mode;
    if (buildModeInput && Array.isArray(buildModeInput[0])) {
      const zh = preferChineseLocale();
      // Replace the global options list for this node type
      if (zh) {
        buildModeInput[0] = ["新建向量库", "使用已有向量库"];
      } else {
        buildModeInput[0] = ["Create New", "Use Existing"];
      }
    }

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

      const modeWidget = getWidgetByNames(this, BUILD_MODE_NAMES);
      if (modeWidget) {
        const origModeCallback = modeWidget.callback;
        modeWidget.callback = (...args) => {
          if (origModeCallback) origModeCallback.apply(modeWidget, args);
          applyModeUI(this);
          if (app.graph) app.graph.setDirtyCanvas(true, true);
        };
      }

      applyModeUI(this);
      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
      applyModeUI(this);
      return r;
    };
  },
});


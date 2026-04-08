import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";

const TARGET_NODE = "RagPromptVectorStoreBuilder";

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

async function fetchIndexes() {
  const resp = await api.fetchApi("/easyrag/indexes");
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}`);
  }
  const data = await resp.json();
  const items = Array.isArray(data?.items) ? data.items : [];
  return items.length ? items : ["default_index"];
}

app.registerExtension({
  name: "ragprompt.index-refresh",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TARGET_NODE) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

      const refresh = async () => {
        try {
          const values = await fetchIndexes();
          const indexWidget = (this.widgets || []).find((w) => w.name === "index_list");
          if (!indexWidget) return;

          setComboValues(indexWidget, values);
          const current = String(indexWidget.value || "");
          if (!values.includes(current) && values.length > 0) {
            indexWidget.value = values[0];
          }
          if (app.graph) {
            app.graph.setDirtyCanvas(true, true);
          }
        } catch (err) {
          console.warn("[EasyRAG] Failed to refresh index_list:", err);
        }
      };

      const origOnExecuted = this.onExecuted;
      this.onExecuted = function (message) {
        const out = origOnExecuted ? origOnExecuted.call(this, message) : undefined;
        refresh();
        return out;
      };

      refresh();
      return r;
    };
  },
});


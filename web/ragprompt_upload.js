import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// 👇 已改成你节点的真实名称
const TARGET_NODE_NAMES = new Set([
  "RagPromptDocumentLoader",
]);

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

app.registerExtension({
  name: "ragprompt.document-upload",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!TARGET_NODE_NAMES.has(nodeData.name)) {
      return;
    }

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

      const hasButton = (this.widgets || []).some((w) => w.name === "upload_document");
      if (hasButton) return r;

      const self = this;
      this.addWidget("button", "upload_document", "上传文档", async () => {
        const picker = document.createElement("input");
        picker.type = "file";
        picker.accept = ".txt,.json,.md,.pdf";
        picker.style.display = "none";

        picker.onchange = async () => {
          const file = picker.files && picker.files[0];
          if (!file) return;

          try {
            const formData = new FormData();
            formData.append("image", file);
            formData.append("type", "input");

            const resp = await api.fetchApi("/upload/image", {
              method: "POST",
              body: formData,
            });
            if (!resp.ok) {
              const text = await resp.text();
              throw new Error(text || `HTTP ${resp.status}`);
            }
            const data = await resp.json();
            const savedName = data?.name || file.name;

            const docWidget = (self.widgets || []).find((w) => w.name === "document");
            if (docWidget) {
              const values = getComboValues(docWidget).slice();
              if (!values.includes(savedName)) {
                values.push(savedName);
                values.sort();
                setComboValues(docWidget, values);
              }
              docWidget.value = savedName;
            }

            if (app.graph) {
              app.graph.setDirtyCanvas(true, true);
            }
          } catch (err) {
            console.error("[RagPrompt] 文档上传失败:", err);
            alert(`文档上传失败: ${err?.message || err}`);
          } finally {
            picker.remove();
          }
        };

        document.body.appendChild(picker);
        picker.click();
      });

      return r;
    };
  },
});
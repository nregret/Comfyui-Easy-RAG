import { app } from "/scripts/app.js";

const TARGET_NODE_NAMES = new Set([
  "RagPromptLMStudioChatAdvanced",
  "RagPromptLMStudioChatSimple",
  "RagPromptExternalChatAdvanced",
]);

const BASE_IMAGE_INPUT = "image";
const DYNAMIC_IMAGE_PREFIX = "image_";

function parseImageIndex(name) {
  if (name === BASE_IMAGE_INPUT) return 1;
  if (typeof name !== "string" || !name.startsWith(DYNAMIC_IMAGE_PREFIX)) {
    return Number.MAX_SAFE_INTEGER;
  }
  const n = Number.parseInt(name.slice(DYNAMIC_IMAGE_PREFIX.length), 10);
  return Number.isFinite(n) ? n : Number.MAX_SAFE_INTEGER;
}

function isManagedImageInput(input) {
  if (!input || input.type !== "IMAGE") return false;
  const name = String(input.name || "");
  return name === BASE_IMAGE_INPUT || /^image_\d+$/.test(name);
}

function getManagedImageInputs(node) {
  const inputs = Array.isArray(node.inputs) ? node.inputs : [];
  return inputs
    .filter(isManagedImageInput)
    .sort((a, b) => parseImageIndex(a.name) - parseImageIndex(b.name));
}

function nextImageInputName(node) {
  const managed = getManagedImageInputs(node);
  let maxIndex = 1;
  for (const input of managed) {
    const index = parseImageIndex(input.name);
    if (Number.isFinite(index) && index > maxIndex) {
      maxIndex = index;
    }
  }
  return `${DYNAMIC_IMAGE_PREFIX}${maxIndex + 1}`;
}

function syncDynamicImageInputs(node) {
  if (!node || node.__easyRagDynamicImageBusy) return;
  node.__easyRagDynamicImageBusy = true;

  try {
    let managed = getManagedImageInputs(node);

    if (!managed.length) {
      node.addInput(BASE_IMAGE_INPUT, "IMAGE");
      managed = getManagedImageInputs(node);
    }

    while (managed.length && managed[managed.length - 1].link != null) {
      node.addInput(nextImageInputName(node), "IMAGE");
      managed = getManagedImageInputs(node);
    }

    for (let i = managed.length - 1; i >= 1; i--) {
      const current = managed[i];
      const previous = managed[i - 1];
      if (current.link == null && previous.link == null) {
        const index = node.inputs.indexOf(current);
        if (index >= 0) {
          node.removeInput(index);
          managed = getManagedImageInputs(node);
        }
      } else {
        break;
      }
    }

    node.graph?.setDirtyCanvas(true, true);
  } finally {
    node.__easyRagDynamicImageBusy = false;
  }
}

function installDynamicImageBehavior(node) {
  const originalOnConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function () {
    const result = typeof originalOnConnectionsChange === "function"
      ? originalOnConnectionsChange.apply(this, arguments)
      : undefined;
    syncDynamicImageInputs(this);
    return result;
  };

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function () {
    const result = typeof originalOnConfigure === "function"
      ? originalOnConfigure.apply(this, arguments)
      : undefined;
    syncDynamicImageInputs(this);
    return result;
  };

  syncDynamicImageInputs(node);
}

app.registerExtension({
  name: "rag.dynamic-images",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!TARGET_NODE_NAMES.has(nodeData.name)) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = typeof originalOnNodeCreated === "function"
        ? originalOnNodeCreated.apply(this, arguments)
        : undefined;
      installDynamicImageBehavior(this);
      return result;
    };
  },
});

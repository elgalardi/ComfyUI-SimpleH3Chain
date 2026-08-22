import { app } from "../../scripts/app.js";

const NODE_NAME = "SimpleH3ChainContext";

function widget(node, name) {
    return node?.widgets?.find((item) => item?.name === name);
}

function hideWidget(item) {
    if (!item || item._simpleH3Hidden) return;
    item._simpleH3Hidden = true;
    item.hidden = true;
    item.options ??= {};
    item.options.hidden = true;

    if (!window.LiteGraph?.vueNodesMode) {
        item._simpleH3ComputeSize = item.computeSize;
        item._simpleH3Draw = item.draw;
        item.computeSize = () => [0, -4];
        item.draw = () => {};
    }
    if (item.element) item.element.style.display = "none";
}

function showWidget(item) {
    if (!item || !item._simpleH3Hidden) return;
    item._simpleH3Hidden = false;
    item.hidden = false;
    item.options ??= {};
    item.options.hidden = false;

    if (!window.LiteGraph?.vueNodesMode) {
        if (item._simpleH3ComputeSize === undefined) delete item.computeSize;
        else item.computeSize = item._simpleH3ComputeSize;
        if (item._simpleH3Draw === undefined) delete item.draw;
        else item.draw = item._simpleH3Draw;
        delete item._simpleH3ComputeSize;
        delete item._simpleH3Draw;
    }
    if (item.element) item.element.style.display = "";
}

function refresh(node) {
    const contextType = widget(node, "context_type");
    const feather = widget(node, "audio_feather_ticks");
    if (!contextType || !feather) return;

    if (String(contextType.value) === "masked_av") showWidget(feather);
    else hideWidget(feather);

    const computed = node.computeSize?.();
    if (computed && node.setSize) {
        node.setSize([Math.max(node.size?.[0] ?? 0, computed[0]), computed[1]]);
    }
    node.setDirtyCanvas?.(true, true);
}

function mount(node) {
    if (!node || node.comfyClass !== NODE_NAME || node._simpleH3ContextVisibility) return;
    node._simpleH3ContextVisibility = true;
    const contextType = widget(node, "context_type");
    if (!contextType) return;

    const originalCallback = contextType.callback;
    contextType.callback = function () {
        const result = originalCallback?.apply(this, arguments);
        queueMicrotask(() => refresh(node));
        return result;
    };
    queueMicrotask(() => refresh(node));
}

app.registerExtension({
    name: "SimpleH3Chain.contextVisibility",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            queueMicrotask(() => mount(this));
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = configured?.apply(this, arguments);
            queueMicrotask(() => {
                mount(this);
                refresh(this);
            });
            return result;
        };
    },
    async nodeCreated(node) {
        mount(node);
    },
    async afterConfigureGraph() {
        for (const node of app.graph?._nodes ?? []) mount(node);
    },
});

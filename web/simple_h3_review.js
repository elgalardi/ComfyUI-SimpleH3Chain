import {app} from "/scripts/app.js";
import {api} from "/scripts/api.js";

const NODE_NAME = "SimpleH3ChainReview";
const mounted = new Set();
let pollTimer = null;

function nodeType(node) {
    return node?.comfyClass ?? node?.type ?? "";
}

function nodes() {
    return app.graph?._nodes ?? [];
}

function findNode(data) {
    const exact = app.graph?.getNodeById?.(data?.node_id);
    if (nodeType(exact) === NODE_NAME) return exact;
    const execution = app.graph?.getNodeById?.(data?.execution_id);
    if (nodeType(execution) === NODE_NAME) return execution;
    return nodes().find((node) => nodeType(node) === NODE_NAME) ?? null;
}

function mediaKey(item, revision = "") {
    if (!item?.filename) return "";
    return [item.type ?? "output", item.subfolder ?? "", item.filename, revision].join("|");
}

function mediaUrl(item, revision = "") {
    if (!item?.filename) return "";
    const query = new URLSearchParams({
        filename: item.filename,
        subfolder: item.subfolder ?? "",
        type: item.type ?? "output",
        // Stable for repeated status polling, but changes when the backend
        // publishes a genuinely new synchronized preview.
        v: String(revision),
    });
    return api.apiURL(`/view?${query}`);
}

function loadMedia(node, item, revision = "") {
    const key = mediaKey(item, revision);
    if (!key || node._simpleH3MediaKey === key) return false;
    node._simpleH3MediaKey = key;
    node._simpleH3Video.src = mediaUrl(item, revision);
    node._simpleH3Video.load();
    return true;
}

function style(element, values) {
    Object.assign(element.style, values);
    return element;
}

function button(label, className, action) {
    const element = document.createElement("button");
    element.type = "button";
    element.textContent = label;
    element.className = className;
    element.addEventListener("click", action);
    return element;
}

function setBusy(node, busy, message = "") {
    node._simpleH3Busy = busy;
    for (const item of node._simpleH3Buttons ?? []) item.disabled = busy;
    if (message) node._simpleH3Status.textContent = message;
}

async function decide(node, action) {
    const review = node._simpleH3Review;
    if (!review?.token || node._simpleH3Busy) return;
    setBusy(node, true, `Sending ${action}…`);
    try {
        const body = {
            token: review.token,
            action,
            scene_prompt: node._simpleH3Prompt.value,
            seed: node._simpleH3Seed.value,
        };
        const response = await api.fetchApi("/simple_h3_chain/review", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(body),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error ?? "Review request failed.");
        node._simpleH3Status.textContent = action === "approve"
            ? "Approved — continuing…"
            : action === "stop"
                ? "Approved — stopping…"
                : `Regenerating with seed ${result.seed}…`;
    } catch (error) {
        setBusy(node, false, error.message);
    }
}

function showReview(data) {
    const node = findNode(data);
    if (!node) return;
    if (!node._simpleH3Mounted) mount(node);
    const isNewReview = node._simpleH3Review?.token !== data.token;
    node._simpleH3Review = data;
    node._simpleH3Title.textContent = `Scene ${data.clip_index}/${data.clip_count} — ${data.shot_id}`;
    // Polling must never erase edits the user is preparing for Edit + Retry.
    if (isNewReview) {
        node._simpleH3Prompt.value = data.scene_prompt ?? "";
        node._simpleH3Seed.value = String(data.seed ?? "0");
    }
    node._simpleH3Status.textContent = data.warning || "Ready for review.";
    node._simpleH3Badge.textContent = data.has_audio ? "video + audio" : "video";
    loadMedia(node, data.video, `${data.token}:${data.preview_revision ?? 0}`);
    if (isNewReview) setBusy(node, Boolean(data.auto_continue));
}

function resolveReview(data) {
    const node = findNode(data);
    if (!node?._simpleH3Mounted) return;
    if (data.final_video) {
        loadMedia(node, data.final_video, `final:${data.token ?? ""}`);
        node._simpleH3Badge.textContent = "final assembled video";
    } else if (data.partial_video) {
        loadMedia(node, data.partial_video, `partial:${data.token ?? ""}`);
        node._simpleH3Badge.textContent = "partial assembled video";
    }
    node._simpleH3Status.textContent = data.status ?? "Review resolved.";
    setBusy(node, true);
}

async function fetchPending() {
    try {
        const response = await api.fetchApi("/simple_h3_chain/reviews");
        if (!response.ok) return;
        const data = await response.json();
        for (const review of data.reviews ?? []) showReview(review);
    } catch (_error) {
        // ComfyUI may be restarting; the websocket/status event retries later.
    }
}

function updatePolling() {
    if (mounted.size && !pollTimer) {
        pollTimer = setInterval(fetchPending, 2000);
    } else if (!mounted.size && pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

function mount(node) {
    if (!node || node._simpleH3Mounted) return;
    node._simpleH3Mounted = true;

    const root = style(document.createElement("section"), {
        display: "flex", flexDirection: "column", gap: "8px",
        width: "100%", height: "100%", boxSizing: "border-box",
        padding: "8px", color: "#ddd", background: "#202124",
        borderRadius: "7px", overflow: "hidden",
    });
    for (const event of ["pointerdown", "pointerup", "click", "dblclick", "wheel"])
        root.addEventListener(event, (value) => value.stopPropagation());

    const header = style(document.createElement("div"), {
        display: "flex", justifyContent: "space-between", gap: "8px",
        fontWeight: "700",
    });
    const title = document.createElement("span");
    title.textContent = "Waiting for a generated scene";
    const badge = document.createElement("span");
    badge.textContent = "preview";
    badge.style.color = "#9fb7e9";
    header.append(title, badge);

    const video = style(document.createElement("video"), {
        display: "block", width: "100%", minHeight: "220px",
        maxHeight: "520px", background: "#080808", objectFit: "contain",
        borderRadius: "5px",
    });
    video.controls = true;
    video.playsInline = true;
    video.preload = "metadata";

    const prompt = style(document.createElement("textarea"), {
        width: "100%", minHeight: "110px", resize: "vertical",
        boxSizing: "border-box", color: "#eee", background: "#151619",
        border: "1px solid #51545c", borderRadius: "5px", padding: "7px",
    });
    prompt.placeholder = "The current scene prompt appears here for editing.";

    const seedRow = style(document.createElement("label"), {
        display: "flex", alignItems: "center", gap: "8px",
    });
    seedRow.append("Seed");
    const seed = style(document.createElement("input"), {
        flex: "1", color: "#eee", background: "#151619",
        border: "1px solid #51545c", borderRadius: "4px", padding: "5px",
    });
    seed.type = "text";
    seedRow.append(seed);

    const actions = style(document.createElement("div"), {
        display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px",
    });
    const approve = button("Approve & continue", "simple-h3-approve", () => decide(node, "approve"));
    const retry = button("Edit + retry", "simple-h3-retry", () => decide(node, "retry"));
    const reroll = button("Reroll seed", "simple-h3-reroll", () => decide(node, "reroll"));
    const stop = button("Approve & stop", "simple-h3-stop", () => decide(node, "stop"));
    for (const item of [approve, retry, reroll, stop]) style(item, {
        padding: "7px", color: "#eee", background: "#363940",
        border: "1px solid #5e6470", borderRadius: "5px", cursor: "pointer",
    });
    approve.style.background = "#285b3d";
    stop.style.background = "#64333d";
    actions.append(approve, retry, reroll, stop);

    const status = style(document.createElement("div"), {
        minHeight: "20px", color: "#aeb7c8", whiteSpace: "pre-wrap",
    });
    status.textContent = "The player will receive each saved scene automatically.";

    root.append(header, video, prompt, seedRow, actions, status);
    const widget = node.addDOMWidget("simple_h3_review", "simple-h3-review", root, {
        serialize: false, hideOnZoom: false, getMinHeight: () => 500,
    });
    widget.serialize = false;

    node._simpleH3Title = title;
    node._simpleH3Badge = badge;
    node._simpleH3Video = video;
    node._simpleH3Prompt = prompt;
    node._simpleH3Seed = seed;
    node._simpleH3Buttons = [approve, retry, reroll, stop];
    node._simpleH3Status = status;
    node.setSize?.([Math.max(node.size?.[0] ?? 560, 560), Math.max(node.size?.[1] ?? 720, 720)]);
    mounted.add(node);
    updatePolling();

    const removed = node.onRemoved;
    node.onRemoved = function () {
        mounted.delete(this);
        updatePolling();
        return removed?.apply(this, arguments);
    };
    fetchPending();
}

api.addEventListener("simple_h3_chain_review", (event) => showReview(event.detail));
api.addEventListener("simple_h3_chain_review_resolved", (event) => resolveReview(event.detail));
api.addEventListener("status", fetchPending);

app.registerExtension({
    name: "simple_h3_chain.review",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            setTimeout(() => mount(this), 0);
            return result;
        };
    },
    async nodeCreated(node) {
        if (nodeType(node) === NODE_NAME) mount(node);
    },
    async afterConfigureGraph() {
        for (const node of nodes()) if (nodeType(node) === NODE_NAME) mount(node);
        await fetchPending();
    },
});

import { app } from "../../scripts/app.js";

function installFullSourcePreview(node) {
    const preview = node.widgets?.find((w) => w.name === "videopreview");
    if (!preview?.updateSource || preview._simpleH3FullSource) return;
    const updateSource = preview.updateSource;
    preview.updateSource = function (...args) {
        if (node.properties?.simple_h3_full_source_preview && this.value?.params) {
            Object.assign(this.value.params, {
                frame_load_cap: 0, skip_first_frames: 0, start_time: 0,
                select_every_nth: 1, force_rate: 0,
            });
        }
        return updateSource.apply(this, args);
    };
    preview.callback = preview.updateSource;
    preview._simpleH3FullSource = true;
}

app.registerExtension({
    name: "SimpleH3.FullSourceVideoPreview",
    nodeCreated(node) {
        if (node.comfyClass === "VHS_LoadVideo") installFullSourcePreview(node);
    },
    loadedGraphNode(node) {
        if (node.comfyClass !== "VHS_LoadVideo") return;
        installFullSourcePreview(node);
        if (node.properties?.simple_h3_full_source_preview) {
            node.widgets?.find((w) => w.name === "videopreview")?.updateSource?.();
        }
    },
});

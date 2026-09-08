import { app } from "../../scripts/app.js";

const AUDIO_MODES = new Set([
    "source_track",
    "generated_audio",
    "source_plus_timeline",
]);

app.registerExtension({
    name: "SimpleH3Chain.workflowMigration",
    beforeConfigureGraph(graphData) {
        for (const node of graphData?.nodes ?? []) {
            if (node.type !== "SimpleH3CompactContinuousPlanJSON") continue;
            // The former combo serialized seconds as a string, in the same slot.
            const values = node.widgets_values;
            if (Array.isArray(values) && typeof values[0] === "string") {
                const seconds = Number(values[0]);
                if (Number.isFinite(seconds)) values[0] = seconds;
            }
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SimpleH3ChainPlan") return;

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const values = Array.isArray(info?.widgets_values)
                ? [...info.widgets_values]
                : null;

            if (values && !AUDIO_MODES.has(values[2])) {
                // v1/v3: width, height, transition, context, audio, output
                if (values[2] === "h3_clean_cuts" && AUDIO_MODES.has(values[4])) {
                    info = { ...info, widgets_values: [values[0], values[1], values[4], values[5]] };
                // Short-lived v2: width, height, context, audio, output
                } else if (AUDIO_MODES.has(values[3])) {
                    info = { ...info, widgets_values: [values[0], values[1], values[3], values[4]] };
                // Variant without a serialized context selector.
                } else if (values[2] === "h3_clean_cuts" && AUDIO_MODES.has(values[3])) {
                    info = { ...info, widgets_values: [values[0], values[1], values[3], values[4]] };
                }
            }

            return originalOnConfigure?.apply(this, [info]);
        };
    },
});

// delivered_frames is commonly converted to an input, but LiteGraph still
// serializes its hidden numeric widget. Older experimental workflow exports
// omitted that placeholder and shifted window_frames/filename/save_output.
app.registerExtension({
    name: "SimpleH3Chain.refinedPreviewMigration",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SimpleH3FinalWindowPreviewAssemble") return;
        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const values = Array.isArray(info?.widgets_values)
                ? [...info.widgets_values]
                : null;
            if (values?.length === 3 && [90, 141, 192, 243, 294, 345, 396].includes(values[0])) {
                info = {...info, widgets_values: [1, ...values]};
            }
            return originalOnConfigure?.apply(this, [info]);
        };
    },
});

import { app } from "../../scripts/app.js";

const AUDIO_MODES = new Set([
    "source_track",
    "generated_audio",
    "source_plus_timeline",
]);

app.registerExtension({
    name: "SimpleH3Chain.workflowMigration",
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

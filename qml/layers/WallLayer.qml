import QtQuick 6.5
import QtQuick3D 6.5

// WallLayer: computes wall metrics & joint policy for debugSnapshot().
// Rendering stays minimal; this file focuses on deterministic metrics in GL units.
Node {
    id: layer
    objectName: "wallLayer"

    // Model contract:
    // - model.count : int
    // - model.get(i): QVariantMap with fields:
    //     type=="wall", polyline: [[x_mm,y_mm,z_mm], ...] (z usually 0),
    //     thickness (mm) OR radius (mm), height (mm),
    //     visible: bool, missing: bool
    property var model

    // Constants table injected from Python via context:
    // must be accessed via bracket notation (e.g. constants["WSCALE"])
    property var constants

    // World scale: GL = mm * wscale
    readonly property real wscale: (constants && constants["WSCALE"] !== undefined)
                                   ? Number(constants["WSCALE"]) : 0.02

    // -------- Helpers ----------------------------------------------------

    // Return only "wall" entries from the model, preserving insertion order.
    function _items() {
        if (!model) return []
        const n = (model.count !== undefined) ? model.count : 0
        const out = []
        for (let i = 0; i < n; ++i) {
            const e = model.get ? model.get(i) : null
            if (!e || e.type !== "wall") continue
            out.push(e)
        }
        return out
    }

    // Per-entry visibility helpers (kept for consistency across layers)
    function renderEnabled(entry) {
        return entry && !entry.missing && (entry.visible === undefined || entry.visible === true)
    }
    function brightness(entry) {
        return (entry && entry.visible === false) ? 0.5 : 1.0
    }

    // -------- Debug snapshot (deterministic GL values) -------------------

    function debugSnapshot() {
        const data = _items()

        // Deterministic empty snapshot
        const emptySnapshot = {
            "lengthsGL": [],
            "thicknessGL": 0.0,
            "heightGL": 0.0,
            "joint": "miter",
            "caps": "none"
        }
        if (!data || data.length === 0)
            return emptySnapshot

        // Numerics
        const EPS_MM = 1e-6                          // mm tolerance on raw data
        const EPS_GL = EPS_MM * wscale               // same tolerance in GL space
        const ACUTE_RAD = 20.0 * Math.PI / 180.0     // acute threshold for miter→butt

        // Accumulators (global across all wall entries; tests use one entry)
        const lengthsGL = []
        let thicknessGL = 0.0
        let heightGL = 0.0
        let joint = "miter"  // fallback to "butt" if any corner is acute/degenerate

        // Process each wall entry in insertion order
        for (let i = 0; i < data.length; ++i) {
            const entry = data[i]
            if (!renderEnabled(entry)) {
                // Even if not rendered, keep metrics conservative (no-op),
                // but we do not remove previously accumulated values.
            }

            // --- Polyline → GL points (skip invalid points) ---
            const poly = entry.polyline || []
            if (poly.length < 2) continue

            const pts = []
            for (let j = 0; j < poly.length; ++j) {
                const p = poly[j]
                if (!p || p.length < 3) continue
                // Convert mm to GL; use Number() to avoid stringly-typed values
                pts.push([ Number(p[0]) * wscale, Number(p[1]) * wscale, Number(p[2]) * wscale ])
            }
            if (pts.length < 2) continue

            // --- Segment lengths (skip near-zero) ---
            for (let j = 0; j < pts.length - 1; ++j) {
                const a = pts[j], b = pts[j + 1]
                const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2]
                const seg = Math.hypot(dx, dy, dz)
                if (!(seg > 0)) { // NaN or zero
                    joint = "butt"
                    continue
                }
                if (seg < EPS_GL) {
                    // Treat as degenerate; do not accumulate length
                    joint = "butt"
                    continue
                }
                lengthsGL.push(seg)
            }

            // --- Corner angle check at each interior vertex ---
            // Use forward vectors: u = (B - A), v = (C - B)
            // theta = acos( clamp( dot(û, ṽ), -1, 1 ) ), θ in [0, π]
            // If θ < 20°, the miter would spike → fallback to "butt".
            for (let j = 1; j < pts.length - 1; ++j) {
                const A = pts[j - 1], B = pts[j], C = pts[j + 1]

                const ux = B[0] - A[0], uy = B[1] - A[1], uz = B[2] - A[2]
                const vx = C[0] - B[0], vy = C[1] - B[1], vz = C[2] - B[2]

                const um = Math.hypot(ux, uy, uz)
                const vm = Math.hypot(vx, vy, vz)

                if (!(um > 0) || !(vm > 0)) {
                    // Near-zero leg: treat as butt
                    joint = "butt"
                    continue
                }

                // Normalized dot with clamp to avoid NaN from FP noise
                const dot = (ux / um) * (vx / vm) + (uy / um) * (vy / vm) + (uz / um) * (vz / vm)
                const clamped = Math.max(-1.0, Math.min(1.0, dot))
                let theta = Math.acos(clamped)    // 0..π
                if (!isFinite(theta)) {
                    // Defensive: numerical glitch → assume worst case
                    joint = "butt"
                    continue
                }

                if (theta < ACUTE_RAD) {
                    joint = "butt"
                    break   // one acute corner is enough to mark global "butt"
                }
            }

            // --- Thickness & height (in GL) ---
            // thickness: prefer explicit 'thickness'; fallback to 'radius * 2'
            const thicknessMm =
                (entry.thickness !== undefined && entry.thickness !== null) ? Number(entry.thickness) :
                (entry.radius    !== undefined && entry.radius    !== null) ? Number(entry.radius) * 2.0 :
                0.0
            thicknessGL = thicknessMm * wscale

            const heightMm = (entry.height !== undefined && entry.height !== null) ? Number(entry.height) : 0.0
            heightGL = heightMm * wscale
        }

        // Deterministic return object (arrays preserve insertion order)
        return {
            "lengthsGL": lengthsGL,
            "thicknessGL": thicknessGL,
            "heightGL": heightGL,
            "joint": joint,
            "caps": "none"
        }
    }
}

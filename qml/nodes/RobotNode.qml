import QtQuick 6.5
import QtQuick3D 6.5

Node {
    id: robot
    property var model

    function _item() {
        if (!model) {
            return null
        }
        const count = model.count !== undefined ? model.count : 0
        for (let i = 0; i < count; ++i) {
            const entry = model.get ? model.get(i) : null
            if (!entry || entry.type !== "robot") {
                continue
            }
            if (entry.missing) {
                continue
            }
            return entry
        }
        return null
    }

    function renderEnabled(entry) {
        return entry ? (!entry.missing && entry.visible) : false
    }

    function brightness(entry) {
        return entry && entry.visible ? 1.0 : 0.5
    }

    function debugSnapshot() {
        const entry = _item()
        if (!entry) {
            return {}
        }
        const thetaRad = entry.theta !== undefined && entry.theta !== null ? entry.theta : 0.0
        return {
            "thetaDeg": thetaRad * 180.0 / Math.PI,
            "holdingState": !!entry.holding
        }
    }
}

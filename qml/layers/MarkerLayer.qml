import QtQuick 6.5
import QtQuick3D 6.5

Node {
    id: layer
    property var model
    property var constants
    property string markerType: "apriltag"

    readonly property real wscale: constants && constants["WSCALE"] !== undefined ? constants["WSCALE"] : 1.0

    function _items() {
        if (!model) {
            return []
        }
        const count = model.count !== undefined ? model.count : 0
        const items = []
        for (let i = 0; i < count; ++i) {
            const entry = model.get ? model.get(i) : null
            if (!entry || entry.type !== markerType) {
                continue
            }
            items.push(entry)
        }
        return items
    }

    function renderEnabled(entry) {
        return !entry.missing && entry.visible
    }

    function brightness(entry) {
        return entry.visible ? 1.0 : 0.5
    }

    function debugSnapshot() {
        const data = _items()
        if (data.length === 0) {
            return { "thetaDeg": [], "sizeGL": [] }
        }
        const theta = []
        const size = []
        for (let i = 0; i < data.length; ++i) {
            const entry = data[i]
            const thetaRad = entry.theta !== undefined && entry.theta !== null ? entry.theta : 0.0
            theta.push(thetaRad * 180.0 / Math.PI)
            const sizeMm = entry.size !== undefined && entry.size !== null ? entry.size : 0.0
            size.push(sizeMm * wscale)
        }
        return { "thetaDeg": theta, "sizeGL": size }
    }
}

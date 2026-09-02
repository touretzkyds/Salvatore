import QtQuick 6.5
import QtQuick3D 6.5

Node {
    id: layer
    property var model
    property var constants

    readonly property real wscale: constants && constants["WSCALE"] !== undefined ? constants["WSCALE"] : 1.0

    function _items() {
        if (!model) {
            return []
        }
        const count = model.count !== undefined ? model.count : 0
        const items = []
        for (let i = 0; i < count; ++i) {
            const entry = model.get ? model.get(i) : null
            if (!entry || entry.type !== "sports_ball") {
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
            return { "zFinalGL": [] }
        }
        const values = []
        for (let i = 0; i < data.length; ++i) {
            const entry = data[i]
            const radius = entry.radius !== undefined && entry.radius !== null ? entry.radius : 0.0
            const z = (entry.z !== undefined && entry.z !== null ? entry.z : 0.0) + radius
            values.push(z * wscale)
        }
        return { "zFinalGL": values }
    }
}

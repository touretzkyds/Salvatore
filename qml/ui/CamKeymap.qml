import QtQuick 6.5

Item {
    id: keymap

    signal toggleCrosshair()
    signal toggleAnnotated()
    signal snapshotRaw()
    signal snapshotAnnotated()
    signal helpRequested()
    signal exitRequested()

    Keys.onPressed: (event) => {
        if (event.key === Qt.Key_H) {
            helpRequested()
            event.accepted = true
            return
        }

        switch (event.key) {
        case Qt.Key_C:
            toggleCrosshair()
            event.accepted = true
            break
        case Qt.Key_M:
            toggleAnnotated()
            event.accepted = true
            break
        case Qt.Key_S:
            if (event.modifiers & Qt.ShiftModifier) {
                snapshotAnnotated()
            } else {
                snapshotRaw()
            }
            event.accepted = true
            break
        case Qt.Key_Escape:
            exitRequested()
            event.accepted = true
            break
        default:
            break
        }
    }
}

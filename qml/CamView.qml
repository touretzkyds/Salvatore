import QtQuick 6.5
import QtQuick.Controls 6.5
import "ui" as UI

Item {
    id: root
    objectName: "camView"
    width: 800
    height: 600

    property bool showAnnotated: true
    property bool showCrosshair: false
    property bool showHelp: false
    property string lastSnapshotInfo: ""
    readonly property real overlayScale: AIVISION_RESOLUTION_SCALE > 0 ? AIVISION_RESOLUTION_SCALE : 1

    function toggleCrosshair() {
        showCrosshair = !showCrosshair
        if (showCrosshair) {
            overlay.requestPaint()
        }
        if (viewerApp && typeof viewerApp.setCrosshairEnabled === "function") {
            viewerApp.setCrosshairEnabled(showCrosshair)
        }
    }

    function toggleHelp() {
        showHelp = !showHelp
    }

    function captureSnapshot(annotated) {
        if (typeof viewerApp === "undefined" || viewerApp === null) {
            return
        }
        const path = annotated ? viewerApp.captureAnnotatedSnapshot()
                               : viewerApp.captureRawSnapshot()
        if (path && path.length > 0) {
            lastSnapshotInfo = `Snapshot saved: ${path}`
            console.log(lastSnapshotInfo)
        } else {
            lastSnapshotInfo = "Snapshot failed: no frame available"
            console.log("[CamView]", lastSnapshotInfo)
        }
        snapshotToast.restart()
    }

    UI.CamKeymap {
        id: keymap
        anchors.fill: parent
        focus: true
        onToggleCrosshair: root.toggleCrosshair()
        onToggleAnnotated: root.showAnnotated = !root.showAnnotated
        onSnapshotRaw: root.captureSnapshot(false)
        onSnapshotAnnotated: root.captureSnapshot(true)
        onHelpRequested: root.toggleHelp()
        onExitRequested: {
            if (viewerApp && typeof viewerApp.requestQuit === "function") {
                viewerApp.requestQuit()
            } else {
                Qt.quit()
            }
        }
    }

    Component.onCompleted: {
        if (viewerApp && typeof viewerApp.setCrosshairEnabled === "function") {
            viewerApp.setCrosshairEnabled(showCrosshair)
        }
    }

    Timer {
        id: snapshotToast
        interval: 2500
        repeat: false
        onTriggered: lastSnapshotInfo = ""
    }

    Image {
        id: liveFrame
        anchors.fill: parent
        fillMode: Image.PreserveAspectFit
        cache: false
        source: "image://camera/live?v=" + cameraFrameId
    }

    Image {
        id: annotatedFrame
        anchors.fill: liveFrame
        fillMode: Image.PreserveAspectFit
        cache: false
        visible: root.showAnnotated
        opacity: 0.65
        source: "image://camera/annotated?v=" + cameraFrameId
    }

    Canvas {
        id: overlay
        anchors.fill: liveFrame
        visible: root.showCrosshair
        onPaint: {
            const ctx = getContext("2d")
            ctx.reset()
            const scale = root.overlayScale
            ctx.save()
            ctx.scale(scale, scale)
            const logicalWidth = width / scale
            const logicalHeight = height / scale
            ctx.strokeStyle = "#ffd95a"
            ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(logicalWidth / 2, 0)
            ctx.lineTo(logicalWidth / 2, logicalHeight)
            ctx.moveTo(0, logicalHeight / 2)
            ctx.lineTo(logicalWidth, logicalHeight / 2)
            ctx.stroke()
            ctx.restore()
        }
        onVisibleChanged: {
            if (visible) {
                requestPaint()
            }
        }
    }

    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton
        onPressed: keymap.forceActiveFocus()
    }

    Rectangle {
        id: helpOverlay
        anchors.fill: parent
        visible: root.showHelp
        color: Qt.rgba(0, 0, 0, 0.7)
        z: 100

        Text {
            anchors.centerIn: parent
            width: parent.width * 0.6
            color: "#f5f5f5"
            text: cameraHelpText || ""
            wrapMode: Text.WordWrap
            horizontalAlignment: Text.AlignLeft
        }

        MouseArea {
            anchors.fill: parent
            onClicked: root.showHelp = false
        }
    }

    Rectangle {
        id: snapshotBanner
        visible: root.lastSnapshotInfo.length > 0
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 12
        radius: 6
        color: Qt.rgba(0, 0, 0, 0.6)
        z: 90
        border.width: 0
        width: snapshotLabel.implicitWidth + 24
        height: snapshotLabel.implicitHeight + 12

        Text {
            id: snapshotLabel
            anchors.centerIn: parent
            color: "#f5f5f5"
            text: root.lastSnapshotInfo
            horizontalAlignment: Text.AlignHCenter
        }
    }
}

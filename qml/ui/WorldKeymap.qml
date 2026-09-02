import QtQuick 6.5

Item {
    id: keymap

    signal moveForward()
    signal moveBackward()
    signal moveLeft()
    signal moveRight()
    signal yawLeft()
    signal yawRight()
    signal pitchUp()
    signal pitchDown()
    signal zoomIn()
    signal zoomOut()
    signal resetCamera()
    signal toggleAxes()
    signal toggleParams()
    signal toggleRedisplay()

    focus: true

    Keys.onPressed: (event) => {
        switch (event.key) {
        case Qt.Key_W:
            moveForward()
            event.accepted = true
            break
        case Qt.Key_S:
            moveBackward()
            event.accepted = true
            break
        case Qt.Key_A:
            moveLeft()
            event.accepted = true
            break
        case Qt.Key_D:
            moveRight()
            event.accepted = true
            break
        case Qt.Key_J:
            yawLeft()
            event.accepted = true
            break
        case Qt.Key_L:
            yawRight()
            event.accepted = true
            break
        case Qt.Key_I:
            pitchUp()
            event.accepted = true
            break
        case Qt.Key_K:
            pitchDown()
            event.accepted = true
            break
        case Qt.Key_Less:
        case Qt.Key_Comma:
            zoomIn()
            event.accepted = true
            break
        case Qt.Key_Greater:
        case Qt.Key_Period:
            zoomOut()
            event.accepted = true
            break
        case Qt.Key_Z:
            resetCamera()
            event.accepted = true
            break
        case Qt.Key_X:
            toggleAxes()
            event.accepted = true
            break
        case Qt.Key_V:
            toggleParams()
            event.accepted = true
            break
        case Qt.Key_NumberSign:
            toggleRedisplay()
            event.accepted = true
            break
        }
    }
}

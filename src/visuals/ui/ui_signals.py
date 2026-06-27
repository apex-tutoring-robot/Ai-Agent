from PyQt5.QtCore import QObject, pyqtSignal


class UISignals(QObject):
    listening = pyqtSignal()
    thinking = pyqtSignal()
    teaching_text = pyqtSignal(str)
    clear_canvas = pyqtSignal()
    mouth_level = pyqtSignal(float)
    start_talking = pyqtSignal()
    stop_talking = pyqtSignal()
    draw_actions = pyqtSignal(object)
    show_face_fullscreen = pyqtSignal()
    show_teaching_layout = pyqtSignal()
    scanning = pyqtSignal()

DARK_THEME_QSS = """
QMainWindow {
    background-color: #0D1117;
    color: #C9D1D9;
    font-family: 'Segoe UI', 'SF Pro Text', sans-serif;
    font-size: 13px;
}

QWidget {
    background-color: #0D1117;
    color: #C9D1D9;
    font-family: 'Segoe UI', sans-serif;
}

/* Group Boxes and Cards */
QGroupBox {
    background-color: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    margin-top: 18px;
    padding: 14px;
    font-weight: 600;
    color: #58A6FF;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    left: 12px;
}

/* Tab Widget and Bars */
QTabWidget::pane {
    border: 1px solid #30363D;
    border-radius: 8px;
    background-color: #161B22;
    padding: 12px;
    top: -1px;
}

QTabBar::tab {
    background-color: #0D1117;
    color: #8B949E;
    border: 1px solid #30363D;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 14px;
    font-weight: 600;
    font-size: 12px;
    margin-right: 4px;
}

QTabBar::tab:selected {
    background-color: #161B22;
    color: #58A6FF;
    border-bottom: 2px solid #58A6FF;
}

QTabBar::tab:hover:!selected {
    background-color: #21262D;
    color: #C9D1D9;
}

/* Buttons */
QPushButton {
    background-color: #21262D;
    color: #C9D1D9;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 13px;
}

QPushButton:hover {
    background-color: #30363D;
    border-color: #8B949E;
    color: #FFFFFF;
}

QPushButton:pressed {
    background-color: #1F242C;
}

QPushButton:disabled {
    background-color: #161B22;
    border-color: #21262D;
    color: #484F58;
}

QPushButton#btnPrimary {
    background-color: #238636;
    border: 1px solid #2EA043;
    color: #FFFFFF;
    font-size: 14px;
    font-weight: 700;
    padding: 10px 20px;
}

QPushButton#btnPrimary:hover {
    background-color: #2EA043;
    border-color: #3FB950;
}

QPushButton#btnDownload {
    background-color: #1F6FEB;
    border: 1px solid #388BFD;
    color: #FFFFFF;
    font-weight: 600;
    padding: 8px 14px;
}

QPushButton#btnDownload:hover {
    background-color: #388BFD;
}

QPushButton#btnCancel {
    background-color: #DA3633;
    border: 1px solid #F85149;
    color: #FFFFFF;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 16px;
}

QPushButton#btnCancel:hover {
    background-color: #F85149;
}

/* Combo Boxes */
QComboBox {
    background-color: #0D1117;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 6px 12px;
    color: #C9D1D9;
    font-weight: 500;
}

QComboBox:hover {
    border-color: #58A6FF;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left-width: 0px;
}

QComboBox QAbstractItemView {
    background-color: #161B22;
    border: 1px solid #30363D;
    selection-background-color: #1F6FEB;
    selection-color: #FFFFFF;
    color: #C9D1D9;
    outline: none;
    padding: 4px;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 6px;
    background: #21262D;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #1F6FEB;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #58A6FF;
    border: 2px solid #0D1117;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #79C0FF;
}

/* Checkboxes */
QCheckBox {
    color: #C9D1D9;
    font-weight: 500;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #30363D;
    background-color: #0D1117;
}

QCheckBox::indicator:checked {
    background-color: #1F6FEB;
    border-color: #58A6FF;
}

/* Progress Bar */
QProgressBar {
    background-color: #161B22;
    border: 1px solid #30363D;
    border-radius: 6px;
    text-align: center;
    color: #FFFFFF;
    font-weight: 600;
    height: 20px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1F6FEB, stop:1 #2EA043);
    border-radius: 5px;
}

/* Log Terminal */
QPlainTextEdit#logConsole {
    background-color: #090D12;
    border: 1px solid #30363D;
    border-radius: 6px;
    font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 11px;
    color: #7EE787;
    padding: 8px;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #0D1117;
    width: 8px;
    margin: 0px 0px 0px 0px;
}

QScrollBar::handle:vertical {
    background: #30363D;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #8B949E;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""

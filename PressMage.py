"""
PressMage — магия нажатий в двух слогах. v4.4 —  (ядро + GUI)

"""

from __future__ import annotations
import sys
import time
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple
from collections import deque

# =============================
#       ЯДРО ОЧЕРЕДИ НАЖАТИЙ
# =============================

@dataclass
class PressEvent:
    direction: str                 # 'up','down','left','right'
    appeared_at: float             # время первого появления (сек)


class ArrowQueueEngine:
    """Очередь для нажатий с двумя ограничениями по времени.

    Правила:
      • События добавляются в порядке появления (enqueue_event),
        даже если одновременно видны несколько стрелок.
      • Нажимать можно, только если:
          - прошло минимум `min_press_interval` с предыдущего нажатия;
          - событие пролежало в очереди не меньше `min_age_before_press`.

    Параметры:
      - on_press: callback(direction:str, when:float) — вызывается при нажатии.
      - dedup_window_sec: игнорировать повторные события одного направления,
        пришедшие слишком быстро (защита от дребезга детектора).
    """

    def __init__(
        self,
        *,
        min_press_interval: float = 0.5,
        min_age_before_press: float = 0.0,
        dedup_window_sec: float = 0.10,
        on_press: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        if min_press_interval < 0.0:
            raise ValueError("min_press_interval must be >= 0")
        if min_age_before_press < 0.0:
            raise ValueError("min_age_before_press must be >= 0")
        if dedup_window_sec < 0.0:
            raise ValueError("dedup_window_sec must be >= 0")

        self.min_press_interval = float(min_press_interval)
        self.min_age_before_press = float(min_age_before_press)
        self.dedup_window_sec = float(dedup_window_sec)
        self.on_press = on_press

        self._q: Deque[PressEvent] = deque()
        self._last_press_time: Optional[float] = None
        self._last_enqueued_by_dir: Dict[str, float] = {}
        self.pressed_log: List[Tuple[str, float]] = []  # для тестов/отладки

    # --- API ---
    def enqueue_event(self, direction: str, when: float) -> bool:
        """Добавить событие в очередь. Возвращает True, если событие добавлено."""
        direction = direction.lower().strip()
        if direction not in {"up", "down", "left", "right"}:
            return False
        last = self._last_enqueued_by_dir.get(direction)
        if last is not None and (when - last) < self.dedup_window_sec:
            return False
        self._q.append(PressEvent(direction, when))
        self._last_enqueued_by_dir[direction] = when
        return True

    def can_press_now(self, now: float) -> bool:
        if not self._q:
            return False
        if self._last_press_time is not None:
            if (now - self._last_press_time) < self.min_press_interval:
                return False
        head = self._q[0]
        if (now - head.appeared_at) < self.min_age_before_press:
            return False
        return True

    def tick(self, now: float) -> List[Tuple[str, float]]:
        pressed: List[Tuple[str, float]] = []
        # Жмём не более одной стрелки за тик, чтобы соблюдать интервалы строго
        if self.can_press_now(now):
            ev = self._q.popleft()
            self._last_press_time = now
            self.pressed_log.append((ev.direction, now))
            if self.on_press:
                self.on_press(ev.direction, now)
            pressed.append((ev.direction, now))
        return pressed

    def pending_count(self) -> int:
        return len(self._q)


# ======================================================
#                    GUI (PyQt5)
# ======================================================

def run_gui_app() -> None:
    """Импорт и запуск GUI."""
    import os
    import ctypes
    from ctypes import wintypes
    import winsound
    import psutil
    import numpy as np
    import cv2
    from PIL import ImageGrab
    import pyautogui
    import win32gui
    import win32process
    import win32con

    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit,
        QHBoxLayout, QComboBox, QMessageBox, QGroupBox, QFormLayout, QDoubleSpinBox,
        QCheckBox, QTabWidget, QProgressBar, QFileDialog, QTableWidget, QTableWidgetItem,
        QStatusBar, QSpinBox, QGridLayout, QFrame
    )
    from PyQt5.QtCore import Qt, QTimer, QSettings, pyqtSignal, QObject, QThread, QSize, QRect, QAbstractNativeEventFilter
    from PyQt5.QtGui import QColor, QPixmap, QImage, QPainter, QPen

    # ---------- WinAPI утилиты ----------
    def get_window_by_process(process_name: str) -> Optional[int]:
        def enum_windows(hwnd, acc):
            if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindow(hwnd):
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    proc = psutil.Process(pid)
                    if proc.name().lower() == process_name.lower():
                        acc.append(hwnd)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        windows: List[int] = []
        win32gui.EnumWindows(enum_windows, windows)
        return windows[0] if windows else None

    def get_window_geometry(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        try:
            l, t, r, b = win32gui.GetWindowRect(hwnd)
            return (l, t, r, b)
        except Exception:
            return None

    def bring_window_to_front(hwnd: int) -> bool:
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            return True
        except Exception:
            return False

    # ---------- Раскраска логов ----------
    COLOR_MAP = {
        "success": "#4CAF50",
        "error": "#f44336",
        "warning": "#FF9800",
        "info": "#2196F3",
    }

    def color_span(text: str, level: str = "info") -> str:
        color = COLOR_MAP.get(level, "#222222")
        return f'<span style="color:{color}">{text}</span>'

    # ---------- Overlay для выбора ROI (строго в окне игры) ----------
    class RoiOverlay(QWidget):
        selected = pyqtSignal(QRect)
        canceled = pyqtSignal()
        def __init__(self, game_rect: QRect, grid_size: int = 16, snap: bool = True, show_grid: bool = True):
            super().__init__()
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
            self.setAttribute(Qt.WA_NoSystemBackground, True)
            self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self._origin = None
            self._rect = QRect()
            self._bounds = game_rect  # ограничение — прямоугольник окна игры в координатах экрана
            self._grid = max(2, int(grid_size))
            self._snap = bool(snap)
            self._show_grid = bool(show_grid)
            self.setGeometry(self._bounds)
            self.show()

        def paintEvent(self, e):
            p = QPainter(self)
            # затемнение
            p.fillRect(self.rect(), QColor(0,0,0,60))
            # сетка
            if self._show_grid:
                p.setPen(QPen(QColor(255,255,255,60), 1))
                w, h = self.width(), self.height()
                step = self._grid
                for x in range(0, w, step):
                    p.drawLine(x, 0, x, h)
                for y in range(0, h, step):
                    p.drawLine(0, y, w, y)
            # прямоугольник выделения
            if not self._rect.isNull():
                p.setPen(QPen(QColor(0,255,0), 2, Qt.SolidLine))
                p.drawRect(self._rect)
            p.end()

        def _clamp(self, x: int, y: int) -> Tuple[int,int]:
            x = min(max(x, 0), self.width()-1)
            y = min(max(y, 0), self.height()-1)
            if self._snap:
                g = self._grid
                # к ближайшей линии сетки
                x = int(round(x / g) * g)
                y = int(round(y / g) * g)
                x = min(max(x, 0), self.width()-1)
                y = min(max(y, 0), self.height()-1)
            return x, y

        def mousePressEvent(self, e):
            if e.button() == Qt.LeftButton:
                x,y = self._clamp(e.pos().x(), e.pos().y())
                self._origin = (x,y)
                self._rect = QRect(x, y, 1, 1)
                self.update()

        def mouseMoveEvent(self, e):
            if self._origin is not None:
                x,y = self._clamp(e.pos().x(), e.pos().y())
                ox, oy = self._origin
                x1, x2 = (x, ox) if x < ox else (ox, x)
                y1, y2 = (y, oy) if y < oy else (oy, y)
                # для гарантированного покрытия ячейки, правый/нижний край слегка расширим при снапе
                if self._snap:
                    g = self._grid
                    x1 = (x1 // g) * g
                    y1 = (y1 // g) * g
                    x2 = ((x2 + g - 1) // g) * g
                    y2 = ((y2 + g - 1) // g) * g
                    x2 = min(x2, self.width()-1)
                    y2 = min(y2, self.height()-1)
                self._rect = QRect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
                self.update()

        def mouseReleaseEvent(self, e):
            if e.button() == Qt.LeftButton and not self._rect.isNull():
                self.selected.emit(self._rect)
                self.close()

        def keyPressEvent(self, e):
            if e.key() == Qt.Key_Escape:
                self.canceled.emit(); self.close()

    class MainWindow(QMainWindow):(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle('Помощник для стрелок v4.4 — Последовательный режим')
            self.setGeometry(120, 120, 940, 740)

            self.settings = QSettings('ArrowHelper', 'ArrowBotV44')
            self.templates: Dict[str, np.ndarray] = {}

            self.worker = BotWorker()
            self.thread = QThread(); self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.log_sig.connect(self._on_log)
            self.worker.stats_sig.connect(self._on_stats)
            self.worker.progress_sig.connect(self._on_progress)
            self.thread.start()

            self._build_ui()
            self._load_templates()
            self._load_settings()

            self.stats = {"sessions": 0, "start": None}
            self.session_timer = QTimer(self); self.session_timer.timeout.connect(self._update_time)

            self.hotkey_filter = HotkeyFilter(self._start_hotkey, self._pause_hotkey, self._stop_hotkey)
            QApplication.instance().installNativeEventFilter(self.hotkey_filter)

        # --- UI ---
        def _build_ui(self):
            tab = QTabWidget(); self.setCentralWidget(tab)
            tab.addTab(self._tab_control(), "Управление")
            tab.addTab(self._tab_settings(), "Настройки")
            tab.addTab(self._tab_stats(), "Статистика")
            tab.addTab(self._tab_templates(), "Шаблоны")
            self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)
            self.status_bar.showMessage('Готов к работе — v4.4')

        def _tab_control(self) -> QWidget:
            w = QWidget(); v = QVBoxLayout(w)
            title = QLabel('Помощник для игры со стрелками (Последовательный режим)')
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-size:16px;font-weight:bold;margin:8px;color:#2196F3;")
            v.addWidget(title)

            gb = QGroupBox("Выбор игры"); gl = QGridLayout(gb)
            gl.addWidget(QLabel('Процесс:'), 0, 0)
            self.process_combo = QComboBox(); self._refresh_processes(); gl.addWidget(self.process_combo, 0, 1)
            btn_refresh = QPushButton('Обновить'); btn_refresh.clicked.connect(self._refresh_processes); gl.addWidget(btn_refresh, 0, 2)
            btn_focus = QPushButton('Активировать окно'); btn_focus.clicked.connect(self._focus_window); gl.addWidget(btn_focus, 0, 3)
            btn_roi = QPushButton('Выбрать ROI мышью'); btn_roi.clicked.connect(self._select_roi_overlay); gl.addWidget(btn_roi, 0, 4)
            v.addWidget(gb)

            hl = QHBoxLayout()
            self.btn_start = QPushButton('Запустить'); self.btn_start.clicked.connect(self._start)
            self.btn_pause = QPushButton('Пауза'); self.btn_pause.clicked.connect(self._pause); self.btn_pause.setEnabled(False)
            self.btn_stop = QPushButton('Остановить'); self.btn_stop.clicked.connect(self._stop); self.btn_stop.setEnabled(False)
            for btn, c, h in ((self.btn_start, '#4CAF50', '#45a049'), (self.btn_pause, '#FF9800', '#e68a00'), (self.btn_stop, '#f44336', '#da190b')):
                btn.setStyleSheet(f"QPushButton {{background:{c};color:white;font-weight:bold;padding:10px;border-radius:6px;}}"
                                  f"QPushButton:hover {{background:{h};}}")
            for b in (self.btn_start, self.btn_pause, self.btn_stop): hl.addWidget(b)
            v.addLayout(hl)

            gb2 = QGroupBox("Статус"); vb2 = QVBoxLayout(gb2)
            self.status_label = QLabel('Статус: Ожидание запуска'); self.status_label.setStyleSheet("font-weight:bold;font-size:14px;")
            vb2.addWidget(self.status_label)
            self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False); vb2.addWidget(self.progress_bar)

            self.preview_label = QLabel(); self.preview_label.setFixedSize(560, 320)
            self.preview_label.setFrameShape(QFrame.StyledPanel); self.preview_label.setAlignment(Qt.AlignCenter)
            btn_prev = QPushButton('Предпросмотр распознавания'); btn_prev.clicked.connect(self._preview)
            hb = QHBoxLayout(); hb.addWidget(self.preview_label); hb.addWidget(btn_prev)
            vb2.addLayout(hb)
            v.addWidget(gb2)

            gb3 = QGroupBox("Логи"); vb3 = QVBoxLayout(gb3)
            self.log_text = QTextEdit(); self.log_text.setReadOnly(True); self.log_text.setMaximumHeight(220); vb3.addWidget(self.log_text)
            btn_clear = QPushButton('Очистить логи'); btn_clear.clicked.connect(lambda: self.log_text.clear()); vb3.addWidget(btn_clear)
            v.addWidget(gb3)

            info = QLabel('• Очередь: жмём в порядке появления.
• Две задержки действуют вместе: между нажатиями и после появления.
• ROI выбирается ТОЛЬКО внутри окна игры.')
            info.setStyleSheet("background:#e8f4f8;padding:10px;border-radius:8px;border:2px solid #b3e0f2;font-weight:bold;")
            v.addWidget(info)
            return w

        def _tab_settings(self) -> QWidget:
            w = QWidget(); v = QVBoxLayout(w)
            gb = QGroupBox("Основные настройки"); form = QFormLayout(gb)
            self.loop_delay = QDoubleSpinBox(); self.loop_delay.setRange(0.005, 2.0); self.loop_delay.setValue(0.10)
            self.key_delay = QDoubleSpinBox(); self.key_delay.setRange(0.0, 1.0); self.key_delay.setValue(0.03)
            self.threshold = QDoubleSpinBox(); self.threshold.setRange(0.5, 1.0); self.threshold.setValue(0.80)
            self.disappear = QDoubleSpinBox(); self.disappear.setRange(0.05, 5.0); self.disappear.setValue(0.50)
            self.min_press_interval = QDoubleSpinBox(); self.min_press_interval.setRange(0.0, 3.0); self.min_press_interval.setValue(0.50)
            self.min_age_before_press = QDoubleSpinBox(); self.min_age_before_press.setRange(0.0, 3.0); self.min_age_before_press.setValue(0.50)
            self.dedup_window = QDoubleSpinBox(); self.dedup_window.setRange(0.0, 1.0); self.dedup_window.setValue(0.10)
            form.addRow("Задержка цикла (сек):", self.loop_delay)
            form.addRow("Задержка между key.press (сек):", self.key_delay)
            form.addRow("Порог совпадения (0.5–1.0):", self.threshold)
            form.addRow("Таймаут исчезновения (сек):", self.disappear)
            form.addRow("Мин. интервал между нажатиями (сек):", self.min_press_interval)
            form.addRow("Ожидание после появления (сек):", self.min_age_before_press)
            form.addRow("Окно дедупликации (сек):", self.dedup_window)

            # ROI
            roi = QGroupBox("Область поиска (внутренние отступы)"); grid = QGridLayout(roi)
            self.roi_left = QSpinBox(); self.roi_top = QSpinBox(); self.roi_right = QSpinBox(); self.roi_bottom = QSpinBox()
            for sb in (self.roi_left, self.roi_top, self.roi_right, self.roi_bottom): sb.setRange(0, 3000)
            grid.addWidget(QLabel("Слева:"), 0, 0); grid.addWidget(self.roi_left, 0, 1)
            grid.addWidget(QLabel("Сверху:"), 0, 2); grid.addWidget(self.roi_top, 0, 3)
            grid.addWidget(QLabel("Справа:"), 1, 0); grid.addWidget(self.roi_right, 1, 1)
            grid.addWidget(QLabel("Снизу:"), 1, 2); grid.addWidget(self.roi_bottom, 1, 3)

            # ROI-сетка/магнит
            grid2 = QGroupBox("ROI‑магнит (сетка)"); grid2l = QGridLayout(grid2)
            self.roi_grid_enable = QCheckBox("Привязка ROI к сетке"); self.roi_grid_enable.setChecked(True)
            self.roi_grid_show = QCheckBox("Показывать сетку при выборе ROI"); self.roi_grid_show.setChecked(True)
            self.roi_grid_size = QSpinBox(); self.roi_grid_size.setRange(2, 256); self.roi_grid_size.setValue(16)
            grid2l.addWidget(self.roi_grid_enable, 0, 0, 1, 2)
            grid2l.addWidget(self.roi_grid_show, 1, 0, 1, 2)
            grid2l.addWidget(QLabel("Шаг сетки (px):"), 2, 0); grid2l.addWidget(self.roi_grid_size, 2, 1)

            self.use_canny = QCheckBox("Поиск по контурам (Canny)")
            self.simulation = QCheckBox("Симуляция (не нажимать клавиши)"); self.simulation.setChecked(True)
            self.preview_draw_boxes = QCheckBox("Показывать рамки детекций в предпросмотре"); self.preview_draw_boxes.setChecked(True)

            form.addRow(roi)
            form.addRow(grid2)
            form.addRow("Доп. режимы:", self.use_canny)
            form.addRow("", self.simulation)
            form.addRow("Предпросмотр:", self.preview_draw_boxes)

            v.addWidget(gb)

            # Назначение клавиш
            key_gb = QGroupBox("Назначение клавиш"); kform = QFormLayout(key_gb)
            self.key_up = QComboBox(); self.key_down = QComboBox(); self.key_left = QComboBox(); self.key_right = QComboBox()
            keys = ['up','down','left','right','w','a','s','d','i','j','k','l','1','2','3','4','5','6','7','8','9','0','space','enter','tab','shift','ctrl','alt']
            for cb in (self.key_up, self.key_down, self.key_left, self.key_right): cb.addItems(keys)
            self.key_up.setCurrentText('up'); self.key_down.setCurrentText('down'); self.key_left.setCurrentText('left'); self.key_right.setCurrentText('right')
            kform.addRow("Вверх:", self.key_up); kform.addRow("Вниз:", self.key_down); kform.addRow("Влево:", self.key_left); kform.addRow("Вправо:", self.key_right)
            v.addWidget(key_gb)

            # Кнопки
            hl = QHBoxLayout(); btn_save = QPushButton("Сохранить настройки"); btn_reset = QPushButton("Сбросить настройки")
            btn_save.clicked.connect(self._save_settings); btn_reset.clicked.connect(self._reset_settings)
            hl.addWidget(btn_save); hl.addWidget(btn_reset); v.addLayout(hl)
            v.addStretch(1)
            return w

        def _tab_stats(self) -> QWidget:
            w = QWidget(); v = QVBoxLayout(w)
            gb = QGroupBox("Статистика сессии"); form = QFormLayout(gb)
            self.arrows_found = QLabel("0"); self.arrows_found.setStyleSheet("font-weight:bold;color:#2196F3;")
            self.keys_pressed = QLabel("0"); self.keys_pressed.setStyleSheet("font-weight:bold;color:#4CAF50;")
            self.session_time = QLabel("00:00:00"); self.sessions_count = QLabel("0")
            form.addRow("Найдено стрелок:", self.arrows_found)
            form.addRow("Нажато клавиш:", self.keys_pressed)
            form.addRow("Время работы:", self.session_time)
            form.addRow("Сессий:", self.sessions_count)
            v.addWidget(gb); v.addStretch(1)
            return w

        def _tab_templates(self) -> QWidget:
            w = QWidget(); v = QVBoxLayout(w)
            gb = QGroupBox("Управление шаблонами"); vb = QVBoxLayout(gb)
            self.table = QTableWidget(); self.table.setColumnCount(3)
            self.table.setHorizontalHeaderLabels(["Тип", "Файл", "Статус"]) ; self.table.horizontalHeader().setStretchLastSection(True)
            vb.addWidget(self.table)

            tools = QGroupBox("Запись шаблонов из экрана"); tform = QHBoxLayout(tools)
            self.template_dir_combo = QComboBox(); self.template_dir_combo.addItems(['up','down','left','right'])
            btn_record = QPushButton("Записать из окна игры")
            btn_record.clicked.connect(self._record_template_from_screen)
            tform.addWidget(QLabel("Направление:")); tform.addWidget(self.template_dir_combo); tform.addWidget(btn_record)
            vb.addWidget(tools)

            hl = QHBoxLayout(); btn_add = QPushButton("Добавить файл"); btn_del = QPushButton("Удалить шаблон"); btn_reload = QPushButton("Обновить")
            btn_add.clicked.connect(self._add_template); btn_del.clicked.connect(self._del_template); btn_reload.clicked.connect(self._load_templates)
            hl.addWidget(btn_add); hl.addWidget(btn_del); hl.addWidget(btn_reload); vb.addLayout(hl)
            v.addWidget(gb)
            info = QLabel("PNG 32×32 рекомендовано. Имена: arrow_up.png, arrow_down.png, arrow_left.png, arrow_right.png")
            info.setStyleSheet("background:#d4edda;padding:10px;border-radius:8px;border:1px solid #c3e6cb;font-size:11px;")
            v.addWidget(info); v.addStretch(1)
            return w

        # --- кнопки управления ---
        def _start(self):
            if not self.templates:
                QMessageBox.warning(self, "Нет шаблонов", "Загрузите шаблоны стрелок")
                return
            params = {
                "process_name": self._current_process_name(),
                "loop_delay": self.loop_delay.value(),
                "threshold": self.threshold.value(),
                "disappear_timeout": self.disappear.value(),
                "roi_left": self.roi_left.value(), "roi_top": self.roi_top.value(),
                "roi_right": self.roi_right.value(), "roi_bottom": self.roi_bottom.value(),
                "use_canny": self.use_canny.isChecked(),
                "simulation": self.simulation.isChecked(),
                "min_press_interval": self.min_press_interval.value(),
                "min_age_before_press": self.min_age_before_press.value(),
                "dedup_window": self.dedup_window.value(),
                "key_up": self.key_up.currentText(),
                "key_down": self.key_down.currentText(),
                "key_left": self.key_left.currentText(),
                "key_right": self.key_right.currentText(),
            }
            self.worker.set_params(params, self.templates)
            self.worker.start()

            self.btn_start.setEnabled(False); self.btn_pause.setEnabled(True); self.btn_stop.setEnabled(True)
            self.status_label.setText("Статус: Работает — Последовательный режим"); self.status_label.setStyleSheet("color:green;font-weight:bold;font-size:14px;")
            self.status_bar.showMessage('Бот запущен — v4.4')
            self.stats["sessions"] += 1; self.stats["start"] = time.time(); self.session_timer.start(1000)
            self._on_log("Бот запущен", "success")

        def _pause(self):
            self.worker.toggle_pause()

        def _stop(self):
            self.worker.stop()
            self.btn_start.setEnabled(True); self.btn_pause.setEnabled(False); self.btn_stop.setEnabled(False)
            self.status_label.setText("Статус: Остановлен"); self.status_label.setStyleSheet("color:red;font-weight:bold;font-size:14px;")
            self.status_bar.showMessage('Бот остановлен')
            self.progress_bar.setVisible(False); self.session_timer.stop()
            self._on_log("Бот остановлен", "info")

        # --- горячие клавиши callbacks ---
        def _start_hotkey(self):
            if self.btn_start.isEnabled(): self._start()
        def _pause_hotkey(self):
            if self.btn_pause.isEnabled(): self._pause()
        def _stop_hotkey(self):
            if self.btn_stop.isEnabled(): self._stop()

        # --- сервис ---
        def _on_log(self, message: str, level: str = "info"):
            ts = time.strftime("%H:%M:%S"); self.log_text.append(color_span(f"[{ts}] {message}", level))
            self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

        def _on_stats(self, arrows_found: int, keys_pressed: int, sessions: int):
            self.arrows_found.setText(str(arrows_found)); self.keys_pressed.setText(str(keys_pressed)); self.sessions_count.setText(str(self.stats["sessions"]))

        def _on_progress(self):
            self.progress_bar.setVisible(True); self.progress_bar.setValue(100)
            QTimer.singleShot(350, lambda: self.progress_bar.setVisible(False))

        def _update_time(self):
            if not self.stats["start"]: return
            elapsed = int(time.time() - self.stats["start"]); h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
            self.session_time.setText(f"{h:02d}:{m:02d}:{s:02d}")

        def _refresh_processes(self):
            cur = self.process_combo.currentText(); self.process_combo.clear()
            items: List[str] = []
            for proc in psutil.process_iter(['pid','name']):
                try:
                    name = proc.info['name'] or ''
                    if name.lower().endswith('.exe'):
                        items.append(f"{name} (PID: {proc.info['pid']})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            items.sort(); self.process_combo.addItems(items)
            saved = self.settings.value('last_process', '', type=str)
            if saved:
                idx = self.process_combo.findText(saved)
                if idx >= 0: self.process_combo.setCurrentIndex(idx)
            elif cur:
                idx = self.process_combo.findText(cur)
                if idx >= 0: self.process_combo.setCurrentIndex(idx)

        def _current_process_name(self) -> str:
            text = self.process_combo.currentText()
            return text.split(' (PID:')[0] if text else ''

        def _focus_window(self):
            name = self._current_process_name()
            hwnd = get_window_by_process(name)
            if not hwnd:
                QMessageBox.warning(self, 'Окно не найдено', 'Не удалось найти окно процесса')
                return
            if bring_window_to_front(hwnd):
                self._on_log("Окно активировано", "success")
            else:
                self._on_log("Не удалось активировать окно", "warning")

        def _preview(self):
            name = self._current_process_name(); hwnd = get_window_by_process(name)
            if not hwnd:
                QMessageBox.warning(self, 'Окно не найдено', 'Не удалось найти окно процесса'); return
            geom = get_window_geometry(hwnd)
            if not geom:
                QMessageBox.warning(self, 'Ошибка', 'Не удалось получить геометрию окна'); return
            l, t, r, b = geom
            roi = (l + int(self.roi_left.value()), t + int(self.roi_top.value()), r - int(self.roi_right.value()), b - int(self.roi_bottom.value()))
            roi = (max(roi[0], l), max(roi[1], t), max(min(roi[2], r), l+1), max(min(roi[3], b), t+1))
            try:
                shot = ImageGrab.grab(bbox=roi)
                frame_bgr = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
                draw_bgr = frame_bgr.copy()
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                use_canny = self.use_canny.isChecked()
                if use_canny:
                    gray = cv2.Canny(gray, 60, 120)
                # детекция как в воркере
                matches: List[Tuple[int,int,int,int,str,float]] = []
                thr = float(self.threshold.value())
                for direction, templ in self.templates.items():
                    tpl = cv2.Canny(templ, 60, 120) if use_canny else templ
                    res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= thr)
                    h, w = tpl.shape[:2]
                    for pt in zip(*loc[::-1]):
                        score = float(res[pt[1], pt[0]])
                        matches.append((pt[0], pt[1], w, h, direction, score))
                matches = nms_detections(matches, radius=12)
                if self.preview_draw_boxes.isChecked():
                    color_map = {
                        'up':   (0,255,0),
                        'down': (0,0,255),
                        'left': (255,0,0),
                        'right':(0,255,255)
                    }
                    for (x,y,w,h,d,score) in matches:
                        cv2.rectangle(draw_bgr, (x,y), (x+w,y+h), color_map.get(d,(255,255,255)), 2)
                        cv2.putText(draw_bgr, f"{d}:{score:.2f}", (x, max(0,y-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map.get(d,(255,255,255)), 1, cv2.LINE_AA)
                disp = cv2.cvtColor(draw_bgr, cv2.COLOR_BGR2RGB)
                h, w, _ = disp.shape
                qimg = QImage(disp.data, w, h, 3*w, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.preview_label.setPixmap(pix)
                self._on_log(f"Предпросмотр: найдено {len(matches)} совпадений", "info")
            except Exception as e:
                self._on_log(f"Ошибка предпросмотра: {e}", "error")

        def _select_roi_overlay(self):
            name = self._current_process_name(); hwnd = get_window_by_process(name)
            if not hwnd:
                QMessageBox.warning(self, 'Окно не найдено', 'Выберите процесс игры'); return
            geom = get_window_geometry(hwnd)
            if not geom:
                QMessageBox.warning(self, 'Ошибка', 'Не удалось получить геометрию окна'); return
            l, t, r, b = geom
            bounds = QRect(l, t, r-l, b-t)
            overlay = RoiOverlay(bounds, grid_size=self.roi_grid_size.value(), snap=self.roi_grid_enable.isChecked(), show_grid=self.roi_grid_show.isChecked())
            def on_sel(rect: QRect):
                win_w, win_h = bounds.width(), bounds.height()
                left = max(0, rect.left())
                top = max(0, rect.top())
                right = max(0, win_w - (rect.left() + rect.width()))
                bottom = max(0, win_h - (rect.top() + rect.height()))
                self.roi_left.setValue(left)
                self.roi_top.setValue(top)
                self.roi_right.setValue(right)
                self.roi_bottom.setValue(bottom)
                self._on_log("ROI обновлён из выделения (внутри окна игры, с привязкой)", "success")
            overlay.selected.connect(on_sel)
            overlay.canceled.connect(lambda: self._on_log("Выбор ROI отменён", "warning"))
            overlay.show()

        def _record_template_from_screen(self):
            # Записываем шаблон только в границах окна игры; работает и с привязкой к сетке
            direction = self.template_dir_combo.currentText()
            name = self._current_process_name(); hwnd = get_window_by_process(name)
            if not hwnd:
                QMessageBox.warning(self, 'Окно не найдено', 'Выберите процесс игры'); return
            geom = get_window_geometry(hwnd)
            if not geom:
                QMessageBox.warning(self, 'Ошибка', 'Не удалось получить геометрию окна'); return
            l, t, r, b = geom
            bounds = QRect(l, t, r-l, b-t)
            overlay = RoiOverlay(bounds, grid_size=self.roi_grid_size.value(), snap=self.roi_grid_enable.isChecked(), show_grid=self.roi_grid_show.isChecked())
            def on_sel(rect: QRect):
                try:
                    x1 = l + rect.left(); y1 = t + rect.top(); x2 = x1 + rect.width(); y2 = y1 + rect.height()
                    shot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
                    img = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2GRAY)
                    fn = f"arrow_{direction}.png"
                    cv2.imwrite(fn, img)
                    self._load_templates()
                    self._on_log(f"Шаблон сохранён: {fn}", "success")
                except Exception as e:
                    QMessageBox.critical(self, 'Ошибка', f'Не удалось записать шаблон: {e}')
            overlay.selected.connect(on_sel)
            overlay.canceled.connect(lambda: self._on_log("Запись шаблона отменена", "warning"))
            overlay.show()

        def _add_template(self):
            path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение шаблона", "", "PNG Files (*.png);;All Files (*)")
            if not path: return
            from shutil import copy2
            try:
                dst = os.path.join(os.getcwd(), os.path.basename(path)); copy2(path, dst)
                self._load_templates(); self._on_log(f"Шаблон добавлен: {os.path.basename(path)}", "success")
            except Exception as e:
                QMessageBox.critical(self, 'Ошибка', f'Не удалось добавить: {e}')

        def _del_template(self):
            row = self.table.currentRow();
            if row < 0: return
            filename = self.table.item(row, 1).text()
            try:
                if os.path.exists(filename): os.remove(filename)
                self._load_templates(); self._on_log(f"Шаблон удалён: {filename}", "info")
            except Exception as e:
                QMessageBox.critical(self, 'Ошибка', f'Не удалось удалить: {e}')

        def _load_templates(self):
            import numpy as np
            self.templates.clear(); self.table.setRowCount(0)
            arrow_types = ['up','down','left','right']
            loaded = 0
            for typ in arrow_types:
                fn = f'arrow_{typ}.png'
                if os.path.exists(fn):
                    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        self.templates[typ] = img; status = "Загружен"; bg = QColor(144,238,144); loaded += 1
                    else:
                        status = "Ошибка загрузки"; bg = QColor(255,99,71)
                else:
                    status = "Файл не найден"; bg = QColor(255,165,0)
                r = self.table.rowCount(); self.table.insertRow(r)
                self.table.setItem(r, 0, QTableWidgetItem(typ.capitalize()))
                self.table.setItem(r, 1, QTableWidgetItem(fn))
                it = QTableWidgetItem(status); it.setBackground(bg); self.table.setItem(r, 2, it)
            self._on_log(f"Загружено {loaded} шаблонов из 4", "info")

        def _save_settings(self):
            s = self.settings
            s.setValue('loop_delay', self.loop_delay.value())
            s.setValue('key_delay', self.key_delay.value())
            s.setValue('threshold', self.threshold.value())
            s.setValue('disappear', self.disappear.value())
            s.setValue('roi_left', self.roi_left.value()); s.setValue('roi_top', self.roi_top.value())
            s.setValue('roi_right', self.roi_right.value()); s.setValue('roi_bottom', self.roi_bottom.value())
            s.setValue('roi_grid_enable', self.roi_grid_enable.isChecked())
            s.setValue('roi_grid_show', self.roi_grid_show.isChecked())
            s.setValue('roi_grid_size', self.roi_grid_size.value())
            s.setValue('use_canny', self.use_canny.isChecked()); s.setValue('simulation', self.simulation.isChecked())
            s.setValue('preview_draw_boxes', self.preview_draw_boxes.isChecked())
            s.setValue('min_press_interval', self.min_press_interval.value())
            s.setValue('min_age_before_press', self.min_age_before_press.value())
            s.setValue('dedup_window', self.dedup_window.value())
            s.setValue('key_up', self.key_up.currentText()); s.setValue('key_down', self.key_down.currentText())
            s.setValue('key_left', self.key_left.currentText()); s.setValue('key_right', self.key_right.currentText())
            s.setValue('last_process', self.process_combo.currentText())
            self._on_log("Настройки сохранены", "success")

        def _load_settings(self):
            s = self.settings
            self.loop_delay.setValue(float(s.value('loop_delay', 0.10)))
            self.key_delay.setValue(float(s.value('key_delay', 0.03)))
            self.threshold.setValue(float(s.value('threshold', 0.80)))
            self.disappear.setValue(float(s.value('disappear', 0.50)))
            self.roi_left.setValue(int(s.value('roi_left', 0))); self.roi_top.setValue(int(s.value('roi_top', 0)))
            self.roi_right.setValue(int(s.value('roi_right', 0))); self.roi_bottom.setValue(int(s.value('roi_bottom', 0)))
            self.roi_grid_enable.setChecked(bool(s.value('roi_grid_enable', True, type=bool)))
            self.roi_grid_show.setChecked(bool(s.value('roi_grid_show', True, type=bool)))
            self.roi_grid_size.setValue(int(s.value('roi_grid_size', 16)))
            self.use_canny.setChecked(bool(s.value('use_canny', False, type=bool)))
            self.simulation.setChecked(bool(s.value('simulation', True, type=bool)))
            self.preview_draw_boxes.setChecked(bool(s.value('preview_draw_boxes', True, type=bool)))
            self.min_press_interval.setValue(float(s.value('min_press_interval', 0.50)))
            self.min_age_before_press.setValue(float(s.value('min_age_before_press', 0.50)))
            self.dedup_window.setValue(float(s.value('dedup_window', 0.10)))
            self.key_up.setCurrentText(s.value('key_up', 'up')); self.key_down.setCurrentText(s.value('key_down', 'down'))
            self.key_left.setCurrentText(s.value('key_left', 'left')); self.key_right.setCurrentText(s.value('key_right', 'right'))

        def closeEvent(self, event):
            try:
                self._save_settings()
                self.worker.stop()
                self.hotkey_filter.unregister()
            finally:
                event.accept()

    # ---- точка входа GUI ----
    pyautogui.FAILSAFE = False
    app = QApplication(sys.argv); app.setStyle('Fusion')
    win = MainWindow(); win.show()
    sys.exit(app.exec_())


# =====================================
#           ЮНИТ‑ТЕСТЫ ЯДРА
# =====================================

def _run_tests() -> None:
    import unittest

    class EngineTests(unittest.TestCase):
        def test_fifo_and_min_press_interval(self):
            eng = ArrowQueueEngine(min_press_interval=0.5, min_age_before_press=0.0)
            eng.enqueue_event('up', 0.0)
            eng.enqueue_event('left', 0.10)
            eng.enqueue_event('down', 0.20)
            out = []
            for t in [0.0, 0.25, 0.50, 0.75, 1.00, 1.25]:
                out.extend(eng.tick(t))
            self.assertEqual([d for d,_ in out], ['up','left','down'])
            self.assertAlmostEqual(out[0][1], 0.0, places=2)
            self.assertAlmostEqual(out[1][1], 0.5, places=2)
            self.assertAlmostEqual(out[2][1], 1.0, places=2)

        def test_min_age_before_press(self):
            eng = ArrowQueueEngine(min_press_interval=0.2, min_age_before_press=0.5)
            eng.enqueue_event('up', 0.0)
            eng.enqueue_event('right', 0.10)
            out = []
            for t in [0.0, 0.49, 0.50, 0.69, 0.70, 0.90]:
                out.extend(eng.tick(t))
            self.assertEqual([d for d,_ in out], ['up','right'])
            self.assertAlmostEqual(out[0][1], 0.5, places=2)
            self.assertAlmostEqual(out[1][1], 0.7, places=2)

        def test_dedup_window(self):
            eng = ArrowQueueEngine(min_press_interval=0.1, min_age_before_press=0.0, dedup_window_sec=0.10)
            self.assertTrue(eng.enqueue_event('up', 0.0))
            self.assertFalse(eng.enqueue_event('up', 0.05))
            self.assertTrue(eng.enqueue_event('up', 0.20))
            out = []
            for t in [0.0, 0.10, 0.20, 0.30]:
                out.extend(eng.tick(t))
            self.assertEqual([d for d,_ in out], ['up','up'])

        def test_buildup_with_both_delays(self):
            eng = ArrowQueueEngine(min_press_interval=0.5, min_age_before_press=0.5)
            for i, d in enumerate(['up','left','down','right']):
                eng.enqueue_event(d, i*0.05)
            out = []
            for t in [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]:
                out.extend(eng.tick(t))
            self.assertEqual([d for d,_ in out], ['up','left','down'])
            self.assertAlmostEqual(out[0][1], 0.5, places=2)
            self.assertAlmostEqual(out[1][1], 1.0, places=2)
            self.assertAlmostEqual(out[2][1], 1.5, places=2)

    suite = unittest.TestLoader().loadTestsFromTestCase(EngineTests)
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    if not res.wasSuccessful():
        sys.exit(1)


# =====================================
#                MAIN
# =====================================
if __name__ == '__main__':
    if '--test' in sys.argv:
        _run_tests()
    else:
        # Без проверок наличия PyQt5 — запускаем GUI
        run_gui_app()

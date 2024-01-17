import threading
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont

import cv2
import time
import numpy as np
from functools import partial


class ComboBoxTkApp:
    def __init__(self, root, cands_1, cands_2):
        self.cands_1 = cands_1
        self.cands_2 = cands_2
        self.root = root
        self.root.title("双下拉框示例")
        self.root.geometry("400x160")  # 窗口大小设置为原来的两倍

        # 设置字体
        large_font = tkFont.Font(size=40)

        # 第一个下拉框
        self.combo_box1 = ttk.Combobox(root, values=cands_1, font=large_font)
        self.combo_box1.pack(pady=10)  # 增加垂直间距
        self.combo_box1.bind("<<ComboboxSelected>>", self.on_select1)

        # 第二个下拉框
        self.combo_box2 = ttk.Combobox(root, values=cands_2, font=large_font)
        self.combo_box2.pack(pady=10)  # 增加垂直间距
        self.combo_box2.bind("<<ComboboxSelected>>", self.on_select2)

        # 成员变量，存储下拉框的当前值
        self.value1 = None
        self.value2 = None

    def on_select1(self, event):
        self.value1 = self.combo_box1.get()
        # print(f"下拉框1选择了: {self.value1}")

    def on_select2(self, event):
        self.value2 = self.combo_box2.get()
        # print(f"下拉框2选择了: {self.value2}")


app = None
def run_gui(cands_1, cands_2):
    global app

    root = tk.Tk()
    app = ComboBoxTkApp(root, cands_1, cands_2)
    root.mainloop()


def run_combo_box(cands_1, cands_2):
    global app

    threading.Thread(target=partial(run_gui, cands_1=cands_1, cands_2=cands_2)).start()

    while app is None:
        time.sleep(0.1)

    return app


if __name__ == "__main__":
    # 在主线程中运行 GUI
    cands_1 = ["1", "2", "3"]
    cands_2 = ["a", "b", "c"]
    app = run_combo_box(cands_1=cands_1, cands_2=cands_2)

    while True:
        empty_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imshow("empty", empty_image)
        cv2.waitKey(1000)
        print(app.value1, app.value2)

    pass
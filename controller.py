import ctypes
import sys, os, time
import win32gui, win32ui, win32con, win32api, win32print
import cv2
import numpy as np
from win32com.client import Dispatch


def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        overlap = inter / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick]


class Controller:
    def __init__(self, HWND, AJ, task_queue) -> None:
        self.aj = AJ
        self.task_queue = task_queue
        self.hwnd = HWND

        self.w_ratio, self.h_ratio = self.set_window_size()
        self.__kqhoutai()

    # 错误对话框提示
    def log_text(self, msg: str, type="log"):
        self.task_queue.put({"msg": msg, "type": type}, block=True)
        if type == "error":
            time.sleep(2)

    def set_window_size(self):
        w_width = 465
        w_height = 850

        hDC = win32gui.GetDC(0)
        high = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
        height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

        rect_ratio = round(high / height, 2)

        if height > 1080:
            self.aj.SetWindowSize(self.hwnd, 558, 1020)
            return round(558 / w_width * rect_ratio, 2), round(1020 / w_height * rect_ratio, 2)
        elif height >= 1024:
            self.aj.SetWindowSize(self.hwnd, 465, 850)
            return round(465 / w_width * rect_ratio, 2), round(850 / w_height * rect_ratio, 2)
        else:
            self.aj.SetWindowSize(self.hwnd, 372, 680)
            return round(372 / w_width * rect_ratio, 2), round(680 / w_height * rect_ratio, 2)


    def ishas(self):
        res = self.aj.GetWindowState(self.hwnd, 0)
        if res == 1:
            return True

    def __kqhoutai(self):
        self.aj.GBHouTai()
        res = self.aj.KQHouTai(self.hwnd, "FD", "WM", "WM", "LAA|LAM", 0)

    def click(self, x, y):
        if self.ishas():
            x = int(x * self.w_ratio)
            y = int(y * self.h_ratio)
            self.aj.MoveTo(x, y)
            self.aj.LeftClick()

    def swipe(self, sx, sy, ex, ey, delay):
        """滑动"""
        if self.ishas():
            sx = int(sx * self.w_ratio)
            sy = int(sy * self.h_ratio)
            ex = int(ex * self.w_ratio)
            ey = int(ey * self.h_ratio)
            # print("开始移动")
            self.aj.MoveTo(sx, sy)
            self.aj.LeftDown()
            self.aj.MoveTo(ex, ey)
            time.sleep(delay)
            self.aj.LeftUp()
            # print("到达")


    def captrure_win32(self):
        """
        通过win32方式截图
        """
        ctypes.windll.user32.SetProcessDPIAware()

        rect = win32gui.GetWindowRect(self.hwnd)

        width, height = rect[2] - rect[0], rect[3] - rect[1]

        hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        for i in range(10):
            try:
                save_dc = mfc_dc.CreateCompatibleDC()
                break
            except:
                time.sleep(0.1)
        else:
            return

        save_bit_map = win32ui.CreateBitmap()
        save_bit_map.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(save_bit_map)

        ctypes.windll.user32.PrintWindow(self.hwnd, save_dc.GetSafeHdc(), 3)
        bmpinfo = save_bit_map.GetInfo()
        bmpstr = save_bit_map.GetBitmapBits(True)

        capture = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
        capture = np.ascontiguousarray(capture)[..., :-1]

        win32gui.DeleteObject(save_bit_map.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwnd_dc)

        capture = cv2.cvtColor(capture, cv2.COLOR_RGBA2RGB)
        # cv2.imwrite("im_opencv.png", capture)
        return capture

    def find_pic(self, template: str, threshold: float, tp=None, show_res=False):
        """找单图"""
        threshold = 1 - threshold
        template_list = template.split("|")

        for template in template_list:
            scr = cv2.imread(self.get_resource_path(template))

            if scr is None:
                self.log_text("错误：目标图片文件缺失", "error")

            # 小程序菜单图标和重新进入小程序图标在任何分辨率不需要缩放
            if "cdan.png" not in template and "cxjr.png" not in template:
                scr = cv2.resize(scr, (int(scr.shape[1] * self.w_ratio), int(scr.shape[0] * self.h_ratio)))

            for i in range(3):
                if tp is not None:
                    break
                tp = self.captrure_win32()  # 截图
            else:
                return

            try:
                result = cv2.matchTemplate(scr, tp, cv2.TM_SQDIFF_NORMED)
            except Exception as e:
                # print(e)
                self.log_text("图片资源错误，请检查游戏是否正常运行", "error")
            h, w = scr.shape[:2]
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if show_res:
                print(template, 1 - min_val, min_loc)
            if min_val <= threshold:
                coordinate = (
                    min_loc[0] + int(w / 2),
                    min_loc[1] + int(h / 2),
                    min_loc[0],
                    min_loc[1],
                    min_loc[0] + w,
                    min_loc[1] + h,
                )

                return coordinate

    def find_pic_all(self, template: str, threshold: float, tp=None) -> list:
        """找多图"""
        threshold = 1 - threshold
        scr = cv2.imread(self.get_resource_path(template))

        if scr is None:
            self.log_text("错误：目标图片文件缺失", "error")

        scr = cv2.resize(scr, (int(scr.shape[1] * self.w_ratio), int(scr.shape[0] * self.h_ratio)))

        if tp is None:
            tp = self.captrure_win32()

        result = cv2.matchTemplate(tp, scr, cv2.TM_SQDIFF_NORMED)
        h, w = scr.shape[:2]

        loc = np.where(result <= threshold)

        coordinates = []
        for pt in zip(*loc[::-1]):
            coordinate = (
                int(pt[0] + w / 2),
                int(pt[1] + h / 2),
                int(pt[0]),
                int(pt[1]),
                int(pt[0] + w),
                int(pt[1] + h),
            )
            coordinates.append(coordinate)

        boxes = np.array([(c[2], c[3], c[4], c[5]) for c in coordinates])
        filtered_boxes = non_max_suppression(boxes, overlapThresh=0.1)

        filtered_coordinates = []
        for box in filtered_boxes:
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            coordinate = (center_x, center_y, int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            filtered_coordinates.append(coordinate)

        return filtered_coordinates

    def get_resource_path(self, relative_path):
        """获取资源文件的绝对路径"""
        if hasattr(sys, "_MEIPASS"):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)


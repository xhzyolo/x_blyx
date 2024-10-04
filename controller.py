import ctypes
import sys, os, time
import win32gui, win32ui, win32con
import cv2
import numpy as np

# import atexit
from win32com.client import Dispatch


# def onexit():
#     if not HWND:
#         return
#     x = Controller(CLASSNAME,start=False)
#     x.exitclose(HWND)


# atexit.register(onexit)


def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    # 初始化选择的框
    pick = []

    # 获取边界框的坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 计算边界框的面积
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 按照边界框的某个坐标排序
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # 计算与当前框重叠的框的交集
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # 计算交集的宽度和高度
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # 计算交集的面积
        inter = w * h

        # 计算重叠率
        overlap = inter / area[idxs[:last]]

        # 删除重叠率大于阈值的框
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick]


class Controller:
    def __init__(self, HWND, AJ) -> None:
        self.aj = AJ
        self.hwnd = HWND
        self.aj.SetWindowSize(self.hwnd, 465, 850)
        self.__kqhoutai()

    # def __ajreg(self):
    #     """注册插件"""
    #     try:
    #         ARegJ = ctypes.windll.LoadLibrary(os.getcwd() + "\\ARegJ64.dll")
    #         ARegJ.SetDllPathW(os.getcwd() + "\\AoJia64.dll", 0)
    #         AJ = Dispatch("AoJia.AoJiaD")
    #         if AJ.VerS() != 0:
    #             return AJ
    #         return False
    #     except Exception as e:
    #         print(e)
    #         return False

    def ishas(self):
        res = self.aj.GetWindowState(self.hwnd, 0)
        if res == 1:
            return True

    def __kqhoutai(self):
        """开启后台"""
        res = self.aj.KQHouTai(self.hwnd, "FD", "WM", "WM", "LAA|LAM", 0)
        # if res == 0:
        #     print("后台绑定失败，程序将在前台运行")

    def click(self, x, y):
        """单击"""
        if self.ishas():
            self.aj.MoveTo(x, y)
            self.aj.LeftClick()

    def swipe(self, sx, sy, ex, ey, delay):
        """滑动"""
        if self.ishas():
            # print("开始移动")
            self.aj.MoveTo(sx, sy)
            self.aj.LeftDown()
            self.aj.MoveTo(ex, ey)
            time.sleep(delay)
            self.aj.LeftUp()
            # print("到达")

    def screenshot(self, sx, sy, ex, ey, pic, type=0, quality=0, td=0, t=0, flag=1, mouse=0):
        """截图"""
        return self.aj.ScreenShot(sx, sy, ex, ey, pic, type, quality, td, t, flag, mouse)

    def findpic(self, sx, sy, ex, ey, pic, color=111111, sim=0.9, dir=9, type=1):
        """找图"""
        return self.aj.FindPic(sx, sy, ex, ey, pic, color, sim, dir, type)

    def findcolor(self, sx, sy, ex, ey, color, sim=0.9, dir=9):
        """找色"""
        return self.aj.FindColor(sx, sy, ex, ey, color, sim, dir)

    def getcolor(self, x, y, type=1, typed=1):
        return self.aj.GetColor(x, y, type, typed)

    def captrure_win32(self):
        """
        通过win32方式截图
        """
        ctypes.windll.user32.SetProcessDPIAware()
        # hwnd = win32gui.FindWindow(None, "百炼英雄")

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

    def find_pic(self, template: str, threshold: float, tp=None):
        """找单图"""
        threshold = 1 - threshold
        template_list = template.split("|")

        for template in template_list:
            scr = cv2.imread(self.get_resource_path(template))

            for i in range(3):
                if tp is not None:
                    break
                tp = self.captrure_win32()  # 截图
            else:
                return

            result = cv2.matchTemplate(scr, tp, cv2.TM_SQDIFF_NORMED)
            h, w = scr.shape[:2]
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            # print(min_val, min_loc)
            if min_val <= threshold:
                coordinate = (
                    min_loc[0] + int(w / 2),
                    min_loc[1] + int(h / 2),
                    min_loc[0],
                    min_loc[1],
                    min_loc[0] + w,
                    min_loc[1] + h,
                )
                # print(coordinate)
                return coordinate

    def find_pic_all(self, template: str, threshold: float, tp=None) -> list:
        """找多图"""
        threshold = 1 - threshold
        # scr = cv2.imread(template, cv2.IMREAD_UNCHANGED) 这种方法无法程序出错
        scr = cv2.imread(self.get_resource_path(template))
        if tp is None:
            tp = self.captrure_win32()  # 截图

        # 使用 TM_SQDIFF_NORMED 方法进行模板匹配
        result = cv2.matchTemplate(tp, scr, cv2.TM_SQDIFF_NORMED)
        h, w = scr.shape[:2]

        # 找到所有匹配度低于阈值的位置
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

        # 使用非极大值抑制过滤重叠的匹配结果
        boxes = np.array([(c[2], c[3], c[4], c[5]) for c in coordinates])
        # 调整overlapThresh参数来控制重叠的匹配结果过滤的阈值，范围为0到1之间，越小则过滤的越少
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
            # PyInstaller 创建的临时目录
            base_path = sys._MEIPASS
        else:
            # 开发环境中的当前目录
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)


# if __name__ == "__main__":
#     hwnd = win32gui.FindWindow(None, "百炼英雄")
#     ARegJ = ctypes.windll.LoadLibrary(os.getcwd() + "\\ARegJ64.dll")
#     ARegJ.SetDllPathW(os.getcwd() + "\\AoJia64.dll", 0)
#     AJ = Dispatch("AoJia.AoJiaD")
#     c = Controller(hwnd, AJ)

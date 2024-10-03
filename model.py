import time
import win32gui
import threading
import controller
import ctypes
import os, sys
from win32com.client import Dispatch


COLOR_FILE = {
    "白": "images/color/white.png",
    "蓝": "images/color/blue.png",
    "紫": "images/color/purple.png",
    "金": "images/color/golden.png",
    "红": "images/color/red.png",
}

HERO_RED = {
    "王子": "images/hero/wangzi.png",
    "黑寡妇": "images/hero/heiguafu.png",
    "德鲁伊": "images/hero/deluyi.png",
    "敖丙": "images/hero/aobing.png",
    "酒神": "images/hero/jiushen.png",
    "圣骑": "images/hero/shengqi.png",
    "李白": "images/hero/libai.png",
    "卡卡西": "images/hero/kakaxi.png",
}

HERO_GLODEN = {
    "天使": "images/hero/tianshi.png",
    "舞姬": "images/hero/wuji.png",
}


ZB_ZUIDA = (355, 560)  # 最大/3倍按钮
ZB_ZHAOMU = (230, 615)  # 招募按钮
ZB_QUEREN = (300, 515)  # 确认按钮
ZB_FANGQI = (223, 672)  # 放弃按钮

DELAY_ZHAOMU = 2.2


class Model:
    def __init__(self, task_queue, config_data):
        self.task_queue = task_queue
        self.config_data = config_data
        # config_data: {'所有金色': False, '双枪': True, '猎人': False, '天使': True, '白无常': False, '舞姬': False, 'cards_count': '0', 'h_stop': 'False'}

    def get_resource_path(self, relative_path):
        """获取资源文件的绝对路径"""
        if hasattr(sys, "_MEIPASS"):
            # PyInstaller 创建的临时目录
            base_path = sys._MEIPASS
        else:
            # 开发环境中的当前目录
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def __ajreg(self):
        """注册插件"""
        try:
            ARegJ = ctypes.windll.LoadLibrary(self.get_resource_path("ARegJ64.dll"))
            ARegJ.SetDllPathW(self.get_resource_path("AoJia64.dll"), 0)
            AJ = Dispatch("AoJia.AoJiaD")
            if AJ.VerS() != 0:
                return AJ
        except Exception as e:
            # print("插件注册失败:", e)
            pass

        self.log_text("插件初始化失败！", "error")

    # def __ajreg(self):
    #     """注册插件"""
    #     try:
    #         ARegJ = ctypes.windll.LoadLibrary(os.getcwd() + "\\ARegJ64.dll")
    #         ARegJ.SetDllPathW(os.getcwd() + "\\AoJia64.dll", 0)
    #         AJ = Dispatch("AoJia.AoJiaD")
    #         if AJ.VerS() != 0:
    #             return AJ
    #     except Exception as e:
    #         print("插件注册失败:",e)

    #     self.log_text("插件初始化失败！","error")

    def __gethwnd(self):
        """获取窗口句柄"""
        hwnd = win32gui.FindWindow(None, "百炼英雄")
        if not hwnd:
            self.log_text("未找到游戏窗口，程序停止", "error")
        return hwnd

    def update_res(self, up_data: dict):
        """更新UI界面结果"""
        # dict = {"双枪":1}
        self.task_queue.put({"data": up_data, "type": "update"}, block=True)

    def log_text(self, msg: str, type="log"):
        """错误对话框提示"""
        self.task_queue.put({"msg": msg, "type": type}, block=True)

    def exit_process(self):
        """停止程序"""
        self.task_queue.put({"type": "stop"}, block=True)

    def run_cards(self):
        """主线程"""
        try:
            # print("开始执行")
            self.ctrl = controller.Controller(self.__gethwnd(), self.__ajreg())
            if not self.ctrl.find_pic("images/zmui1.png", 0.9):
                self.log_text("请进入招募界面再启动程序", "error")

            check = threading.Thread(target=self.run_check_thread, daemon=True)
            check.start()

            self.run_cards_thread()
        except Exception as e:
            # print(e)
            with open("error.log", "w") as f:
                f.write(str(e.__class__) + str(e))
            self.log_text("程序出错", "error")

    def run_check_thread(self):
        """检测线程"""
        while True:
            if not self.ctrl.ishas():
                self.log_text("游戏窗口关闭，程序停止", "error")
                break

            for i in range(3):
                if self.ctrl.find_pic("images/zmui1.png", 0.9) or self.ctrl.find_pic("images/zmui2.png", 0.9):
                    break
                else:
                    time.sleep(0.2)
            else:
                self.log_text("已退出招募界面，程序停止", "error")
                break

            if self.ctrl.find_pic("images/jb100_h.png", 0.9):
                self.log_text("金币不足！", "error")
                break

            time.sleep(1)

    def run_cards_thread(self):
        """主逻辑"""
        # 判断颜色
        count = 0
        while True:
            if not self.ctrl.find_pic("images/jb100.png", 0.9):
                # 执行放弃
                self.log_text("执行放弃")
                self.ctrl.click(*ZB_FANGQI)
                time.sleep(0.3)
                self.ctrl.click(*ZB_QUEREN)
            # 获取颜色
            color_list = self.get_color_list()
            # print(color_list)
            self.log_text(",".join(color_list))
            # 获取英雄
            hero_list = self.get_hero_list(color_list)
            # print(hero_list)
            new_hero = [hero for hero in hero_list if hero != "空"]
            if new_hero:
                self.log_text("发现英雄：%s" % ",".join(new_hero))
            # 三倍
            is3s = self.check_3s(color_list, hero_list)
            # 招募
            delay, index = self.zhaomu(hero_list, is3s)
            if index != -1 and hero_list[index] != "空":
                self.log_text("获得英雄：%s" % hero_list[index])
                self.update_res({hero_list[index]: 1})
            if delay:
                time.sleep(DELAY_ZHAOMU)
            count += 1
            if self.config_data["cards_count"] != "0":
                if count >= int(self.config_data["cards_count"]):
                    self.log_text("已招募%s次，程序停止" % count, "error")

        # 判断英雄
        # 判断是否跳转按钮
        # 判断是否选完
        # 结果
        # 判断是否下一轮
        # 判断次数

    def get_color_list(self):
        """获取颜色列表"""
        color_list = ["空", "空", "空"]
        while True:
            screen_shot = self.ctrl.captrure_win32()
            # 是否在招募界面
            if not self.ctrl.find_pic("images/zm.png", 0.9) and not self.ctrl.find_pic("images/jb100.png", 0.9):
                time.sleep(0.2)
                continue
            for key, value in COLOR_FILE.items():
                # print("key:", key)
                res_list = self.ctrl.find_pic_all(value, 0.99, tp=screen_shot)
                for res in res_list:
                    # print("res:", res)
                    if res[1] > 530:
                        continue
                    if res[0] < 150:
                        color_list[0] = key
                        continue
                    if res[0] < 300:
                        color_list[1] = key
                        continue
                    if res[0] < 450:
                        color_list[2] = key
                if "空" not in color_list:
                    return color_list

    def get_hero_list(self, color_list):
        """获取英雄列表"""
        hero_list = ["空", "空", "空"]
        screen_shot = self.ctrl.captrure_win32()
        if "红" in color_list:
            if self.config_data["h_stop"] == "True":
                self.log_text("发现红色英雄，程序停止", "error")

            for key, value in HERO_RED.items():
                res = self.ctrl.find_pic(value, 0.95, tp=screen_shot)
                if res:
                    if res[0] < 150 and color_list[0] == "红":
                        hero_list[0] = key
                        continue
                    if res[0] < 300 and color_list[1] == "红":
                        hero_list[1] = key
                        continue
                    if res[0] < 450 and color_list[0] == "红":
                        hero_list[2] = key

        if "金" in color_list:
            for key, value in HERO_GLODEN.items():
                res = self.ctrl.find_pic(value, 0.95, tp=screen_shot)
                if res:
                    if res[0] < 150 and color_list[0] == "金":
                        hero_list[0] = key
                        continue
                    if res[0] < 300 and color_list[1] == "金":
                        hero_list[1] = key
                        continue
                    if res[0] < 450 and color_list[2] == "金":
                        hero_list[2] = key

        return hero_list

    def check_3s(self, color_list, hero_list):
        if "红" in color_list:
            return True

        for hero in hero_list:
            if hero == "空":
                continue
            # 如果有配置文件则三倍
            if self.config_data[hero] or self.config_data[hero] == "True":
                return True

    def zhaomu(self, hero_list, is3s):
        """招募"""
        if is3s:
            self.log_text("三倍招募")
            self.ctrl.click(*ZB_ZUIDA)
            money = 300
            time.sleep(0.2)
        else:
            money = 100
        time.sleep(0.1)
        self.ctrl.click(*ZB_ZHAOMU)
        time.sleep(0.1)
        self.ctrl.click(*ZB_ZHAOMU)
        self.update_res({"总次数": 1, "总消耗": money})
        time.sleep(3)
        while True:
            self.ctrl.click(*ZB_QUEREN)
            time.sleep(0.5)
            ok_rect = self.ctrl.find_pic("images/ok.png", 0.99)
            if ok_rect:
                self.log_text("执行放弃")
                self.ctrl.click(*ZB_FANGQI)
                time.sleep(0.2)
                self.ctrl.click(*ZB_QUEREN)
                if ok_rect[0] < 150:
                    return True, 0
                if ok_rect[0] < 300:
                    return True, 1
                return True, 2
            if self.ctrl.find_pic("images/jb100.png", 0.9):
                self.log_text("招募完成")
                for index, hero in enumerate(hero_list):
                    if hero == "空":
                        continue
                    return False, index
                else:
                    return False, -1

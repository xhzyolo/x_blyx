import time
import win32gui
import threading
import controller
import ctypes
import os, sys
import config as cf
from win32com.client import Dispatch


class Model:
    def __init__(self, task_queue, config_data):
        self.task_queue = task_queue
        self.config_data = config_data
        self.flag_rest = False
        self.aj = None
        self.threshold = 0.99  # 英雄颜色识别精度
        # {'cards': {'天使': True, '舞姬': True, '王子': True, '黑寡妇': False, '德鲁伊': False, '敖丙': False, '酒神': True, '圣骑': False, '李白': True, '卡卡西': True, 'cards_count': '0', 'h_stop': 'False'}, 'coins': {'树精领主enable': False, '树精领主delay': 0.5, '树精长老enable': True, '树精长老delay': 13.0, '火焰石像enable': False, '火焰石像delay': 1.0, '疯牛魔王enable': False, '疯牛魔王delay': 1.5, '剧毒蝎王enable': True, '剧毒蝎王delay': 0.5, 'refresh_mode': 1, 'coins_mode': 1}}

    # 获取资源文件绝对路径 通常  C:\Windows\Temp\_MEIxxxx 或 C:\Users\用户名\AppData\Local\Temp\_MEIxxxx
    def get_resource_path(self, relative_path):
        """获取资源文件的绝对路径"""
        if hasattr(sys, "_MEIPASS"):
            # PyInstaller 创建的临时目录
            base_path = sys._MEIPASS
        else:
            # 开发环境中的当前目录
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    # 插件注册
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

        self.log_text("错误：插件初始化失败！请尝试关闭杀毒软件。", "error")

    # 获取窗口句柄
    def __gethwnd(self):
        """获取窗口句柄"""
        hwnd = win32gui.FindWindow(None, "百炼英雄")
        if not hwnd:
            # self.log_text("未找到游戏窗口，程序停止", "error")
            return None
        return hwnd

    # 更新UI界面结果
    def update_res(self, up_data: dict):
        """更新UI界面结果"""
        # dict = {"双枪":1}
        self.task_queue.put({"data": up_data, "type": "update"}, block=True)

    # 错误对话框提示
    def log_text(self, msg: str, type="log"):
        """错误对话框提示"""
        self.task_queue.put({"msg": msg, "type": type}, block=True)
        if type == "error":
            time.sleep(2)

    # 退出程序
    def exit_process(self):
        """停止程序"""
        self.task_queue.put({"type": "stop"}, block=True)

    # 抽卡主线程
    def run_cards(self):
        """主线程"""
        # print("开始执行")
        if self.aj is None:
            self.aj = self.__ajreg()

        hwnd = self.__gethwnd()
        if hwnd is None:
            self.log_text("未找到游戏窗口，程序停止", "error")
            return
        self.ctrl = controller.Controller(hwnd, self.aj, self.task_queue)

        if not self.ctrl.find_pic("images/zmui1.png", 0.95):
            self.log_text("请进入招募界面再启动程序", "error")

        check = threading.Thread(target=self.run_check_thread, daemon=True)
        check.start()

        self.run_cards_thread()

    # 抽卡检测线程
    def run_check_thread(self):
        """检测线程"""
        while True:
            if not self.ctrl.ishas():
                self.log_text("游戏窗口关闭，程序停止", "error")
                break

            for i in range(3):
                if self.ctrl.find_pic("images/zmui1.png", 0.95) or self.ctrl.find_pic("images/zmui2.png", 0.95):
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

    # 抽卡逻辑线程
    def run_cards_thread(self):
        """主逻辑"""
        # 判断颜色
        count = 0
        while True:
            try:
                if not self.ctrl.find_pic("images/jb100.png|images/jb90.png|images/zhe.png", 0.95):
                    # 执行放弃
                    self.log_text("执行放弃")
                    self.ctrl.click(*cf.ZB_FANGQI)
                    time.sleep(0.3)
                    self.ctrl.click(*cf.ZB_QUEREN)
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
                is3s = self.check_3s(hero_list)
                # 招募
                delay, index = self.zhaomu(hero_list, is3s)
                if index != -1 and hero_list[index] != "空":
                    self.log_text("获得英雄：%s" % hero_list[index])
                    self.update_res({hero_list[index]: 1})
                if delay:
                    time.sleep(cf.DELAY_ZHAOMU)
                count += 1
                if self.config_data["cards"]["cards_count"] != "0":
                    if count >= int(self.config_data["cards"]["cards_count"]):
                        self.log_text("已招募%s次，程序停止" % count, "error")
            except Exception as e:
                # print(e)
                self.error_write(e)
                continue

    # 获取颜色列表
    def get_color_list(self):
        """获取颜色列表"""
        color_list = ["空", "空", "空"]
        i = 1
        while True:
            screen_shot = self.ctrl.captrure_win32()
            # 是否在招募界面
            if not self.ctrl.find_pic("images/zm.png", 0.95) and not self.ctrl.find_pic(
                "images/jb100.png|images/jb90.png|images/zhe.png", 0.95
            ):
                time.sleep(0.2)
                continue
            for key, value in cf.COLOR_FILE.items():
                # print("key:", key)
                res_list = self.ctrl.find_pic_all(value, self.threshold, tp=screen_shot)
                for res in res_list:
                    # print("res:", res)
                    if res[1] > cf.ZB_BOTTOM * self.ctrl.h_ratio:
                        continue
                    if res[0] < cf.ZB_RIGHT1 * self.ctrl.w_ratio:
                        color_list[0] = key
                        continue
                    if res[0] < cf.ZB_RIGHT2 * self.ctrl.w_ratio:
                        color_list[1] = key
                        continue
                    if res[0] < cf.ZB_RIGHT3 * self.ctrl.w_ratio:
                        color_list[2] = key
                if "空" not in color_list:
                    return color_list

            if i > 10:
                if self.threshold == 0.99:
                    self.log_text("未识别到英雄，尝试降低识别精度")
                    self.threshold = 0.98
                else:
                    self.log_text("未识别到英雄，程序退出", "error")
            i += 1

    # 获取英雄列表
    def get_hero_list(self, color_list: list):
        """获取英雄列表"""
        hero_list = ["空", "空", "空"]
        screen_shot = self.ctrl.captrure_win32()
        if "红" in color_list:
            if self.config_data["cards"]["h_stop"] == "True":
                self.log_text("发现红色英雄，程序停止", "error")

            for _ in range(10):  # 用于多次检测，有10次容错
                for key, value in cf.HERO_RED.items():
                    res = self.ctrl.find_pic(value, 0.95, tp=screen_shot)
                    if res:
                        if res[0] < cf.ZB_RIGHT1 * self.ctrl.w_ratio and color_list[0] == "红":
                            hero_list[0] = key
                            continue
                        if res[0] < cf.ZB_RIGHT2 * self.ctrl.w_ratio and color_list[1] == "红":
                            hero_list[1] = key
                            continue
                        if res[0] < cf.ZB_RIGHT3 * self.ctrl.w_ratio and color_list[2] == "红":
                            hero_list[2] = key
                if color_list.count("红") == 3 - hero_list.count("空"):
                    break

        if "金" in color_list:
            for _ in range(10):  # 用于多次检测，有10次容错
                for key, value in cf.HERO_GLODEN.items():
                    res = self.ctrl.find_pic(value, 0.97, tp=screen_shot)
                    if res:
                        if res[0] < cf.ZB_RIGHT1 * self.ctrl.w_ratio and color_list[0] == "金":
                            hero_list[0] = key
                            continue
                        if res[0] < cf.ZB_RIGHT2 * self.ctrl.w_ratio and color_list[1] == "金":
                            hero_list[1] = key
                            continue
                        if res[0] < cf.ZB_RIGHT3 * self.ctrl.w_ratio and color_list[2] == "金":
                            hero_list[2] = key
                if color_list.count("金") == 3 - hero_list.count("空") - hero_list.count("红"):
                    break
        return hero_list

    # 检查是否三倍
    def check_3s(self, hero_list):
        for hero in hero_list:
            if hero == "空":
                continue
            # 如果有配置文件则三倍
            if self.config_data["cards"][hero] or self.config_data["cards"][hero] == "True":
                return True

    # 招募
    def zhaomu(self, hero_list, is3s):
        """招募"""
        # 如何不是金币则返回
        if not self.ctrl.find_pic("images/jb100.png|images/jb90.png|images/zhe.png", 0.95):
            return False, -1

        if is3s:
            self.log_text("三倍招募")
            self.ctrl.click(*cf.ZB_ZUIDA)
            money = 300
            time.sleep(0.2)
        else:
            money = 100
        time.sleep(0.1)
        self.ctrl.click(*cf.ZB_ZHAOMU)
        time.sleep(0.1)
        self.ctrl.click(*cf.ZB_ZHAOMU)
        self.update_res({"总次数": 1, "总消耗": money})
        time.sleep(3)
        while True:
            self.ctrl.click(*cf.ZB_QUEREN)
            time.sleep(0.5)
            ok_rect = self.ctrl.find_pic("images/ok.png|images/ok2.png", self.threshold)
            if ok_rect:
                self.log_text("执行放弃")
                self.ctrl.click(*cf.ZB_FANGQI)
                time.sleep(0.2)
                self.ctrl.click(*cf.ZB_QUEREN)
                if ok_rect[0] < cf.ZB_RIGHT1 * self.ctrl.w_ratio:
                    return True, 0
                if ok_rect[0] < cf.ZB_RIGHT2 * self.ctrl.w_ratio:
                    return True, 1
                return True, 2
            if self.ctrl.find_pic("images/jb100.png|images/jb90.png|images/zhe.png", 0.95):
                self.log_text("招募完成")
                for index, hero in enumerate(hero_list):
                    if hero == "空":
                        continue
                    return False, index
                else:
                    return False, -1

    # 刷金主线程
    def run_coins(self):
        """主线程"""
        # print("开始执行刷金")
        if self.aj is None:
            self.aj = self.__ajreg()

        hwnd = self.__gethwnd()
        if hwnd is None:
            self.log_text("未找到游戏窗口，程序停止", "error")
            return
        self.ctrl = controller.Controller(hwnd, self.aj, self.task_queue)

        # 开始检测线程
        check = threading.Thread(target=self.run_coins_check_thread, daemon=True)
        check.start()

        # 判断模式
        if self.config_data["coins"]["coins_mode"] == 1:
            self.log_text("开始执行自定义模式")
            self.run_coins_custom_thread()
        elif self.config_data["coins"]["coins_mode"] == 2:
            self.log_text("开始执行默认模式")
            self.run_coins_default_thread()

    def run_coins_check_thread(self):
        pass

    # {'cards': {'天使': True, '舞姬': True, '王子': True, '黑寡妇': False, '德鲁伊': False, '敖丙': False, '酒神': True, '圣骑': False, '李白': True, '卡卡西': True, 'cards_count': '0', 'h_stop': 'False'}, 'coins': {'树精领主enable': False, '树精领主delay': 0.5, '树
    # 精长老enable': True, '树精长老delay': 13.0, '火焰石像enable': False, '火焰石像delay': 1.0, '疯牛魔王enable': False, '疯牛魔王delay': 1.5, '剧毒蝎王enable': True, '剧毒蝎王delay': 0.5, 'refresh_mode': 1, 'coins_mode': 1}}
    # 自定义刷金模式
    def run_coins_custom_thread(self):
        count = 1
        custom_script = self.config_data["script"]
        # print(custom_script)
        custom_script = [x for x in custom_script if not x.startswith(("#", "//"))]
        if len(custom_script) == 0:
            self.log_text("自定义模式：未配置脚本，程序停止", type="error")
            return
        while True:
            self.log_text(f"自定义模式：第{count}轮")
            for i, script in enumerate(custom_script):
                if script.startswith(("#", "//")):
                    continue

                # 获取函数名和参数
                func_parts = script.split("(")
                func_name = func_parts[0].strip()
                args = func_parts[1].rstrip(")").split(",")

                if func_name in cf.FUNC_LIST:
                    try:
                        func = getattr(self, func_name)
                        _args = []
                        for arg in args:
                            arg = arg.strip()
                            if "." in arg:
                                arg = float(arg)
                            elif arg.isdigit():
                                arg = int(arg)
                            else:
                                arg = arg
                            _args.append(arg)

                        # print(f"执行：{func_name}{_args}")
                        if len(_args) and _args[0]:
                            func(*_args)
                        else:
                            func()

                        if self.flag_rest:
                            self.flag_rest = False
                            self.log_text("多次未找到传送点，尝试重新启动游戏")
                            self.重启()
                            break
                    except Exception as e:
                        # print("错误：", e)
                        self.error_write(e)
                        self.log_text(f"第{i+1}行脚本有误，请检查", type="error")
                        continue
                else:
                    self.log_text(f"第{i+1}行脚本有误，请检查", type="error")
                    continue
            count += 1

    # 默认刷金模式
    def run_coins_default_thread(self):
        count = 1

        for key, value in self.config_data["coins"].items():
            if "enable" in key and value == True:
                break
        else:
            self.log_text("默认模式：未开启任何功能，程序停止", type="error")
            return
        while True:
            self.log_text(f"默认模式：第{count}轮")

            if self.config_data["coins"]["refresh_mode"] == 1:
                # 树精长老
                if self.config_data["coins"]["树精长老enable"]:
                    self.传送(3, 3)
                    self.移动("右", 1.5)
                    self.移动("右上", 1.7)
                    self.等待(self.config_data["coins"]["树精长老delay"])
                    self.移动("左下", 1.7)
                    self.移动("左", 1.2)

                # 火焰石像
                if (
                    self.config_data["coins"]["火焰石像enable"]
                    or self.config_data["coins"]["疯牛魔王enable"]
                    or self.config_data["coins"]["剧毒蝎王enable"]
                ):
                    self.传送(3, 1)
                    self.移动("右", 1.5)
                    self.移动("右下", 0.5)
                    self.移动("下", 0.7)
                    self.移动("左下", 2.7)
                    self.移动("下", 1)
                    if self.config_data["coins"]["火焰石像enable"]:
                        self.等待(self.config_data["coins"]["火焰石像delay"])

                    if self.config_data["coins"]["疯牛魔王enable"] or self.config_data["coins"]["剧毒蝎王enable"]:
                        self.移动("下", 6.4)
                        if self.config_data["coins"]["疯牛魔王enable"]:
                            self.等待(self.config_data["coins"]["疯牛魔王delay"])

                        if self.config_data["coins"]["剧毒蝎王enable"]:
                            self.移动("右下", 3.2)
                            self.移动("下", 1)
                            self.移动("右下", 2.1)
                            self.移动("右", 1.5)
                            self.移动("右下", 2)
                            self.移动("右", 2.8)
                            self.移动("右下", 1)
                            self.等待(self.config_data["coins"]["剧毒蝎王delay"])

                    self.回城()

                # 树精领主
                if self.config_data["coins"]["树精领主enable"]:
                    self.传送(2, 1)
                    self.移动("右", 1)
                    self.移动("右上", 1)
                    self.等待(self.config_data["coins"]["树精领主delay"])
                    self.移动("左下", 1)
                    self.移动("左", 0.7)

                self.log_text("王座刷新")
                self.传送(5, 1)

            elif self.config_data["coins"]["refresh_mode"] == 2:
                # 树精长老
                if self.config_data["coins"]["树精长老enable"]:
                    self.传送(3, 3)
                    self.移动("右", 1.5)
                    self.移动("右上", 1.7)
                    self.等待(self.config_data["coins"]["树精长老delay"])
                    self.移动("左下", 1.7)
                    self.移动("左", 1.2)

                # 树精领主
                if self.config_data["coins"]["树精领主enable"]:
                    self.传送(2, 1)
                    self.移动("右", 1)
                    self.移动("右上", 1)
                    self.等待(self.config_data["coins"]["树精领主delay"])
                    self.移动("左下", 1)
                    self.移动("左", 0.7)

                # 火焰石像
                if (
                    self.config_data["coins"]["火焰石像enable"]
                    or self.config_data["coins"]["疯牛魔王enable"]
                    or self.config_data["coins"]["剧毒蝎王enable"]
                ):
                    self.传送(3, 1)
                    self.移动("右", 1.5)
                    self.移动("右下", 0.5)
                    self.移动("下", 0.7)
                    self.移动("左下", 2.7)
                    self.移动("下", 1)
                    if self.config_data["coins"]["火焰石像enable"]:
                        self.等待(self.config_data["coins"]["火焰石像delay"])

                    if self.config_data["coins"]["疯牛魔王enable"] or self.config_data["coins"]["剧毒蝎王enable"]:
                        self.移动("下", 6.4)
                        if self.config_data["coins"]["疯牛魔王enable"]:
                            self.等待(self.config_data["coins"]["疯牛魔王delay"])

                        if self.config_data["coins"]["剧毒蝎王enable"]:
                            self.移动("右下", 3.2)
                            self.移动("下", 1)
                            self.移动("右下", 2.1)
                            self.移动("右", 1.5)
                            self.移动("右下", 2)
                            self.移动("右", 2.8)
                            self.移动("右下", 1)
                            self.等待(self.config_data["coins"]["剧毒蝎王delay"])

                self.log_text("重启刷新")
                self.重启()

    def 传送(self, section, map):
        self.log_text("执行传送")
        cs_rect = self.ctrl.find_pic("images/back.png", 0.95)
        if cs_rect:
            # 不在城里
            # print("不在城里")
            for i in range(10):
                cs_rect = self.ctrl.find_pic(cf.CS_PIC, 0.95)
                if cs_rect:
                    self.ctrl.click(cs_rect[0] / self.ctrl.w_ratio, cs_rect[1] / self.ctrl.h_ratio)
                    break
                self.log_text("未找到传送点")
                self.ctrl.click(*cf.ZB_KONGBAI)
                time.sleep(2)
            else:
                self.flag_rest = True
                return
        else:
            # 在城里
            # print("在城里")
            for i in range(10):
                dcr_rect = self.ctrl.find_pic("images/dcr.png", 0.95)
                if dcr_rect:
                    x = dcr_rect[0] / self.ctrl.w_ratio - 5
                    y = dcr_rect[1] / self.ctrl.h_ratio + 110
                else:
                    bx_rect = self.ctrl.find_pic("images/bx.png", 0.95)
                    if bx_rect:
                        x = bx_rect[0] / self.ctrl.w_ratio + 23
                        y = bx_rect[1] / self.ctrl.h_ratio + 64
                if dcr_rect or bx_rect:
                    if self.ctrl.find_pic(cf.CS_PIC, 0.95):
                        self.ctrl.click(x, y)
                        break
                self.log_text("未找到传送点")
                self.ctrl.click(*cf.ZB_KONGBAI)
                time.sleep(2)
            else:
                self.flag_rest = True
                return
        if self.config_data["coins"]["coins_mode"] == 2 and self.config_data["coins"]["refresh_mode"] == 1:
            time.sleep(1.3)
        else:
            time.sleep(0.6)
        self.ctrl.click(*cf.ZB_SECTION[str(section)])
        time.sleep(0.2)
        self.ctrl.click(*cf.ZB_MAP[str(map)])
        time.sleep(3)
        for _ in range(20):
            if self.ctrl.find_pic(cf.CS_PIC, 0.95):
                break
            time.sleep(0.5)

    def 移动(self, direction, delay):
        self.log_text("执行移动")
        self.ctrl.swipe(*cf.ZB_DIRECTION["中"], *cf.ZB_DIRECTION[direction], delay)

    def 等待(self, delay):
        self.log_text("执行等待", delay)
        time.sleep(delay)

    def 回城(self):
        self.log_text("执行回城")
        self.ctrl.click(*cf.ZB_HUICHENG)
        time.sleep(0.5)
        self.ctrl.click(*cf.ZB_QUEREN)
        time.sleep(3)
        for _ in range(20):
            if self.ctrl.find_pic(cf.CS_PIC, 0.95):
                break
            time.sleep(0.5)

    # 重启游戏和脚本
    def 重启(self):
        self.log_text("执行重启")
        for _ in range(10):
            cd_rect = self.ctrl.find_pic("images/cdan.png", 0.95)
            if cd_rect:
                self.ctrl.click(cd_rect[0] / self.ctrl.w_ratio, cd_rect[1] / self.ctrl.h_ratio)
                time.sleep(1)
                cxjr_rect = self.ctrl.find_pic("images/cxjr.png", 0.95)
                if cxjr_rect:
                    self.ctrl.click(cxjr_rect[0] / self.ctrl.w_ratio, cxjr_rect[1] / self.ctrl.h_ratio)
                    break
            self.log_text("未找到菜单按钮，重新尝试")
            time.sleep(2)
        else:
            self.error_write("未找到菜单按钮")
            self.log_text("多次未找到菜单按钮，重启失败", type="error")

        # 等待重启延迟
        time.sleep(5)

        for _ in range(40):
            try:
                hwnd = self.__gethwnd()
                if hwnd is None:
                    raise Exception
                self.ctrl = controller.Controller(hwnd, self.aj, self.task_queue)
                break
            except:
                time.sleep(0.5)
        else:
            self.error_write("重启后，长时间未进入游戏")
            self.log_text("重启游戏时间过长，请检查", type="error")

        for _ in range(30):
            if self.ctrl.find_pic(cf.CS_PIC + "|images/liaotian.png", 0.95):
                break
            time.sleep(0.5)

        time.sleep(2)
        self.log_text("关闭活动窗口")
        for _ in range(3):
            self.ctrl.click(*cf.ZB_KONGBAI)
            time.sleep(0.8)

    def error_write(self, msg):
        with open("error.log", "a", encoding="utf-8") as f:
            f.write(time.strftime("%m-%d %H:%M:%S : ", time.localtime()) + str(msg))

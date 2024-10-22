import tkinter as tk
import multiprocessing
import time
import os, sys
import configparser
import requests
import json
from tkinter import ttk
from tkinter import messagebox
from model import Model


BASE_URL = "https://xzdwz.com/api/blyx/blyx"


class App:
    def __init__(self, root, version):
        self.root = root
        self.version = version
        self.running = False
        self.label_show = {}
        self.label_count = {}
        self.config_data = {}
        self.p_running = None
        title = f"x-blyx   V: {self.version}"
        self.__create_window(title, 450, 550)
        self.__create_tabs()
        self.root.iconbitmap(self.get_resource_path("images/blyx.ico"))
        self.config = self.get_config()

        self.task_queue = multiprocessing.Queue()
        # 注册窗口关闭处理函数
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.check_version()

        self.log_text("载入成功!")

    # 检查版本
    def check_version(self):
        try:
            self.log_text("检查版本...")
            res = requests.post(BASE_URL, data={"action": "get_version"}, timeout=3).json()
            if res["errorCode"] == 0:
                min_version = res["data"]["min_version"]
                max_version = res["data"]["max_version"]
                if float(self.version) >= float(min_version):
                    if float(self.version) == float(max_version):
                        self.log_text("当前为最新版本")
                        return
                    elif float(self.version) < float(max_version):
                        self.log_text(f"发现新版本,v{max_version}")
                        return
                    elif float(self.version) > float(max_version):
                        self.log_text("当前为测试版本")
                        return
                else:
                    messagebox.showinfo(title="提示", message="当前版本过低，请更新到最新版本")
            else:
                messagebox.showerror(title="错误", message="版本获取失败，请重试")
        except requests.exceptions.Timeout:
            messagebox.showerror(title="错误", message="服务器错误，请稍后重试")
        self.root.destroy()

    # 获取资源文件绝对路径
    def get_resource_path(self, relative_path):
        """获取资源文件的绝对路径"""
        if hasattr(sys, "_MEIPASS"):
            # PyInstaller 创建的临时目录
            base_path = sys._MEIPASS
        else:
            # 开发环境中的当前目录
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    # 创建窗口
    def __create_window(self, title: str, w_width: int, w_height: int):
        """创建窗口"""
        # self.root.resizable(False, False)
        self.root.title(title)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        pos_x = (screen_width - w_width) // 2
        pos_y = (screen_height - w_height) // 2
        self.root.geometry(f"{w_width}x{w_height}+{pos_x}+{pos_y}")
        self.root.bind("<Button-1>", self.on_click)

    # 创建选项卡
    def __create_tabs(self):
        """创建选项卡"""
        self.tab_control = ttk.Notebook(self.root)
        self.tab_card = ttk.Frame(self.tab_control)
        self.tab_coin = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_card, text="抽卡")
        self.tab_control.add(self.tab_coin, text="金币")
        self.tab_control.pack(expand=1, fill="both")
        self.tab_control.bind("<<NotebookTabChanged>>", self.on_tab_change)

        # 选项卡一的内容
        self.create_tab_card()

        # 选项卡二的内容
        self.create_tab_coin()

        # 日志
        self.text = tk.Text(self.root, height=10)
        self.text.pack(side="top", fill="both", padx=5, pady=5)

        # 运行时间
        self.label_timer = tk.Label(self.root, text="")
        self.label_timer.pack(side="left", anchor="n", padx=(20, 0), pady=10)

        # 运行按钮
        self.button_run = tk.Button(self.root, text="运行", command=self.start_action)
        self.button_run.config(width=8, relief="solid")
        self.button_run.pack(side="right", anchor="n", padx=(0, 10), pady=10)

    # 选项卡一（抽卡）
    def create_tab_card(self):
        """选项卡一的总体框架"""
        top_half = ttk.Frame(self.tab_card)
        top_half.pack(side="top", fill="x")

        # 第一个区域
        frame_3s = ttk.Frame(top_half, borderwidth=2, relief="groove")
        frame_3s.pack(side="left", anchor="n", fill="both", expand="true", padx=2, pady=2)
        self.tab_card_frame_3s(frame_3s)

        # 第二区域
        frame_config = ttk.Frame(top_half, borderwidth=2, relief="groove")
        frame_config.pack(side="right", anchor="n", fill="both", expand="true", padx=2, pady=2)
        self.tab_card_frame_res(frame_config)

    # 选项卡一抽卡三倍区域
    def tab_card_frame_3s(self, frame):
        """三倍抽卡区域"""
        self.label1 = tk.Label(frame, text="是否三倍")
        self.label1.pack(side="top", anchor="n", padx=(20, 10))

        checkboxes = [
            "天使",
            "舞姬",
            "铁娘子",
            "王子",
            "黑寡妇",
            "德鲁伊",
            "敖丙",
            "酒神",
            "圣骑",
            "李白",
            "卡卡西",
        ]
        self.checkbox_vars = [tk.BooleanVar() for _ in range(len(checkboxes))]
        self.check_list = []
        for i, (text, var) in enumerate(zip(checkboxes, self.checkbox_vars)):
            chk = tk.Checkbutton(frame, text=text, variable=var)
            chk.pack(side="top", anchor="n" if i == 0 else "n", pady=0)  # 使第一个复选框居中，后面的也居中
            self.check_list.append(chk)

    # 选项卡一抽卡结果区域
    def tab_card_frame_res(self, frame):
        """抽卡结果区域ui"""

        # frame1 = ttk.Frame(frame)
        # frame1.pack(side="top", anchor="n", fill="x", padx=5)
        self.label2 = tk.Label(frame, text="抽卡次数")
        self.label2.grid(row=0, column=0, padx=(30, 0), pady=5)
        self.entry = tk.Entry(frame, width=5)
        self.entry.grid(row=0, column=1, pady=5)
        self.entry.bind("<Key>", self.allow_only_digits)
        self.entry.bind("<BackSpace>", self.on_backspace)
        # 设置初始值
        self.entry.insert(0, "0")
        self.label3 = tk.Label(frame, text="(0为无限次)")
        self.label3.grid(row=0, column=2, pady=5)

        # frame2 = ttk.Frame(frame)
        # frame2.pack(fill="x", padx=13)
        self.check_h_stop_var = tk.BooleanVar()
        self.check_h_stop = tk.Checkbutton(frame, text="遇到红色停止", variable=self.check_h_stop_var)
        self.check_h_stop.grid(row=1, column=0, columnspan=2, pady=5)

        # 第三区域内容
        self.label3 = tk.Label(frame, text="抽卡结果")
        # 字体加粗
        self.label3.configure(font=("宋体", 12, "bold"))
        self.label3.grid(row=2, column=1, padx=(10, 0), pady=(20, 10))

        # 第三区域第二行
        # frame1 = ttk.Frame(frame)
        # frame1.pack(side="top", anchor="n", padx=10, pady=5)
        self.label_show["天使"] = tk.Label(frame, text="天使 0")
        self.label_show["天使"].grid(row=3, column=0, padx=(20, 0), pady=5)
        self.label_count["天使"] = 0

        self.label_show["舞姬"] = tk.Label(frame, text="舞姬 0")
        self.label_show["舞姬"].grid(row=3, column=1, pady=2)
        self.label_count["舞姬"] = 0

        self.label_show["铁娘子"] = tk.Label(frame, text="铁娘子 0")
        self.label_show["铁娘子"].grid(row=3, column=2, pady=2)
        self.label_count["铁娘子"] = 0

        self.label_show["王子"] = tk.Label(frame, text="王子 0")
        self.label_show["王子"].grid(row=4, column=0, padx=(20, 0), pady=2)
        self.label_count["王子"] = 0

        self.label_show["黑寡妇"] = tk.Label(frame, text="黑寡妇 0")
        self.label_show["黑寡妇"].grid(row=4, column=1, pady=2)
        self.label_count["黑寡妇"] = 0
        self.label_show["德鲁伊"] = tk.Label(frame, text="德鲁伊 0")
        self.label_show["德鲁伊"].grid(row=4, column=2, pady=2)
        self.label_count["德鲁伊"] = 0

        # frame2 = ttk.Frame(frame)
        # frame2.pack(side="top", anchor="n", padx=10, pady=(10, 0))
        self.label_show["敖丙"] = tk.Label(frame, text="敖丙 0")
        self.label_show["敖丙"].grid(row=5, column=0, padx=(20, 0), pady=2)
        self.label_count["敖丙"] = 0

        self.label_show["酒神"] = tk.Label(frame, text="酒神 0")
        self.label_show["酒神"].grid(row=5, column=1, pady=2)
        self.label_count["酒神"] = 0

        # frame3 = ttk.Frame(frame)
        # frame3.pack(side="top", anchor="n", padx=10, pady=(0, 5))
        self.label_show["圣骑"] = tk.Label(frame, text="圣骑 0")
        self.label_show["圣骑"].grid(row=5, column=2, pady=2)
        self.label_count["圣骑"] = 0
        self.label_show["李白"] = tk.Label(frame, text="李白 0")
        self.label_show["李白"].grid(row=6, column=0, padx=(20, 0), pady=2)
        self.label_count["李白"] = 0

        self.label_show["卡卡西"] = tk.Label(frame, text="卡卡西 0")
        self.label_show["卡卡西"].grid(row=6, column=1, pady=2)
        self.label_count["卡卡西"] = 0

        # frame4 = ttk.Frame(frame)
        # frame4.pack(side="top", anchor="n", padx=10, pady=(10, 5))
        self.label_show["总次数"] = tk.Label(frame, text="总次数 0")
        self.label_show["总次数"].grid(row=7, column=1, pady=(20, 2))
        self.label_count["总次数"] = 0
        self.label_show["总消耗"] = tk.Label(frame, text="总消耗 0")
        self.label_show["总消耗"].grid(row=7, column=2, pady=(20, 2))
        self.label_count["总消耗"] = 0

    # 选项卡二 （刷金）
    def create_tab_coin(self):
        """选项卡二的总体框架"""
        top_half = ttk.Frame(self.tab_coin)
        top_half.pack(side="top", fill="x")

        # 第一个区域
        default = ttk.Frame(top_half, borderwidth=2, relief="groove")
        default.pack(side="right", anchor="n", fill="both", expand="true", padx=2, pady=2)
        self.tab_coin_default(default)

        # 第二区域
        custom = ttk.Frame(top_half, borderwidth=2, relief="groove")
        custom.pack(side="left", anchor="n", fill="both", expand="true", padx=2, pady=2)
        self.tab_coin_custom(custom)

    # 选项卡二（默认模式）
    def tab_coin_default(self, frame):
        """刷金区域ui"""
        self.rad_mode_var = tk.IntVar(value=2)
        self.rad_default = tk.Radiobutton(frame, text="默认模式", variable=self.rad_mode_var, value=2)
        self.rad_default.pack(side="top", anchor="n", padx=2, pady=0)
        self.rad_default.config(command=self.radio_changed)

        self.frame_default = ttk.Frame(frame)
        self.frame_default.pack(side="top", anchor="n", padx=2, pady=0)

        checkboxes = ["树精领主", "树精长老", "火焰石像", "疯牛魔王", "剧毒蝎王"]

        self.boss_chk_vars = [tk.BooleanVar() for _ in range(len(checkboxes))]
        self.boss_chk_list = []
        self.boss_entry_list = []

        for i, (text, var) in enumerate(zip(checkboxes, self.boss_chk_vars)):
            # print(self.config["coins"][text]["enable"])
            chk = tk.Checkbutton(self.frame_default, text=text, variable=var)
            chk.grid(row=i, column=0, pady=3)
            label_boss = tk.Label(self.frame_default, text="击杀时间(秒):")
            label_boss.grid(row=i, column=1, pady=3)
            entry_boss = tk.Entry(self.frame_default, width=5)
            entry_boss.insert(0, "1.5")
            entry_boss.grid(row=i, column=2, pady=3)
            entry_boss.bind("<Key>", self.allow_only_digits)
            entry_boss.bind("<BackSpace>", self.on_backspace)
            self.boss_entry_list.append(entry_boss)
            self.boss_chk_list.append(chk)

        lal_ref_mode = tk.Label(self.frame_default, text="刷新模式")
        lal_ref_mode.grid(row=len(checkboxes) + 1, column=0, pady=5)

        self.ref_mode = tk.IntVar(value=1)
        radio_ref1 = tk.Radiobutton(self.frame_default, text="王座刷新", variable=self.ref_mode, value=1)
        radio_ref1.grid(row=len(checkboxes) + 1, column=1, pady=5)

        radio_ref2 = tk.Radiobutton(self.frame_default, text="重启刷新", variable=self.ref_mode, value=2)
        radio_ref2.grid(row=len(checkboxes) + 1, column=2, pady=5)

        lal_hc = tk.Label(self.frame_default, text="回城时间")
        lal_hc.grid(row=len(checkboxes) + 2, column=0, pady=0)
        self.hc_delay = tk.Entry(self.frame_default, width=5)
        self.hc_delay.insert(0, "5")
        self.hc_delay.grid(row=len(checkboxes) + 3, column=0, pady=0)

        lal_cs = tk.Label(self.frame_default, text="传送时间")
        lal_cs.grid(row=len(checkboxes) + 2, column=1, pady=0)
        self.cs_delay = tk.Entry(self.frame_default, width=5)
        self.cs_delay.insert(0, "6")
        self.cs_delay.grid(row=len(checkboxes) + 3, column=1, pady=0)

        lal_cq = tk.Label(self.frame_default, text="重启时间")
        lal_cq.grid(row=len(checkboxes) + 2, column=2, pady=0)
        self.cq_delay = tk.Entry(self.frame_default, width=5)
        self.cq_delay.insert(0, "12")
        self.cq_delay.grid(row=len(checkboxes) + 3, column=2, pady=0)

    # 选项卡二（自定义指令）
    def tab_coin_custom(self, frame):
        self.rad_custom = tk.Radiobutton(frame, text="自定义刷图", variable=self.rad_mode_var, value=1)
        self.rad_custom.pack(side="top", anchor="n", padx=2, pady=0)
        self.rad_custom.config(command=self.radio_changed)

        self.frame_custom = ttk.Frame(frame)
        self.frame_custom.pack(side="top", anchor="n", padx=2, pady=0)

        self.com_text = tk.Text(self.frame_custom, width=40, height=20)
        self.com_text.pack(side="left", anchor="n", fill="both", expand="True", padx=2, pady=2)
        self.com_text_insert(self.com_text)

    # 指令输入框初始化提示
    def com_text_insert(self, com_text):
        com_text.insert("end", "#         注意\n")
        com_text.insert("end", "#  1.指令每行一条\n")
        com_text.insert("end", "#  2.#和//开头的不执行 \n")
        com_text.insert("end", "#  3.传送需确保在传送旁边\n")
        com_text.insert("end", "\n")
        com_text.insert("end", "\n")
        com_text.insert("end", "# 传送(章节,地图,时间)\n")
        com_text.insert("end", "# 其它括号中的数字均为时间\n")
        com_text.insert("end", "# 单位:秒 支持小数\n")
        com_text.insert("end", "\n")
        com_text.insert("end", "# 例：\n")
        com_text.insert("end", "\n")
        com_text.insert("end", "传送(2, 1, 6)\n")
        com_text.insert("end", "移动(右, 1)\n")
        com_text.insert("end", "移动(右上, 1)\n")
        com_text.insert("end", "等待(2)\n")
        com_text.insert("end", "回城(6)\n")
        com_text.insert("end", "重启(12)\n")

    # 允许输入数字
    def allow_only_digits(self, event):
        # 如果输入的字符不是数字，就阻止它
        if event.char and not event.char.isdigit():
            return "break"

    # 允许退格键
    def on_backspace(self, event):
        # 允许退格键
        return None

    # 选项卡切换事件
    def on_tab_change(self, event):
        current_tab_index = self.tab_control.index(self.tab_control.select())
        self.current_tab = current_tab_index
        # print("当前选项卡索引:", current_tab_index)

    # 模式切换
    def radio_changed(self):
        if self.rad_mode_var.get() == 1:
            for combo in self.frame_default.children.values():
                combo.config(state="disabled")
            for combo in self.frame_custom.children.values():
                combo.config(state="normal")
        else:
            for combo in self.frame_custom.children.values():
                combo.config(state="disabled")
            for combo in self.frame_default.children.values():
                combo.config(state="normal")

    # 更新抽卡结果
    def update_frame_res(self, data):
        """更新抽卡结果"""
        for key, value in data.items():
            self.label_count[key] += value
            self.label_show[key].config(text=key + " " + str(self.label_count[key]))

    # 日志输出
    def log_text(self, msg):
        now_time = time.strftime("%m-%d %H:%M:%S", time.localtime())
        self.text.config(state="normal")
        self.text.insert("end", now_time + ": " + msg + "\n")
        self.text.config(state="disabled")
        self.text.see("end")

    # 初始化抽卡结果
    def init_frame_res(self):
        """初始化抽卡结果"""
        for key in self.label_show.keys():
            self.label_count[key] = 0
            self.label_show[key].config(text=key + " 0")

    # 获取队列中的数据
    def get_queue_data(self):
        """循环获取队列中的数据"""
        # while not self.task_queue.empty():
        # task_data = self.task_queue.get(block=True)
        while True:
            try:
                task_data = self.task_queue.get_nowait()
                # print(task_data["type"])
                if task_data["type"] == "log":
                    self.log_text(task_data["msg"])
                    continue
                if task_data["type"] == "update":
                    self.update_frame_res(task_data["data"])
                    continue
                if task_data["type"] == "error":
                    self.show_error(task_data["msg"])
                    continue
                if task_data["type"] == "stop":
                    self.stop_action()
            except Exception as e:
                break
        if self.running:
            self.root.after(100, self.get_queue_data)

    # 启动程序执行的脚本
    def start_scripts(self):
        """启动时执行的部分代码"""
        self.running = True
        self.start_time = time.time()
        self.start_timer()
        self.button_run["text"] = "停止"
        self.button_run["command"] = self.stop_action
        self.save_config()
        self.init_frame_res()

    # 停止程序执行的脚本
    def stop_scripts(self):
        """停止时执行的部分代码"""
        self.running = False
        self.label_timer.config(text="")
        self.button_run["text"] = "运行"
        self.button_run["command"] = self.start_action

    # 启动计时器
    def start_timer(self):
        """启动计时器"""
        if self.running:
            hours, rem = divmod(time.time() - self.start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            self.label_timer.config(text=f"已运行：{int(hours):02}:{int(minutes):02}:{int(seconds):02}", fg="green")
            self.root.after(1000, self.start_timer)

    # 开始按钮点击事件
    def start_action(self):
        """开始按钮点击事件"""
        if self.button_run["text"] == "运行":
            self.log_text("开始执行")
            self.start_scripts()
            self.get_queue_data()
            # 这里添加执行逻辑
            model = Model(self.task_queue, self.config_data)
            if self.current_tab == 0:
                self.p_running = multiprocessing.Process(target=model.run_cards)
            elif self.current_tab == 1:
                self.p_running = multiprocessing.Process(target=model.run_coins)
            self.p_running.start()
        else:
            self.stop_action()

    # 停止按钮点击事件
    def stop_action(self):
        """停止按钮点击事件"""
        self.stop_scripts()
        self.log_text("已停止")
        if self.p_running is not None and self.p_running.is_alive():
            self.p_running.terminate()  # 终止进程

    # 窗口关闭事件
    def on_closing(self):
        """程序退出时执行"""
        if self.p_running is not None and self.p_running.is_alive():
            # self.task_queue.put(True)  # 发送停止信号 目前不需要向model发送停止信息，不需要model控制关闭进程
            # join(timeout=5)：等待子进程结束，最多等待5秒钟。如果在这段时间内子进程结束，则 join() 方法返回，程序继续执行；如果5秒钟后子进程仍未结束，则 join() 方法超时返回，程序继续执行下一步。
            # terminate()：如果 join() 方法超时返回，并且子进程仍然存活，则调用 terminate() 方法立即终止子进程。这通常用于强制结束那些不响应停止信号的子进程。
            # 通过这种方式，可以确保在窗口关闭时尽量优雅地结束子进程，并且在必要时强制终止它们，从而避免遗留问题。
            self.p_running.join(timeout=1)
            if self.p_running.is_alive():
                self.p_running.terminate()
            print("exit")
        time.sleep(0.05)
        self.release_all()
        self.root.destroy()

    # 释放所有资源
    def release_all(self):
        filename = "error.log"
        current_directory = os.getcwd()
        file_path = os.path.join(current_directory, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                pass

    # 窗口点击事件
    def on_click(self, event):
        event.widget.focus_set()

    # def on_click(self, event):
    #     """点击事件处理函数"""
    #     # 检查点击事件是否发生在 Entry 组件之外
    #     entry_list = [self.entry] + self.boss_entry_list
    #     if event.widget != self.entry and event.widget not in self.boss_entry_list:
    #         # self.root.focus_force()  # 强制让 Entry 组件失去焦点
    #         self.text.focus_set()
    #         try:
    #             cards_count = int(self.entry.get())
    #             self.entry.delete(0, tk.END)
    #             self.entry.insert(0, cards_count)
    #         except ValueError:
    #             self.entry.delete(0, tk.END)
    #             self.entry.insert(0, "0")

    # 读取配置文件
    def get_config(self) -> configparser.ConfigParser:
        """读取配置文件"""
        config = configparser.ConfigParser()
        if os.path.exists("config.ini"):
            try:
                # 抽卡配置
                config.read("config.ini", encoding="utf-8")
                for i, item in enumerate(self.check_list):
                    self.checkbox_vars[i].set(config["cards"][item["text"]])
                self.entry.delete(0, tk.END)
                self.entry.insert(0, config["config"]["cards_count"])
                self.check_h_stop_var.set(config["config"]["h_stop"])

                # 刷金配置
                for i, item in enumerate(self.boss_chk_list):
                    self.boss_chk_vars[i].set(config["coins"][item["text"] + "enable"])
                    self.boss_entry_list[i].delete(0, tk.END)
                    self.boss_entry_list[i].insert(0, config["coins"][item["text"] + "delay"])
                self.ref_mode.set(config["coins"]["refresh_mode"])
                self.rad_mode_var.set(config["coins"]["coins_mode"])

                # 刷金时间配置
                if config.has_option("coins", "hc_delay"):
                    self.hc_delay.delete(0, tk.END)
                    self.hc_delay.insert(0, config["coins"]["hc_delay"])

                if config.has_option("coins", "cs_delay"):
                    self.cs_delay.delete(0, tk.END)
                    self.cs_delay.insert(0, config["coins"]["cs_delay"])

                if config.has_option("coins", "cq_delay"):
                    self.cq_delay.delete(0, tk.END)
                    self.cq_delay.insert(0, config["coins"]["cq_delay"])

                # UI禁用
                if self.rad_mode_var.get() == 1:
                    for combo in self.frame_default.children.values():
                        combo.config(state="disabled")
                    for combo in self.frame_custom.children.values():
                        combo.config(state="normal")
                else:
                    for combo in self.frame_custom.children.values():
                        combo.config(state="disabled")
                    for combo in self.frame_default.children.values():
                        combo.config(state="normal")

                # 自定义配置
                with open("script.json", "r", encoding="utf-8") as f:
                    content = f.read()

                if content.strip():
                    # 文件不为空，加载JSON数据
                    # print("读取配置文件成功")
                    script_list = json.loads(content)
                    self.com_text.delete(1.0, tk.END)
                    for line in script_list:
                        self.com_text.insert(tk.END, line + "\n")

            except Exception as e:
                # print(e)
                pass
        return config

    # 保存配置文件
    def save_config(self):
        """保存配置文件"""
        config = configparser.ConfigParser()
        config_cards = {}
        config_coins = {}
        config_custom = []

        # 保存三倍配置
        for i, item in enumerate(self.check_list):
            config_cards[item["text"]] = self.checkbox_vars[i].get()
        config["cards"] = config_cards

        # 抽卡配置
        try:
            cards_count = int(self.entry.get())
        except ValueError:
            cards_count = 0

        config["config"] = {
            "cards_count": cards_count,
            "h_stop": self.check_h_stop_var.get(),
        }

        # 刷金配置
        for i, item in enumerate(self.boss_chk_list):
            config_coins[item["text"] + "enable"] = self.boss_chk_vars[i].get()
            config_coins[item["text"] + "delay"] = float(
                self.boss_entry_list[i].get() if self.boss_entry_list[i].get() != "" else 0
            )

        config_coins["refresh_mode"] = self.ref_mode.get()
        config_coins["coins_mode"] = self.rad_mode_var.get()

        # 刷金默认时间配置
        try:
            hc_delay = float(self.hc_delay.get())
        except:
            hc_delay = 0
        try:
            cs_delay = float(self.cs_delay.get())
        except:
            cs_delay = 0
        try:
            cq_delay = float(self.cq_delay.get())
        except:
            cq_delay = 0

        config_coins["hc_delay"] = hc_delay
        config_coins["cs_delay"] = cs_delay
        config_coins["cq_delay"] = cq_delay

        # 刷金自定义配置
        for line in self.com_text.get("1.0", "end-1c").splitlines():
            if line.strip() == "":
                continue
            line = line.replace("（", "(").replace("）", ")").replace("，", ",").strip()
            config_custom.append(line)

        # 默认模式刷金配置
        config["coins"] = config_coins

        # 保存到配置文件
        with open("config.ini", "w", encoding="utf-8") as config_file:
            config.write(config_file)

        # 保存自定义配置到json文件
        with open("script.json", "w", encoding="utf-8") as f:
            json.dump(config_custom, f, ensure_ascii=False)

        # 用于传给model
        self.config_data["cards"] = config_cards
        self.config_data["cards"].update(config["config"])

        self.config_data["coins"] = config_coins
        self.config_data["script"] = config_custom

        # print(self.config_data)

    # 显示错误消息框
    def show_error(self, msg):
        """显示错误消息框"""
        self.stop_action()
        messagebox.showerror(title="错误", message=msg)


# if __name__ == "__main__":
#     root = tk.Tk()
#     app = App(root)
#     root.mainloop()

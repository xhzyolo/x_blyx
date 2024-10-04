import tkinter as tk
import multiprocessing
import time
import os, sys
import configparser
import requests
from tkinter import ttk
from tkinter import messagebox
from model import Model


VERSION = "0.3"
BASE_URL = "https://xzdwz.com/api/blyx/blyx"


class App:
    def __init__(self, root):
        self.root = root
        self.running = False
        self.label_show = {}
        self.label_count = {}
        self.p_cards = None
        title = f"x-blyx   V: {VERSION}"
        self.__create_window(title, 400, 500)
        self.__create_tabs()
        self.root.iconbitmap(self.get_resource_path("images/blyx.ico"))
        self.config = self.get_config()

        self.task_queue = multiprocessing.Queue()
        # 注册窗口关闭处理函数
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.check_version()

    def check_version(self):
        try:
            self.log_text("检查版本...")
            res = requests.post(BASE_URL, data={"action": "get_version"}, timeout=3).json()
            if res["errorCode"] == 0:
                min_version = res["data"]["min_version"]
                max_version = res["data"]["max_version"]
                if float(VERSION) >= float(min_version):
                    if float(VERSION) == float(max_version):
                        self.log_text("当前为最新版本")
                        return
                    if float(VERSION) < float(max_version):
                        self.log_text(f"发现新版本,v{max_version}")
                        return
                else:
                    messagebox.showinfo(title="提示", message="当前版本过低，请更新到最新版本")
            else:
                messagebox.showerror(title="错误", message="版本获取失败，请重试")
        except requests.exceptions.Timeout:
            messagebox.showerror(title="错误", message="服务器错误，请稍后重试")
        self.root.destroy()

    def get_resource_path(self, relative_path):
        """获取资源文件的绝对路径"""
        if hasattr(sys, "_MEIPASS"):
            # PyInstaller 创建的临时目录
            base_path = sys._MEIPASS
        else:
            # 开发环境中的当前目录
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def __create_window(self, title: str, w_width: int, w_height: int):
        """创建窗口"""
        self.root.resizable(False, False)
        self.root.title(title)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        pos_x = (screen_width - w_width) // 2
        pos_y = (screen_height - w_height) // 2
        self.root.geometry(f"{w_width}x{w_height}+{pos_x}+{pos_y}")
        self.root.bind("<Button-1>", self.on_click)

    def __create_tabs(self):
        """创建选项卡"""
        self.tab_control = ttk.Notebook(self.root)
        self.tab_card = ttk.Frame(self.tab_control)
        self.tab_coin = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_card, text="抽卡")
        self.tab_control.add(self.tab_coin, text="金币")
        self.tab_control.pack(expand=1, fill="both")

        # 选项卡一的内容
        self.create_tab_card()

        # 选项卡二的内容
        # self.create_tab_coin()

    def create_tab_card(self):
        """选项卡一的总体框架"""
        top_half = ttk.Frame(self.tab_card)
        top_half.pack(side="top", fill="x")

        bottom_half = ttk.Frame(self.tab_card)
        bottom_half.pack(side="bottom", fill="both", expand="true")

        # 第一个区域
        frame_3s = ttk.Frame(top_half, borderwidth=2, relief="groove")
        frame_3s.pack(side="left", anchor="n", fill="both", expand="true", padx=5, pady=5)
        self.tab_card_frame_3s(frame_3s)

        # 第二区域
        frame_config = ttk.Frame(top_half, borderwidth=2, relief="groove")
        frame_config.pack(side="right", anchor="n", fill="both", expand="true", padx=5, pady=5)
        self.tab_card_frame_config(frame_config)

        # 第三区域
        self.frame_res = ttk.Frame(bottom_half, borderwidth=2, relief="groove")
        # self.frame_res.pack(side="bottom", anchor="n", fill="both", expand="true", padx=5, pady=5)
        self.tab_card_frame_res(self.frame_res)
        self.frame_res.pack(side="bottom", anchor="n", fill="both", expand="true", padx=5, pady=5)

    def tab_card_frame_3s(self, frame):
        """三倍抽卡区域"""
        self.label1 = tk.Label(frame, text="是否三倍")
        self.label1.pack(side="top", anchor="n", padx=(20, 10))

        checkboxes = [
            "天使",
            "舞姬",
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
            try:
                chk = tk.Checkbutton(frame, text=text, variable=self.config["cards"][text])
            except Exception:  # 如果在配置文件中没有找到对应的键，则创建一个默认的变量
                chk = tk.Checkbutton(frame, text=text, variable=var)
            chk.pack(side="top", anchor="n" if i == 0 else "n")  # 使第一个复选框居中，后面的也居中
            self.check_list.append(chk)

    def tab_card_frame_config(self, frame):
        """抽卡配置区域"""
        frame1 = ttk.Frame(frame)
        frame1.pack(side="top", anchor="n", fill="x", padx=5)
        self.label2 = tk.Label(frame1, text="抽卡次数")
        self.label2.pack(side="left", anchor="n", padx=(10, 3))
        self.entry = tk.Entry(frame1, width=5)
        self.entry.pack(side="left", anchor="n", padx=(0, 10))
        # 设置初始值
        try:
            self.entry.insert(0, self.config["config"]["card_num"])
        except Exception:
            self.entry.insert(0, "0")
        self.label3 = tk.Label(frame1, text="(0为无限次)")
        self.label3.pack(side="left", anchor="n", padx=(0, 3))

        frame2 = ttk.Frame(frame)
        frame2.pack(fill="x", padx=13)
        self.check_h_stop_var = tk.BooleanVar()
        self.check_h_stop = tk.Checkbutton(frame2, text="遇到红色停止", variable=self.check_h_stop_var)
        self.check_h_stop.pack(side="left")

        # 日志
        self.text = tk.Text(frame, width=30, height=10)
        # self.text.pack(side="top", anchor="n", fill="both", expand="true", padx=5, pady=5)
        self.text.pack(side="bottom", anchor="n", fill="both", padx=5, pady=5)
        self.log_text("载入成功!")

        # 运行时间
        self.label_timer = tk.Label(frame, text="")
        self.label_timer.pack(side="left", anchor="s", padx=(20, 0), pady=(0, 14))

        # 运行按钮
        self.button_run = tk.Button(frame, text="运行", command=self.start_action)
        self.button_run.config(width=8, relief="solid")
        self.button_run.pack(side="right", anchor="s", padx=(0, 10), pady=10)

    def tab_card_frame_res(self, frame):
        """抽卡结果区域ui"""
        # 第三区域内容
        self.label3 = tk.Label(frame, text="抽卡结果")
        self.label3.pack(side="top", anchor="n", padx=(20, 20))

        # 第三区域第二行
        frame1 = ttk.Frame(frame)
        frame1.pack(side="top", anchor="n", padx=10, pady=5)
        self.label_show["天使"] = tk.Label(frame1, text="天使 0")
        self.label_show["天使"].pack(side="left", anchor="n", padx=10)
        self.label_count["天使"] = 0
        self.label_show["舞姬"] = tk.Label(frame1, text="舞姬 0")
        self.label_show["舞姬"].pack(side="left", anchor="n", padx=10)
        self.label_count["舞姬"] = 0
        self.label_show["王子"] = tk.Label(frame1, text="王子 0")
        self.label_show["王子"].pack(side="left", anchor="n", padx=10)
        self.label_count["王子"] = 0
        self.label_show["黑寡妇"] = tk.Label(frame1, text="黑寡妇 0")
        self.label_show["黑寡妇"].pack(side="left", anchor="n", padx=10)
        self.label_count["黑寡妇"] = 0
        self.label_show["德鲁伊"] = tk.Label(frame1, text="德鲁伊 0")
        self.label_show["德鲁伊"].pack(side="left", anchor="n", padx=10)
        self.label_count["德鲁伊"] = 0

        frame2 = ttk.Frame(frame)
        frame2.pack(side="top", anchor="n", padx=10, pady=(10, 0))
        self.label_show["敖丙"] = tk.Label(frame2, text="敖丙 0")
        self.label_show["敖丙"].pack(side="left", anchor="n", padx=10)
        self.label_count["敖丙"] = 0
        self.label_show["酒神"] = tk.Label(frame2, text="酒神 0")
        self.label_show["酒神"].pack(side="left", anchor="n", padx=10)
        self.label_count["酒神"] = 0

        # frame3 = ttk.Frame(frame)
        # frame3.pack(side="top", anchor="n", padx=10, pady=(0, 5))
        self.label_show["圣骑"] = tk.Label(frame2, text="圣骑 0")
        self.label_show["圣骑"].pack(side="left", anchor="n", padx=10)
        self.label_count["圣骑"] = 0
        self.label_show["李白"] = tk.Label(frame2, text="李白 0")
        self.label_show["李白"].pack(side="left", anchor="n", padx=10)
        self.label_count["李白"] = 0
        self.label_show["卡卡西"] = tk.Label(frame2, text="卡卡西 0")
        self.label_show["卡卡西"].pack(side="left", anchor="n", padx=10)
        self.label_count["卡卡西"] = 0

        frame4 = ttk.Frame(frame)
        frame4.pack(side="top", anchor="n", padx=10, pady=(10, 5))
        self.label_show["总次数"] = tk.Label(frame4, text="总次数 0")
        self.label_show["总次数"].pack(side="left", anchor="n", padx=10, pady=(10, 0))
        self.label_count["总次数"] = 0
        self.label_show["总消耗"] = tk.Label(frame4, text="总消耗 0")
        self.label_show["总消耗"].pack(side="left", anchor="n", padx=10, pady=(10, 0))
        self.label_count["总消耗"] = 0

    def update_frame_res(self, data):
        """更新抽卡结果"""
        for key, value in data.items():
            self.label_count[key] += value
            self.label_show[key].config(text=key + " " + str(self.label_count[key]))

    def log_text(self, msg):
        now_time = time.strftime("%m-%d %H:%M:%S", time.localtime())
        self.text.config(state="normal")
        self.text.insert("end", now_time + ": " + msg + "\n")
        self.text.config(state="disabled")
        self.text.see("end")

    def init_frame_res(self):
        """初始化抽卡结果"""
        for key in self.label_show.keys():
            self.label_count[key] = 0
            self.label_show[key].config(text=key + " 0")

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

    def start_scripts(self):
        """启动时执行的部分代码"""
        self.running = True
        self.start_time = time.time()
        self.start_timer()
        self.button_run["text"] = "停止"
        self.button_run["command"] = self.stop_action
        self.save_config()
        self.init_frame_res()

    def stop_scripts(self):
        """停止时执行的部分代码"""
        self.running = False
        self.label_timer.config(text="")
        self.button_run["text"] = "运行"
        self.button_run["command"] = self.start_action

    def start_timer(self):
        """启动计时器"""
        if self.running:
            hours, rem = divmod(time.time() - self.start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            self.label_timer.config(text=f"已运行：{int(hours):02}:{int(minutes):02}:{int(seconds):02}", fg="green")
            self.root.after(1000, self.start_timer)

    def start_action(self):
        """开始按钮点击事件"""
        if self.button_run["text"] == "运行":
            self.log_text("开始执行")
            self.start_scripts()
            self.get_queue_data()
            # 这里添加执行逻辑
            model = Model(self.task_queue, self.config_data)
            self.p_cards = multiprocessing.Process(target=model.run_cards)
            self.p_cards.start()
        else:
            self.stop_action()

    def stop_action(self):
        """停止按钮点击事件"""
        self.stop_scripts()
        self.log_text("已停止")
        if self.p_cards is not None and self.p_cards.is_alive():
            self.p_cards.terminate()  # 终止进程

    def on_closing(self):
        """程序退出时执行"""
        if self.p_cards is not None and self.p_cards.is_alive():
            # self.task_queue.put(True)  # 发送停止信号 目前不需要向model发送停止信息，不需要model控制关闭进程
            # join(timeout=5)：等待子进程结束，最多等待5秒钟。如果在这段时间内子进程结束，则 join() 方法返回，程序继续执行；如果5秒钟后子进程仍未结束，则 join() 方法超时返回，程序继续执行下一步。
            # terminate()：如果 join() 方法超时返回，并且子进程仍然存活，则调用 terminate() 方法立即终止子进程。这通常用于强制结束那些不响应停止信号的子进程。
            # 通过这种方式，可以确保在窗口关闭时尽量优雅地结束子进程，并且在必要时强制终止它们，从而避免遗留问题。
            self.p_cards.join(timeout=1)
            if self.p_cards.is_alive():
                self.p_cards.terminate()
            print("exit")
        time.sleep(0.05)
        self.root.destroy()

    def on_click(self, event):
        """点击事件处理函数"""
        # 检查点击事件是否发生在 Entry 组件之外
        if event.widget != self.entry:
            # self.root.focus_force()  # 强制让 Entry 组件失去焦点
            self.text.focus_set()
            try:
                cards_count = int(self.entry.get())
                self.entry.delete(0, tk.END)
                self.entry.insert(0, cards_count)
            except ValueError:
                self.entry.delete(0, tk.END)
                self.entry.insert(0, "0")

    def get_config(self) -> configparser.ConfigParser:
        """读取配置文件"""
        config = configparser.ConfigParser()
        if os.path.exists("config.ini"):
            try:
                config.read("config.ini", encoding="utf-8")
                for i, item in enumerate(self.check_list):
                    self.checkbox_vars[i].set(config["cards"][item["text"]])

                self.entry.delete(0, tk.END)
                self.entry.insert(0, config["config"]["cards_count"])
                self.check_h_stop_var.set(config["config"]["h_stop"])
            except:
                pass
        return config

    def save_config(self):
        """保存配置文件"""
        config = configparser.ConfigParser()
        config_cards = {}

        for i, item in enumerate(self.check_list):
            config_cards[item["text"]] = self.checkbox_vars[i].get()
        config["cards"] = config_cards

        config["config"] = {"cards_count": self.entry.get(), "h_stop": self.check_h_stop_var.get()}

        with open("config.ini", "w", encoding="utf-8") as config_file:
            config.write(config_file)

        # 用于传给model
        self.config_data = config_cards
        self.config_data.update(config["config"])

    def show_error(self, msg):
        """显示错误消息框"""
        self.stop_action()
        messagebox.showerror(title="错误", message=msg)


# if __name__ == "__main__":
#     root = tk.Tk()
#     app = App(root)
#     root.mainloop()

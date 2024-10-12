import tkinter as tk
import multiprocessing
from view import App


VERSION = "0.5"

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = App(root, VERSION)
    root.mainloop()
"""
     指令包括：
传送(3,1)
移动(左上,1.5)
等待(5.3)
回城()
重启()


      注意

1.指令每行一条

2.使用传送时，需确
  保在传送旁边

3.使用传送，回城，
  重启时，请自行 
  加入等待指令，并
  设置合适的时间
"""

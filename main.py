import tkinter as tk
import multiprocessing
from view import App


END_TIME = "2099-12-31 23:59:59"

if __name__ == "__main__":
    try:
        multiprocessing.freeze_support()
        root = tk.Tk()
        app = App(root)
        root.mainloop()
    except Exception as e:
        print(e)

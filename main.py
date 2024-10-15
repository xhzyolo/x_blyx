import tkinter as tk
import multiprocessing
from view import App


VERSION = "0.6"

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = App(root, VERSION)
    root.mainloop()

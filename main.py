import tkinter as tk
import multiprocessing
from view import App


VERSION = "1.11"

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = App(root, VERSION)
    root.mainloop()

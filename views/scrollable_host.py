"""Reusable scrolling for forms without global mouse-wheel interception."""
import tkinter as tk
from tkinter import ttk


class ScrollableHost(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, highlightthickness=0, background="#ffffff")
        self.canvas.grid(row=0,column=0,sticky="nsew")
        vertical = ttk.Scrollbar(self,orient="vertical",command=self.canvas.yview)
        vertical.grid(row=0,column=1,sticky="ns")
        horizontal = ttk.Scrollbar(self,orient="horizontal",command=self.canvas.xview)
        horizontal.grid(row=1,column=0,sticky="ew")
        self.canvas.configure(yscrollcommand=vertical.set,xscrollcommand=horizontal.set)
        self._tag = f"ScrollHost{id(self)}"
        self.bind_class(self._tag,"<MouseWheel>",self._wheel)
        self.bind("<Destroy>",self._destroy,add="+")

    def mount(self, view):
        self.view = view
        self._window = self.canvas.create_window((0,0),window=view,anchor="nw")
        view.bind("<Configure>",self._resize,add="+")
        self.canvas.bind("<Configure>",self._resize,add="+")
        self._bind_after = self.after_idle(self._bind_descendants)

    def _bind_descendants(self):
        self._bind_after = None
        def visit(widget):
            tags=widget.bindtags()
            if self._tag not in tags:
                widget.bindtags(tags[:2]+(self._tag,)+tags[2:])
            for child in widget.winfo_children(): visit(child)
        visit(self.view)

    def _resize(self, event=None):
        self.canvas.itemconfigure(self._window,width=max(self.canvas.winfo_width(),self.view.winfo_reqwidth()))
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _wheel(self,event):
        if self.canvas.yview() != (0.,1.):
            self.canvas.yview_scroll(-int(event.delta/120) or (-1 if event.delta>0 else 1),"units")
            return "break"

    def _destroy(self,event):
        if event.widget is self:
            if getattr(self,"_bind_after",None):
                self.after_cancel(self._bind_after)
            self.unbind_class(self._tag,"<MouseWheel>")

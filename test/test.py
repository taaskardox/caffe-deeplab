from Tkinter import Tk, Frame, LEFT, BOTH, RIGHT


class Window(Frame):
 def __init__(self, parent):
  Frame.__init__(self, parent, background="white")
  self.parent = parent
  self.initUI()
  
 def initUI(self):
  self.parent.title("Test example")
  self.pack(fill=BOTH, expand=1)
  topframe = Frame(self)
  topframe.pack()
  frame1 = Frame(topframe, width=499, heigh = 500)
  frame1.pack(side=LEFT)
  #frame2 = Frame(topframe, width=499, heigh = 500)
  #frame2.pack(side=RIGHT)
  
root = Tk()
root.geometry("1000x500+300+300")
app = Window(root)
root.mainloop()


from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import *
import PIL
from face_morhper import main

root = Tk()
root.title('人脸融合软件')
root.geometry('1200x500')

decoration = PIL.Image.open('bg.png').resize((1200, 500))
render = ImageTk.PhotoImage(decoration)
img = Label(image=render)
img.image = render
img.place(x=0, y=0)

global path1_, path2_, rate, seg_img_path


# 原图1展示
def show_original1_pic():
    global path1_
    path1_ = askopenfilename(title='选择文件')
    print(path1_)
    Img = PIL.Image.open(r'{}'.format(path1_))
    Img = Img.resize((270,270),PIL.Image.ANTIALIAS)   # 调整图片大小至256x256
    img_png_original = ImageTk.PhotoImage(Img)
    label_Img_original1.config(image=img_png_original)
    label_Img_original1.image = img_png_original  # keep a reference
    cv_orinial1.create_image(5, 5,anchor='nw', image=img_png_original)


# 原图2展示
def show_original2_pic():
    global path2_
    path2_ = askopenfilename(title='选择文件')
    print(path2_)
    Img = PIL.Image.open(r'{}'.format(path2_))
    Img = Img.resize((270,270),PIL.Image.ANTIALIAS)   # 调整图片大小至256x256
    img_png_original = ImageTk.PhotoImage(Img)
    label_Img_original2.config(image=img_png_original)
    label_Img_original2.image = img_png_original  # keep a reference
    cv_orinial2.create_image(5, 5,anchor='nw', image=img_png_original)


# 人脸融合效果展示
def show_morpher_pic():
    global path1_,seg_img_path,path2_
    print(entry.get())
    mor_img_path = main(path1_,path2_,entry.get())
    Img = PIL.Image.open(r'{}'.format(mor_img_path))
    Img = Img.resize((270, 270), PIL.Image.ANTIALIAS)  # 调整图片大小至256x256
    img_png_seg = ImageTk.PhotoImage(Img)
    label_Img_seg.config(image=img_png_seg)
    label_Img_seg.image = img_png_seg  # keep a reference


def quit():
    root.destroy()


# 原图1的展示
Button(root, text = "打开图片1", command = show_original1_pic).place(x=50,y=120)
# 原图2的展示
Button(root, text = "打开图片2", command = show_original2_pic).place(x=50,y=200)
# 进行提取结果的展示
Button(root, text = "人脸融合", command = show_morpher_pic).place(x=50,y=280)

Button(root, text = "退出软件", command = quit).place(x=900,y=40)

Label(root,text = "融合系数",font=10).place(x=50,y=10)
entry = Entry(root)
entry.place(x=130,y=10)


Label(root,text = "图片1",font=10).place(x=280,y=120)
cv_orinial1 = Canvas(root,bg = 'white',width=270,height=270)
cv_orinial1.create_rectangle(8,8,260,260,width=1,outline='red')
cv_orinial1.place(x=180,y=150)
label_Img_original1 = Label(root)
label_Img_original1.place(x=180,y=150)


Label(root,text="图片2",font=10).place(x=600,y=120)
cv_orinial2 = Canvas(root,bg = 'white',width=270,height=270)
cv_orinial2.create_rectangle(8,8,260,260,width=1,outline='red')
cv_orinial2.place(x=500,y=150)
label_Img_original2 = Label(root)
label_Img_original2.place(x=500,y=150)

Label(root, text="融合效果", font=10).place(x=920,y=120)
cv_seg = Canvas(root, bg='white', width=270,height=270)
cv_seg.create_rectangle(8,8,260,260,width=1,outline='red')
cv_seg.place(x=820,y=150)
label_Img_seg = Label(root)
label_Img_seg.place(x=820,y=150)


root.mainloop()
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:34:39 2020

@author: Jinu
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
#load the trained model to classify sign

#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Fake profile  detection')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)
def classify(file_path):
    global label_packed
    print(file_path)
    #image = Image.open(file_path)
    f = open(file_path, "r")
   # print(f.read())
    sign1 = f.read()
    print(sign1)
    label.configure(foreground='#011638', text=sign1)
    f.close()
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify data",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=open(file_path, "r")
        
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass
upload=Button(top,text="Upload user data",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Fake profile  Detection",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
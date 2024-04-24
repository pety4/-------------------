from tkinter import *
from numpy import round

def info_channel_diagram_image_window_create(info_channel_diagram, exposure_time, keyboard_num):
    font_style=("Times New Roman",14)
    H_X,H_X_cond_Y,I_X_Y,H_Y_cond_X,H_Y=info_channel_diagram
    info_channel_diagram_image_window=Tk()
    info_channel_diagram_image = PhotoImage(file="info_channel_diagram.png")
    m=info_channel_diagram_image.width()
    n=info_channel_diagram_image.height()
    info_channel_diagram_image_window.title(f'Диаграмма информационного канала')
    info_channel_diagram_image_window.geometry(f'{m}x{n}')

    canvas = Canvas(bg="white", width=m, height=n)
    canvas.pack(anchor=CENTER, expand=1)
 
    
 
    canvas.create_image(0,0,anchor=NW, image=info_channel_diagram_image)
    
    canvas.create_text(2, 2,
                       anchor=NW,
                       text=f'Клавиатура №{keyboard_num}\n'
                       f'Экспозиция: {exposure_time} мс',
                       font=font_style)

    canvas.create_text(2, (n-50)/2,
                       anchor=NW,
                       text=f'H(X)={round(H_X,4)} бит',
                       font=font_style)
    canvas.create_text((m-100)/2, (n-50)/2,
                       anchor=NW,
                       text=f'I(X,Y)={round(I_X_Y,4)} бит',
                       font=font_style)
    
    canvas.create_text((m)/2, (n-200)/2,
                       anchor=NW,
                       text=f'H(X/Y)={round(H_X_cond_Y,4)} бит',
                       font=font_style)
    canvas.create_text((m)/2, (n+200)/2,
                        anchor=SE,
                        text=f'H(Y/X)={round(H_Y_cond_X,4)} бит',
                        font=font_style)

    canvas.create_text(m-2, (n-50)/2,
                       anchor=NE,
                       text=f'H(Y)={round(H_Y,4)} бит',
                       font=font_style)
    info_channel_diagram_image_window.mainloop()


if __name__=='__main__':
    info_channel_diagram_image_window_create([1.2,1.3,1.4,1.5,1.6],500,1)
    info_channel_diagram_image_window_create([1.293972,1.293972,1.293972,1.293972,1.293972],700,2)
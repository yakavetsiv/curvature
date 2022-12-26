# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import os.path
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter  # useful for `logit` scale
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

figure_canvas_agg = None

class MplColorHelper:
  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)

  def color(self, val):
    return self.cmap(self.norm(val))

def plot_graph(df):
    
    df['c'].plot(kind = "line")
    

def load_data(name, select_every = 1):
    df = pd.DataFrame(columns=['x', 'y', 'Curvature'])
    try:
        data = pd.read_csv(name, index_col=None)
        
        df[['x', 'y', 'Curvature']] = data[['X','Y','Point Curvature (um-1)']]
        
        df = df[df.index % select_every == 0]
    
        dx_dt = np.gradient(df['x'].to_numpy())
        dy_dt = np.gradient(df['y'].to_numpy())
    
        dx_dtdt = np.gradient(dx_dt)
        dy_dtdt = np.gradient(dy_dt)
    
        curv_up = dx_dt*dy_dtdt - dy_dt*dx_dtdt
        curv_down = np.power(dx_dt*dx_dt + dy_dt*dy_dt, 1.5)
    
        curv = np.divide(curv_up,curv_down)
        df['Curvature'] = curv
        
        #df['Curvature'] = np.sign(curv)*df['Curvature']
    except:
        pass
    return df

def draw_line(df, width, height, scale, w, colormap, c_min, c_max, auto, canvas):
    out = Image.new("RGBA", (width, height), (255, 255, 255,0))
    
    if auto:
        c_min = df['Curvature'].min()
        c_max = df['Curvature'].max()

    COL = MplColorHelper(colormap, c_min, c_max)

    draw = ImageDraw.Draw(out)
    for i in range(df.shape[0]-2):
        x1, y1, c1 = df.iloc[i]
        x2, y2, c2 = df.iloc[i+1]
        color = tuple([int(z * 255) for z in COL.get_rgb((c1+c2)/2)])
        draw.line([(int(x1/scale), int(y1/scale)), (int(x2/scale), int(y2/scale))], fill=color, width = w, joint = 'curve')


    figure = plot_results(df, out, COL, canvas)   
    
    return out, figure

    
def plot_results(data, img, color, canvas):
    global figure_canvas_agg
    
    if figure_canvas_agg:
        figure_canvas_agg.get_tk_widget().forget()
        plt.close('all')
   
    parameters = {'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'font.family':'sans-serif',
              'font.sans-serif':['Arial'],
                 'font.size': 14}
    plt.rcParams.update(parameters)
    
    figure, ax = plt.subplots(1, 2, figsize = (7,4))
    
    ax[0].imshow(img)
    ax[0].set_xlabel('')
    ax[0].set_xticks(())
    ax[0].set_yticks(())
    data = data.reset_index()
   
    p = ax[1].scatter(data['index'],data['Curvature'], c = color.get_rgb(data['Curvature'].values), cmap = color.cmap)
    cbar = figure.colorbar(color.scalarMap, ax = ax[1])

    ax[1].set_xlabel('')
    ax[1].set_ylabel('')
    #ax[1].set_yticks(())

    
    
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    
    return figure


def main():
    
    
    
    layout_files = [
                    [   
                        sg.Text("Image File"),
                        sg.In(key = "-IMAGE_FILE-", enable_events=True),
                        sg.FileBrowse(),
                    ], 
                    [
                        sg.Text("Line file *.csv"),
                        sg.In(),
                        sg.FileBrowse(key = "-LINE_FILE-", file_types=(("Data", "*.csv"),))
                    ], 
                    [
                        sg.Text(""),    
                    ],
                    [
                        sg.Text("Width"),
                        sg.In(size=(5, 1), key="-WIDTH-", enable_events=True),
                        sg.Text("Height"),
                        sg.In(size=(5, 1), key="-HEIGHT-", enable_events=True),
                        sg.Text("Scale"),
                        sg.In('0.16',size=(5, 1), key="-SCALE-", enable_events=True),
                        sg.Text("Line Width"),
                        sg.In('10',size=(10, 1), key="-WIDTH_LINE-", enable_events=True),
                       
                    ],
                    [
                        sg.Text("Colormap"),
                        sg.In('rainbow', size=(10, 1), key="-COLORMAP-", enable_events=True),
                        sg.Text("C-min"),
                        sg.In('-0.1', size=(10, 1), key="-C_MIN-", enable_events=True),
                        sg.Text("C-max"),
                        sg.In('0.2', size=(10, 1), key="-C_MAX-", enable_events=True),    
                        sg.Checkbox('Autorange',key="-AUTO-", default=True),
                        sg.Text("Select every:"),
                        sg.In('1', size=(10, 1), key="-SELECT-", enable_events=True),
                    ]
                    
                    ]
    
    
    layout_right= [
                    [
                        sg.Canvas(key='-CANVAS-',size=(50, 50))
                    ], 
                    [
                        sg.Button('Save plot'), 
                        sg.Button('Save image'),
                        sg.Button('Save overlay'), 
                    ]
                    ]


    layout_buttons = [
                    [                           
                        sg.Button('Draw line'),
                        sg.Button('Quit')
                    ]]
                    

    layout_left = layout_files + layout_buttons 
    
    layout = [[sg.Column(layout_left), sg.pin(sg.Column(layout_right, key='-PLOT-', visible = False, pad = (0,0)))]]           
    
    window = sg.Window('Curvature line', layout)

    while True:
        event, values = window.read()


        if event == 'Quit' or event == sg.WIN_CLOSED :
            window.close()
            break
        
        elif event == "-IMAGE_FILE-":  # A file was chosen from the listbox
            try:    
                with Image.open(values["-IMAGE_FILE-"]) as im:
                        width, height = im.size    
                        window["-WIDTH-"].update(width)
                        window["-HEIGHT-"].update(height)
            except:
                pass
        
        elif event == "Draw line": 
            #try:    
            data = load_data(values["-LINE_FILE-"], int(values["-SELECT-"]))
                
            img, figure = draw_line(data, int(values["-WIDTH-"]), int(values["-HEIGHT-"]), float(values["-SCALE-"]), int(values["-WIDTH_LINE-"]), values["-COLORMAP-"], float(values["-C_MIN-"]), float(values["-C_MAX-"]), values["-AUTO-"], window['-CANVAS-'].TKCanvas)
            
            window['-PLOT-'].update(visible = True)
            #except:
                #sg.popup("Cancel", "Error")

        
        elif event == "Save image": 
            text = sg.popup_get_file('Please enter a file name',  save_as = True)
            try:
                img.save(text + ".png", "PNG")
            except:
                sg.popup("Error")

        elif event == "Save plot": 
            name = sg.popup_get_file('Save Plot', save_as=True)
            if not name:
                sg.popup("Cancel", "No filename supplied")
            else:
                try:
                    figure.savefig(name +'.png', dpi = 200)
                except:
                    sg.popup("Cancel", "Error")
                    
        elif event == "Save overlay": 
            
            name = sg.popup_get_file('Save overlay', save_as=True)

            if not name:
                sg.popup("Cancel", "No filename supplied")
            try:    
                with Image.open(values["-IMAGE_FILE-"]) as im:
                    im = im.convert("RGBA")
                    Image.alpha_composite(im, img).save(name +'.png')
            except:
                pass

            
        
            
         
        
            
if __name__ == "__main__":
    main()
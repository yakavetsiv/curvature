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
from matplotlib.ticker import MultipleLocator
from shape import Shape
import math
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import re
from scipy import interpolate, stats
from scipy.signal import savgol_filter
import warnings


figure_canvas_agg = None
img = Image.new("RGBA", (500, 500), (255, 255, 255,0))
fig = None
data = None

fnames = []
csv_filename = ""




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

def filelist(folder, ext):
    try:
        # Get list of files in folder
        file_list = os.listdir(folder)
    except:
        file_list = []
    fnames = [
        f
        for f in file_list
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith((ext))
    ]
    return fnames


def fit_scale(p, mask, an):
    scale = 0.01
    uu, vv = vector_transform(an, scale, p=p) 
    while mask[int(vv),int(uu)]:
        scale = scale + 0.01
        uu, vv = vector_transform(an, scale, p=p) 
    return scale
    
def dir_number(dir):
    r = 0
    for root, dirs, files in os.walk(dir):
        if len(files)>0:
            r += 1   
    return r


def calc_normals(data0, data1, size):
    
    img = Image.new("1", size, color = 0)

    draw = ImageDraw.Draw(img)
    pol1 = [(e[0], e[1]) for e in data1[['x','y']].to_numpy().reshape(len(data1), 2).tolist()]
    draw.polygon(pol1, fill = 1) 
    mask_2d = np.asarray(img)
    
    
    
    data = data0.copy()
    data['an'] = 0
    data['l'] = 0
    data['u'] = 0
    data['v'] = 0
    
    data.loc[0, 'an'] = normal(data0[['x','y']].iloc[-2].values, data0[['x','y']].iloc[0].values, data0[['x','y']].iloc[1].values)
    data.loc[0, 'l'] = fit_scale(data0[['x','y']].iloc[0].values, mask_2d, data[['an']].iloc[0].values)
    pn1_x, pn1_y = vector_transform(data[['an']].iloc[0].values, data[['l']].iloc[0].values, data[['x','y']].iloc[0].values)
    data.loc[0, 'u'] = pn1_x
    data.loc[0, 'v'] = pn1_y
    
    data.loc[len(data)-1, 'an'] = normal(data0[['x','y']].iloc[-2].values, data0[['x','y']].iloc[-1].values, data0[['x','y']].iloc[0].values)
    data.loc[len(data)-1, 'l'] = fit_scale(data0[['x','y']].iloc[-1].values, mask_2d, data[['an']].iloc[-1].values)
    pn1_x, pn1_y = vector_transform(data[['an']].iloc[-1].values, data[['l']].iloc[-1].values, data[['x','y']].iloc[-1].values)
    data.loc[len(data)-1, 'u'] = pn1_x
    data.loc[len(data)-1, 'v'] = pn1_y
    
    
    
    for i in range(1, len(data)-1):
        data.loc[i, 'an'] = normal(data0[['x','y']].iloc[i-1].values, data0[['x','y']].iloc[i].values, data0[['x','y']].iloc[i+1].values)
        data.loc[i, 'l'] = fit_scale(data0[['x','y']].iloc[i].values, mask_2d, data[['an']].iloc[i].values)
        pn1_x, pn1_y = vector_transform(data[['an']].iloc[i].values, data[['l']].iloc[i].values, data[['x','y']].iloc[i].values)
        data.loc[i, 'u'] = pn1_x
        data.loc[i, 'v'] = pn1_y
   
        

            
    return data

def vector_transform(an, l, p = [0,0]):
    ###calculation of angle from slope
    x = p[0] + math.sin(an)*l
    y = p[1] + math.cos(an)*l
    return x,y


def normal(p0, p1, p2):
    length = 1
    ##calculation of tangent vector 
    def tan_v(p0, p1, length):
        
        x0, y0, xa, ya = p0[0], p0[1], p1[0], p1[1]
        dx, dy = xa-x0, ya-y0
        norm = math.hypot(dx, dy) * 1/length
        dx /= norm
        dy /= norm
        return dx, dy
    
    dx1, dy1 = tan_v(p0, p1, length)
    dx2, dy2 = tan_v(p1, p2, length)
    
    dx = (dx1+dx2)/2
    dy = (dy1+dy2)/2
    
    #90degree rotation of tangent vector
    #u, v = p1[0]-dy, p1[1]+dx
    
    if ((-dy)> 0) & (dx >0):
        an = math.acos(dx/length)
    if ((-dy)< 0) & (dx >0):
        an = 2*math.pi - math.acos(dx/length)
        
    if ((-dy)< 0) & (dx <0):
        an = 2*math.pi -  math.acos(dx/length)
    
    if ((-dy)> 0) & (dx <0):
        an = math.acos(dx/length)
    
    return an
    
def cacl_lenght(data):
    data['lenght'] = 0
    data.loc[0, 'lenght'] = 0
    for i in range(len(data.index)-1):       
        data.loc[i+1, 'lenght'] = data.iloc[i]['lenght'] + math.dist((data.iloc[i]['x'], data.iloc[i]['y']), (data.iloc[i+1]['x'], data.iloc[i+1]['y']))
    return data  

def auto_save(filename, img, figure, data):
    
    name = filename
    name = re.search(r'.+(?= )', name).group(0)
    
    if not name:
        sg.popup("Cancel", "Error")
    try:    
        img.save(name + "_img.png", "PNG")
        figure.savefig(name +'_fig.png', dpi = 300)
        df_sum = data.copy()
        df_sum = df_sum.drop('dataset', axis=1)
        with pd.ExcelWriter(name + '_data.xlsx') as writer: 
            df_sum.to_excel(writer, sheet_name='Summary', index=True)
            for i in range(len(data.index)):
                df = data.iloc[i]['dataset'].copy()
                time = data.iloc[i]['time']
                df.to_excel(writer, sheet_name=f'{time} day', index=False)
    except:
        pass


def add_curve(curves, csv_filename, day, width, height, angle, res):
    cur = Shape(csv_filename, day, (width, height), angle, res)
    curves.append(cur)  
    return curves
        
def update_img(curves, res, width, height, w_line, colormap, c_min, c_max, auto_flag, res_flag = False):
    img = Image.new("RGBA", (width, height), (255, 255, 255,0))
    if len(curves)>0:
        for curve in curves:
            if res_flag:
                curve.set_res(res)
            im = curve.make_img(w_line, colormap, c_min, c_max, auto_flag)
            img = Image.alpha_composite(im, img)
    return img
    

def plot_results(curves, img, canvas):
    global figure_canvas_agg
    
    if figure_canvas_agg:
        figure_canvas_agg.get_tk_widget().forget()
        plt.close('all')
   
    parameters = {'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'font.family':'sans-serif',
              'font.sans-serif':['Arial'],
                 'font.size': 12}
    plt.rcParams.update(parameters)
    

    figure, ax = plt.subplots(3, 2, figsize = (8,12))

    times = []
    areas = []
    widths = []
    heights = []
    datasets = []
    curv_mean = []
    lenght_total = []
    res = []
    curv_mean_sm = []
    circ = []
    sol = []
    for curve in curves:
        
        
        #ax[1,0].step(x = [i for i in range(hist_x.size)], y = hist_x, label = f'{curve.time} day')
        #ax[1,1].step(x = [j for j in range(hist_y.size)], y = hist_y, label = f'{curve.time} day')

        times.append(curve.time)
        areas.append(curve.area)
        widths.append(curve.width)
        heights.append(curve.height)
        curv_mean.append(curve.data['c'].mean())
        
        res.append(curve.res)
        
        data = curve.data.reset_index()
        
        cacl_lenght(data)
        lenght_total.append(data.iloc[-1]['lenght']*100)
        data['lenght'] = data['lenght']/data.iloc[-1]['lenght']*100
        
        circ.append(curve.circ)
        sol.append(curve.solidity)
        #x_new = np.linspace(0, 100, curve.res)
        #bspline = interpolate.make_interp_spline(data['lenght'].values, data['c'].values)
        #y_new = bspline(x_new)
        c_smooth = savgol_filter(data['c'].values, 11, 2)
        curv_mean_sm.append(c_smooth.mean())
        data['c_smooth'] = c_smooth
        datasets.append(data)
    
    kin = pd.DataFrame(columns=['time', 'area', 'width', 'height', 'width_raw', 'height_raw', 'curv_mean', 'curv_mean_sm', 'len_total', 'circ', 'sol'])
    kin['time'] = times
    kin['res'] = res
    kin['area'] = areas
    kin['width_raw'] = widths
    kin['height_raw'] = heights
    kin['curv_mean'] = curv_mean
    kin['curv_mean_sm'] = curv_mean_sm
    kin['len_total'] = lenght_total
    kin['circ'] = circ
    kin['sol'] = sol
    
    
    kin['dataset'] = datasets
    kin = kin.sort_values(by = 'time', ignore_index=True)
    kin['width'] = kin['width_raw']-kin['width_raw'].values[0]
    kin['height'] = kin['height_raw']-kin['height_raw'].values[0]
    
    
    for i in range(len(kin.index)):
        data = kin.iloc[i]['dataset']
        time = kin.iloc[i]['time']
        
        ax[0,1].scatter(data['lenght'], data['c'], alpha = 0.4, s = 10) 
        ax[0,1].plot(data['lenght'], data['c_smooth'], label = f'{time} day')      
        
    
    w, h = img.size
    ax[0,0].set_title('Curvature')
    ax[0,0].imshow(img)
    ax[0,0].set_xlabel('x')
    ax[0,0].set_ylabel('y')
    ax[0,0].plot([0,w], [h/2, h/2], '--', color = 'tab:grey', alpha = 0.5, linewidth=0.5 )
    ax[0,0].plot([w/2,w/2], [0, h], '--', color = 'tab:grey', alpha = 0.5, linewidth=0.5 )

    
    ax[0,1].set_title('Curvature distribution', fontsize=14)
    #ax[0,1].scatter(x = kin['time'], y = kin['width'], label = 'x-size')
    #ax[0,1].plot(kin['time'], kin['width'])
    #ax[0,1].scatter(x = kin['time'], y = kin['height'], label = 'y-size')
    #ax[0,1].plot(kin['time'], kin['height'])
    ax[0,1].legend()
    ax[0,1].set_xlabel('Perimeter, %')
    ax[0,1].set_ylabel('Curvature')
    
    if len(curves)>1:
        time0 = kin.iloc[0]['time']
        time1 = kin.iloc[-1]['time']
        COL = curves[0].COL
        
        data0 = kin.iloc[0]['dataset'].copy()
        data1 = kin.iloc[-1]['dataset'].copy()
            
        data = calc_normals(data0, data1, (w,h))
        
        for i in range(len(data.index)):
            x = data.iloc[i]['x']
            y = data.iloc[i]['y']
            an = data.iloc[i]['an']
            c = data.iloc[i]['c']
            u = data.iloc[i]['u']
            v = data.iloc[i]['v']
            l = data.iloc[i]['l']
            
            uu, vv = vector_transform(an, l) 
            if c <= 0:
                ax[1,0].quiver(x, y, uu, vv, angles = 'uv', color=COL.color(c), units='xy', scale = 1, alpha = 1, width = 2)
            #plt.plot((x,u), (y,v), color=cmap(norm(c)))
           
            if i ==0:
                ax[1,0].scatter(x, y, s = 40, c = 'black', alpha = 0.5)
            elif i ==10:
                ax[1,0].scatter(x, y, s = 20, c = 'black', alpha = 0.5)
            else:
                ax[1,0].scatter(x, y, s = 1)
                
        
        #pol = data[['x','y']].to_numpy().reshape(len(data), 2)
        #polygon = Polygon(pol, closed=True)
        pol1 = data1[['x','y']].to_numpy().reshape(len(data1), 2)
        polygon1 = Polygon(pol1, closed=True)
        
    
        collection = PatchCollection([polygon1], alpha=0.05,  zorder= 1, color = ["tab:red"])
        ax[1,0].add_collection(collection)
        
        ax[1,0].set_xlim(0,w)
        ax[1,0].set_ylim(0,h)
    
        
        ax[1,0].set_title('Growth rate', fontsize=14)
        ax[1,0].set_ylabel('y')
        ax[1,0].set_xlabel('x')
        
        kin['pearson'] = 0
        kin['l_mean'] = 0
        kin['l_median'] = 0
        kin['l_max'] = 0
        
        l = []
        l_m = []
        t = []
        
        datasets = [kin.iloc[0]['dataset'].copy()]
        for i in range(len(curves)-1):
            #loading the datasets and time-label
            data0 = kin.iloc[i]['dataset'].copy()
            data1 = kin.iloc[i+1]['dataset'].copy()
            t0 = int(kin.iloc[i]['time'])
            t1 = int(kin.iloc[i+1]['time'])
            t.append(t1)
            
            ###lenght distribution alonge the perimeter
            data = calc_normals(data0, data1, (w,h))
            data = curves[0].filter_outliers(data, 'l')
            datasets.append(data)
            
            #smoothing the growth data
            l_smooth = savgol_filter(data['l'].values, 11, 2)
            data['l_sm'] = l_smooth
            
            #calculation of growth stats
            kin.loc[i+1, 'l_mean'] = data['l'].mean()
            kin.loc[i+1, 'l_median'] = data['l'].median()
            kin.loc[i+1, 'l_max'] = data['l'].max()
            
            #plotting the data + smooth line
            ax[1,1].scatter(data['lenght'], data['l'], alpha = 0.4, s = 10) 
            ax[1,1].plot(data['lenght'], l_smooth, label = f'{t0}-{t1}')  
            l.append(data['l'].values)
            l_m.append(data['l'].values.mean())
            
            
            #filtering point with the negative curvature
            data_sub = data[data['c'] <= 0]
            #caclulation of pearson correlation coefficientbetween growth and curvature (c<0)
            r = np.corrcoef(data_sub['c'], data_sub['l'])[1,0]
            kin.loc[i+1, 'pearson'] = r
            #plotting the growth vs curvature
            ax[2,1].scatter(data_sub['c'], data_sub['l'], s = 10, label = f'{t0}-{t1}: P = {round(r, 2)}')
            
        
        kin['dataset'] = datasets
        
        ax[1,1].set_title('Growth distribution', fontsize=14)
        ax[1,1].legend()
        ax[1,1].set_xlabel('Perimeter, %')
        ax[1,1].set_ylabel('Growth, px')    
        
        ax[2,1].legend()
        ax[2,1].set_title('Curvature (c<0) vs growth', fontsize=14)
        ax[2,1].set_xlabel('Curvature')
        ax[2,1].set_ylabel('Growth')
        
        
        v = ax[2,0].violinplot(l)

        for pc in v['bodies']:
            pc.set_facecolor('red')
            pc.set_edgecolor('black')
            
        for partname in ('cbars','cmins','cmaxes'):
            vp = v[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)
        
        #ax[2,0].scatter(x = t, y = l_m, s= 20, color = 'white')
        
        ax[2,0].set_title('Growth kinetics', fontsize=14)
        xticks = kin['time'].values.tolist()
        ax[2,0].xaxis.set_major_locator(MultipleLocator(1))
        ax[2,0].set_xticklabels(xticks)
        #ax[2,0].set_xlabel('Time, days')
        ax[2,0].set_ylabel('Growth, px')
    
    
    
    figure.tight_layout()
    #figure.show()

    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    
    return figure, kin




def main():
    global fnames
    global csv_filename
    warnings.filterwarnings("ignore")
    curves = []
    
    layout_files = [
                    [
                        sg.Text("Image Folder"),
                        sg.In(size=(50, 1), enable_events=True, key="-FOLDER-"),
                        sg.FolderBrowse(),
                    ],
                    [
                        sg.Listbox(values=fnames, enable_events=True, size=(80, 10), key="-FILE LIST-")
                    ],
                    [
                        sg.Text(""),    
                    ],
                    [
                        sg.Text("Width"),
                        sg.In('1608', size=(5, 1), key="-WIDTH-", enable_events=True),
                        sg.Text("Height"),
                        sg.In('1608',size=(5, 1), key="-HEIGHT-", enable_events=True),
                        sg.Text("Day"),
                        sg.In('0',size=(3, 1), key="-DAY-", enable_events=True),
                        sg.Text("Line Width"),
                        sg.In('10',size=(3, 1), key="-WIDTH_LINE-", enable_events=True),
                        sg.Text("Rotation angle"),
                        sg.In('0',size=(3, 1), key="-ANGLE-", enable_events=True),
                       
                    ],
                    [
                        sg.Text("Resolution"),
                        sg.In('100', size=(5, 1), key="-RES-", enable_events=True),
                        
                        sg.Text("Colormap"),
                        sg.In('rainbow', size=(10, 1), key="-COLORMAP-", enable_events=True),
                        sg.Text("C-min"),
                        sg.In('-0.01', size=(5, 1), key="-C_MIN-", enable_events=True),
                        sg.Text("C-max"),
                        sg.In('0.01', size=(5, 1), key="-C_MAX-", enable_events=True),    
                        sg.Checkbox('Autorange',key="-AUTO-", default=False),
                        
                    ]
                    
                    ]
    
    
    layout_right= [
                    [
                        sg.Canvas(key='-CANVAS-',size=(50, 50))
                    ], 
                    [
                        sg.Button('Save plot'), 
                        sg.Button('Save image'),
                        sg.Button('Save data'),
                        sg.Text(" "),
                        sg.Button('Auto save'),
                    ]
                    ]


    layout_buttons = [
                    [          
                        sg.Button('Add curve'),
                        sg.Button('Automated analysis'),
                        sg.Text(""), 
                        sg.Button('Update settings'),
                        sg.Button('Clear all'),
                        sg.Button('Clear last'),
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
        
        elif event == "Clear all":
            curves.clear()   
            window['-PLOT-'].update(visible = False)
            
            #clean canvas
            
        elif event == "Clear last":
            day, width, height, angle = int(values["-DAY-"]), int(values["-WIDTH-"]), int(values["-HEIGHT-"]), int(values["-ANGLE-"])
            res, w_line, colormap = int(values["-RES-"]), int(values["-WIDTH_LINE-"]), values["-COLORMAP-"]
            c_min, c_max, auto_flag = float(values["-C_MIN-"]), float(values["-C_MAX-"]), values["-AUTO-"]
            
            if len(curves)>1:
                curves.pop()    
            if len(curves)>0:
                img = update_img(curves, res, width, height, w_line, colormap, c_min, c_max, auto_flag, False)
                figure, data = plot_results(curves, img, window['-CANVAS-'].TKCanvas) 
                window['-PLOT-'].update(visible = True)
            else:
                window['-PLOT-'].update(visible = False)
                
            #clean canvas
        
        elif event == "Update settings": 
            width, height = int(values["-WIDTH-"]), int(values["-HEIGHT-"])
            res, w_line, colormap, c_min, c_max, auto_flag = int(values["-RES-"]), int(values["-WIDTH_LINE-"]), values["-COLORMAP-"], float(values["-C_MIN-"]), float(values["-C_MAX-"]), values["-AUTO-"]
            img = update_img(curves, res, width, height, w_line, colormap, c_min, c_max, auto_flag, True)
            
            figure, data = plot_results(curves, img, window['-CANVAS-'].TKCanvas) 
            window['-PLOT-'].update(visible = True)
            
        
        elif event == "Add curve": 
                
            day, width, height, angle = int(values["-DAY-"]), int(values["-WIDTH-"]), int(values["-HEIGHT-"]), int(values["-ANGLE-"])
            res, w_line, colormap = int(values["-RES-"]), int(values["-WIDTH_LINE-"]), values["-COLORMAP-"]
            c_min, c_max, auto_flag = float(values["-C_MIN-"]), float(values["-C_MAX-"]), values["-AUTO-"]
            curves = add_curve(curves, csv_filename, day, width, height, angle, res)
            img = update_img(curves, res, width, height, w_line, colormap, c_min, c_max, auto_flag, False)
            
            figure, data = plot_results(curves, img, window['-CANVAS-'].TKCanvas) 
            window['-PLOT-'].update(visible = True)

        elif event == "Automated analysis":
            root_dir = sg.popup_get_folder('Please enter a folder name')
            
            width, height, angle = int(values["-WIDTH-"]), int(values["-HEIGHT-"]), int(values["-ANGLE-"])
            res, w_line, colormap = int(values["-RES-"]), int(values["-WIDTH_LINE-"]), values["-COLORMAP-"]
            c_min, c_max, auto_flag = float(values["-C_MIN-"]), float(values["-C_MAX-"]), values["-AUTO-"]
            
            curves.clear()   
            window['-PLOT-'].update(visible = False)
            
            i=0
            n = dir_number(root_dir)
            for root, dirs, files in os.walk(root_dir):
                if len(files)>0:
                    fnames = filelist(root, ".csv")
                    curves.clear()
                    if len(fnames)>0:
                        try:
                            for name in fnames:
                                csv_filename = os.path.join(root, name)
                                day = int(re.search(r'(?<= )([0-9]*)(?=.csv)', name).group(1))
                                
                                curves = add_curve(curves, csv_filename, day, width, height, angle, res)
                            img = update_img(curves, res, width, height, w_line, colormap, c_min, c_max, auto_flag)
                            figure, data = plot_results(curves, img, window['-CANVAS-'].TKCanvas) 
                            auto_save(csv_filename, img, figure, data)
                            i += 1
                            progress = round(i/n*100)
                            print(f'{root} - Done!')
                            print(f'Progress: {progress}%')
                        except:
                            i += 1
                            progress = round(i/n*100)
                            print(f'{root} - Error')
                            print(f'Progress: {progress}%')

        elif event == "Save image": 
            name = sg.popup_get_file('Please enter a file name',  save_as = True)
            try:
                img.save(name + ".png", "PNG")
            except:
                sg.popup("Error")

        elif event == "Save plot": 
            name = sg.popup_get_file('Save Plot', save_as=True)
            if not name:
                sg.popup("Cancel", "No filename supplied")
            else:
                try:
                    
                    figure.savefig(name +'.png', dpi = 300)
                except:
                    sg.popup("Cancel", "Error")
                    
        elif event == "Save data": 
            
            name = sg.popup_get_file('Save report', save_as=True)

            if not name:
                sg.popup("Cancel", "No filename supplied")
            try:    
                data.to_csv(name + '.csv',index=True)
            except:
                pass
            
        elif event == "Auto save": 
            auto_save(csv_filename, img, figure, data)
            
        
        elif event == "-FOLDER-":   # New folder has been chosen
            fnames = filelist(values["-FOLDER-"], ".csv")
            window["-FILE LIST-"].update(fnames)
            
        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            try:
                csv_filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
                s = values["-FILE LIST-"][0]
                day = re.search(r'(?<= )([0-9]*)(?=.csv)', s).group(1)
                window['-DAY-'].update(day)
            except:
                pass
        
            
        
            
         
        
            
if __name__ == "__main__":
    main()
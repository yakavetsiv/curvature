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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from shape import Shape, MplColorHelper
import math
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import re
from scipy import interpolate, stats
from scipy.signal import savgol_filter
import warnings
from datetime import date
from scipy.interpolate import UnivariateSpline, CubicSpline

figure_canvas_agg = None
img = Image.new("RGBA", (500, 500), (255, 255, 255,0))
fig = None
data = None

fnames = []
csv_filename = ""

#colors = [(20,42,64), (36,77,111), (56,116,167), (84, 173, 240)]
colors = [(20,42,64), (84, 173, 240)]


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


def fit_scale(p, mask, an, inv_flag):
    if inv_flag:
        inc = -0.01   
        scale = -0.01
    else: 
        inc = 0.01    
        scale = 0.01
    
    uu, vv = vector_transform(an, scale, p=p) 
    while mask[int(vv),int(uu)]:
        scale = scale + inc
        uu, vv = vector_transform(an, scale, p=p) 
    return scale
    
def dir_number(dir):
    r = 0
    for root, dirs, files in os.walk(dir):
        if len(files)>0:
            r += 1   
    return r


def calc_normals(data0, data1, size, inv_flag = False):
    img = Image.new("1", size, color = 0)
    draw = ImageDraw.Draw(img)
    
    if inv_flag:
        
        pol1 = [(e[0], e[1]) for e in data1[['x','y']].to_numpy().reshape(len(data1), 2).tolist()]
        pol0 = [(e[0], e[1]) for e in data0[['x','y']].to_numpy().reshape(len(data0), 2).tolist()]
        draw.polygon(pol1, fill = 1, outline =1) 
        draw.polygon(pol0, fill = 0, outline =0)
        #img.show()
        mask_2d = np.asarray(img)
        
    
        data = data1.copy()
        data['an'] = 0
        data['l'] = 0
        data['u'] = 0
        data['v'] = 0
        
        data.loc[0, 'an'] = normal(data1[['x','y']].iloc[-1].values, data1[['x','y']].iloc[0].values, data1[['x','y']].iloc[1].values)
        data.loc[0, 'l'] = fit_scale(data1[['x','y']].iloc[0].values, mask_2d, data[['an']].iloc[0].values, inv_flag)
        pn1_x, pn1_y = vector_transform(data[['an']].iloc[0].values, data[['l']].iloc[0].values, data[['x','y']].iloc[0].values)
        data.loc[0, 'u'] = pn1_x
        data.loc[0, 'v'] = pn1_y
        
        data.loc[len(data)-1, 'an'] = normal(data1[['x','y']].iloc[-2].values, data1[['x','y']].iloc[-1].values, data1[['x','y']].iloc[0].values)
        data.loc[len(data)-1, 'l'] = fit_scale(data1[['x','y']].iloc[-1].values, mask_2d, data[['an']].iloc[-1].values,inv_flag)
        pn1_x, pn1_y = vector_transform(data[['an']].iloc[-1].values, data[['l']].iloc[-1].values, data[['x','y']].iloc[-1].values)
        data.loc[len(data)-1, 'u'] = pn1_x
        data.loc[len(data)-1, 'v'] = pn1_y
        
        
        
        for i in range(1, len(data)-1):
            data.loc[i, 'an'] = normal(data1[['x','y']].iloc[i-1].values, data1[['x','y']].iloc[i].values, data1[['x','y']].iloc[i+1].values)
            data.loc[i, 'l'] = fit_scale(data1[['x','y']].iloc[i].values, mask_2d, data[['an']].iloc[i].values,inv_flag)
            pn1_x, pn1_y = vector_transform(data[['an']].iloc[i].values, data[['l']].iloc[i].values, data[['x','y']].iloc[i].values)
        
            data.loc[i, 'u'] = pn1_x
        data.loc[i, 'v'] = pn1_y
        
    else:

        pol1 = [(e[0], e[1]) for e in data1[['x','y']].to_numpy().reshape(len(data1), 2).tolist()]
        draw.polygon(pol1, fill = 1) 
        mask_2d = np.asarray(img)
        
        
        
        
        data = data0.copy()
        data['an'] = 0
        data['l'] = 0
        data['u'] = 0
        data['v'] = 0
        
        data.loc[0, 'an'] = normal(data0[['x','y']].iloc[-1].values, data0[['x','y']].iloc[0].values, data0[['x','y']].iloc[1].values)
        data.loc[0, 'l'] = fit_scale(data0[['x','y']].iloc[0].values, mask_2d, data[['an']].iloc[0].values, inv_flag)
        pn1_x, pn1_y = vector_transform(data[['an']].iloc[0].values, data[['l']].iloc[0].values, data[['x','y']].iloc[0].values)
        data.loc[0, 'u'] = pn1_x
        data.loc[0, 'v'] = pn1_y
        
        data.loc[len(data)-1, 'an'] = normal(data0[['x','y']].iloc[-2].values, data0[['x','y']].iloc[-1].values, data0[['x','y']].iloc[0].values)
        data.loc[len(data)-1, 'l'] = fit_scale(data0[['x','y']].iloc[-1].values, mask_2d, data[['an']].iloc[-1].values,inv_flag)
        pn1_x, pn1_y = vector_transform(data[['an']].iloc[-1].values, data[['l']].iloc[-1].values, data[['x','y']].iloc[-1].values)
        data.loc[len(data)-1, 'u'] = pn1_x
        data.loc[len(data)-1, 'v'] = pn1_y
        
        
        
        for i in range(1, len(data)-1):
            data.loc[i, 'an'] = normal(data0[['x','y']].iloc[i-1].values, data0[['x','y']].iloc[i].values, data0[['x','y']].iloc[i+1].values)
            data.loc[i, 'l'] = fit_scale(data0[['x','y']].iloc[i].values, mask_2d, data[['an']].iloc[i].values,inv_flag)
            pn1_x, pn1_y = vector_transform(data[['an']].iloc[i].values, data[['l']].iloc[i].values, data[['x','y']].iloc[i].values)
                
            data.loc[i, 'u'] = pn1_x
            data.loc[i, 'v'] = pn1_y
   
        

            
    return data



def sim_normals(curves, data0, size, l_sim = 10, res = 100, folder = None):
    #img = Image.new("1", size, color = 0)
    #draw = ImageDraw.Draw(img)


    #pol1 = [(e[0], e[1]) for e in data1[['x','y']].to_numpy().reshape(len(data1), 2).tolist()]
    #draw.polygon(pol1, fill = 1) 
    #mask_2d = np.asarray(img)
    
    data = data0.copy()
    data['an'] = 0
    data['l'] = l_sim
    data['u'] = 0
    data['v'] = 0
    
    data.loc[0, 'an'] = normal(data0[['x','y']].iloc[-1].values, data0[['x','y']].iloc[0].values, data0[['x','y']].iloc[1].values)

    pn1_x, pn1_y = vector_transform(data[['an']].iloc[0].values, data[['l']].iloc[0].values, data[['x','y']].iloc[0].values)
    data.loc[0, 'u'] = pn1_x
    data.loc[0, 'v'] = pn1_y
    
    data.loc[len(data)-1, 'an'] = normal(data0[['x','y']].iloc[-2].values, data0[['x','y']].iloc[-1].values, data0[['x','y']].iloc[0].values)
    pn1_x, pn1_y = vector_transform(data[['an']].iloc[-1].values, data[['l']].iloc[-1].values, data[['x','y']].iloc[-1].values)
    data.loc[len(data)-1, 'u'] = pn1_x
    data.loc[len(data)-1, 'v'] = pn1_y
    
    
    for i in range(1, len(data)-1):
        data.loc[i, 'an'] = normal(data0[['x','y']].iloc[i-1].values, data0[['x','y']].iloc[i].values, data0[['x','y']].iloc[i+1].values)
        pn1_x, pn1_y = vector_transform(data[['an']].iloc[i].values, data[['l']].iloc[i].values, data[['x','y']].iloc[i].values)
        data.loc[i, 'u'] = pn1_x
        data.loc[i, 'v'] = pn1_y
    
    data['x'] = data['u']
    data['y'] = data['v']


    data_sm = smooth(data, res = res)
    #print(data_sm)
    data_sim = curves[0].curvature(data_sm)
    data0 =  curves[0].data.copy()
    data1 =  curves[-1].data.copy()
    COL = curves[0].COL
    out = Image.new("RGBA", size, (255, 255, 255, 0))
    img_0 = draw_contour(out, data0, COL, c_day = colors[0])
    img_1 = draw_contour(img_0, data_sim, COL)
    img_out = draw_contour(img_1, data1, COL, c_day = colors[1])


    if folder is None: 
        img_filename = sg.popup_get_file('Please enter a file name',  save_as = True)
    else:
        img_name = f'l_sim-{l_sim}_res-{res}.png'
        img_filename = os.path.join(folder, img_name)
    try:
        img_out.save(img_filename, "PNG")
    except:
        sg.popup("Error")
        
    img_out.show()
    

def draw_contour(out, data, COL, w = 10, scale = 1, c_day = None):
    r = 20
    flag_30 = False
    draw = ImageDraw.Draw(out)
    for i in range(data.shape[0]-1):
        x1, y1, c1, u1 = data.iloc[i]
        x2, y2, c2, u2 = data.iloc[i+1]
        if c_day is None:
            color = tuple([int(z * 255) for z in COL.get_rgb((c1+c2)/2)])
        else:
            color = c_day
        if i == 0:    
            draw.ellipse([(int(x1/scale)-r, int(y1/scale)-r), (int(x1/scale)+r, int(y1/scale)+r)], fill=(0,0,0), width = 1)
        if (u1 > 0.3) & (flag_30 == False):    
            draw.rectangle([(int(x1/scale)-r, int(y1/scale)-r), (int(x1/scale)+r, int(y1/scale)+r)], fill=(0,0,0), width = 1)
            flag_30 = True
        
        draw.line([(int(x1/scale), int(y1/scale)), (int(x2/scale), int(y2/scale))], fill=color, width = w, joint = 'curve')
    
    x1, y1, c1, u1 = data.iloc[-1]
    x2, y2, c2, u2 = data.iloc[0]
    draw.line([(int(x1/scale), int(y1/scale)), (int(x2/scale), int(y2/scale))], fill=color, width = w, joint = 'curve')

    return out

def vector_transform(an, l, p = [0,0]):
    ###calculation of angle from slope
    x = p[0] + math.sin(an)*int(l)
    y = p[1] + math.cos(an)*int(l)
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
    an = 0
    
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
        sg.popup("Cancel", "Report files were saved")
    except:
        pass


def add_curve(curves, csv_filename, day, width, height, angle, res, scale):
    cur = Shape(csv_filename, day, (width, height), angle, res, scale)
    curves.append(cur)  
    return curves
        
def update_img(curves, res, width, height, w_line, colormap, c_min, c_max, auto_flag, res_flag = False, day_flag=False, scale = 1, angle = 0):
    img = Image.new("RGBA", (width, height), (255, 255, 255,0))
    
    
    if len(curves)>0:
        for i, curve in enumerate(curves):
            if res_flag:
                curve.set_scale(scale)
                curve.set_angle(angle)
                curve.set_res(res)
            if day_flag:
                im = curve.make_img(w_line, colormap, c_min, c_max, auto_flag, colors[i])    
            else: 
                im = curve.make_img(w_line, colormap, c_min, c_max, auto_flag)
            img = Image.alpha_composite(im, img)
    return img

def smooth(data, k=3, s= 0.2, res = 100):
        
        points = data[['x', 'y']].to_numpy()
        points = np.append(points,points[:1], axis=0)
        #points = np.roll(points, int(self.res/2), axis = 0)
        
        # Linear length along the line:
        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]
    
        # Build a list of the spline function, one for each dimension:
        splines = [UnivariateSpline(distance, coords, k = k) for coords in points.T]
        #splines = [CubicSpline(distance, coords) for coords in points.T]
    
        # Computed the spline for the asked distances:
        alpha = np.linspace(0, 1, res)

        points_fitted = np.stack([spl(alpha) for spl in splines], axis=1)
        #np.stack([np.ones(3) for _ in range(3)]) 

        new_data = pd.DataFrame(points_fitted, columns = ['x', 'y'])
        
        return new_data 

def create_dataset(curves):
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
        c_smooth = savgol_filter(data['c'].values, 5, 2)
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
    kin['dataset_inv'] = datasets
    
    return kin
    




def plot_results(kin, img, canvas, curve):
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
    #ax[0,0].imshow(img)
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
    
    if len(kin.index)>1:
        time0 = kin.iloc[0]['time']
        time1 = kin.iloc[-1]['time']
        COL = curve.COL
        
        data0 = kin.iloc[0]['dataset'].copy()
        data1 = kin.iloc[-1]['dataset'].copy()
            
        data = calc_normals(data0, data1, (w,h), False)
        
        for i in range(len(data.index)):
            x = data.iloc[i]['x']
            y = data.iloc[i]['y']
            an = data.iloc[i]['an']
            c = data.iloc[i]['c']
            u = data.iloc[i]['u']
            v = data.iloc[i]['v']
            l = data.iloc[i]['l']
            
            uu, vv = vector_transform(an, l) 
            if c >= 0:
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
        
    
        collection = PatchCollection([polygon1], alpha=0.00,  zorder= 1, color = ["tab:red"])
        ax[1,0].add_collection(collection)
        
        
        ax[1,0].set_xlim(0,w)
        ax[1,0].set_ylim(0,h)
    
        
        ax[1,0].set_title('Growth rate', fontsize=14)
        ax[1,0].set_ylabel('y')
        ax[1,0].set_xlabel('x')
        
        #ax[1,0].figure.savefig('test.png')
        
        kin['pearson_neg'] = 0
        kin['pearson_zero'] = 0
        kin['pearson_pos'] = 0
        kin['l_mean'] = 0
        kin['l_median'] = 0
        kin['l_max'] = 0
        
        kin['l_mean_inv'] = 0
        kin['l_median_inv'] = 0
        kin['l_max_inv'] = 0
        
        
        l = []
        l_m = []
        t = []
        
        datasets = [kin.iloc[0]['dataset'].copy()]
        for i in range(len(kin.index)-1):
            #loading the datasets and time-label
            data0 = kin.iloc[i]['dataset'].copy()
            data1 = kin.iloc[i+1]['dataset'].copy()
            t0 = int(kin.iloc[i]['time'])
            t1 = int(kin.iloc[i+1]['time'])
            t.append(t1)
            
            ###lenght distribution alonge the perimeter
            data = calc_normals(data0, data1, (w,h))
            data = curve.filter_outliers(data, 'l')
            datasets.append(data)
            
            #smoothing the growth data
            l_smooth = savgol_filter(data['l'].values, 5, 2)
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
            data_sub = data[data['c_smooth'] <= -0.005]
            #caclulation of pearson correlation coefficientbetween growth and curvature (c<0)
            r_neg = np.corrcoef(data_sub['c_smooth'], data_sub['l_sm'])[1,0]
            kin.loc[i+1, 'pearson_neg'] = r_neg
            
            data_sub = data[(data['c_smooth'] <= 0.005) & (data['c_smooth'] > -0.005)]
            #caclulation of pearson correlation coefficientbetween growth and curvature (c<0)
            r_zero = np.corrcoef(data_sub['c_smooth'], data_sub['l_sm'])[1,0]
            kin.loc[i+1, 'pearson_zero'] = r_zero
            
            data_sub = data[(data['c_smooth'] <= -0.005)]
            #plotting the growth vs curvature
            ax[2,1].scatter(data_sub['c_smooth'], data_sub['l_sm'], s = 10, label = f'{t0}-{t1}: Pneg = {round(r_neg, 2)}')
        
        kin['dataset'] = datasets
        
        
        datasets_inv = [kin.iloc[0]['dataset_inv'].copy()]
        
        for i in range(len(kin.index)-1, 0, -1):
            
            #loading the datasets and time-label
            data0 = kin.iloc[i]['dataset_inv'].copy()
            data1 = kin.iloc[i-1]['dataset_inv'].copy()
            t0 = int(kin.iloc[i]['time'])
            t1 = int(kin.iloc[i-1]['time'])
            #print(t0)
            #print(t1)
            #t.append(t0)
            #print(data0)
            #print(data1)
            ###lenght distribution alonge the perimeter
            data_inv = calc_normals(data1, data0, (w,h), True)
            #data_inv = curve.filter_outliers(data_inv, 'l')
            datasets_inv.append(data_inv)
            #print(data_inv['l'])
            
            #calculation of growth stats
            kin.loc[i, 'l_mean_inv'] = data_inv['l'].mean()
            kin.loc[i, 'l_median_inv'] = data_inv['l'].median()
            kin.loc[i, 'l_max_inv'] = data_inv['l'].max()
            
            #plotting the data + smooth line
            #ax[1,1].scatter(data_inv['lenght'], data_inv['l'], alpha = 0.4, s = 10) 
            
            
            #filtering point with the negative curvature

            data_sub = data_inv[data_inv['c_smooth'] >= 0.003]
            #caclulation of pearson correlation coefficientbetween growth and curvature (c<0)
            r_pos = np.corrcoef(data_sub['c_smooth'], data_sub['l'])[1,0]
            kin.loc[i, 'pearson_pos'] = r_pos
            
        kin['dataset_inv'] = datasets_inv
        
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


def plot_feature(data0, feature, Shapes, label, filename):
    
    fig, ax = plt.subplots(1, len(Shapes), figsize = (12,4))
    
    parameters = {'xtick.labelsize': 12,
                  'ytick.labelsize': 12,
                  'font.family':'sans-serif',
                  'font.sans-serif':['Arial'],
                     'font.size': 12}
    plt.rcParams.update(parameters)
    
    for i, shape in enumerate(Shapes):
        data = data0[data0['Shape'] == shape]
        
        data.boxplot(column=feature, by=['time'], ax=ax[i], grid = False, layout= (3,5))
        ax[i].set_ylabel(feature)
        ax[i].set_title(shape)
    
    for i in range(4):
        
        if label == 'MCF-7':
            if feature == "curv_mean":
                ax[i].set_ylim(-0.012, 0.0015)
            if feature == "pearson":
                ax[i].set_ylim(-0.5, 0.5)
            if feature == "circ":
                ax[i].set_ylim(0.4, 1)
            if feature == "sol":
                ax[i].set_ylim(0.7, 1.05)
            if feature == "l_mean":
                ax[i].set_ylim(0, 85)
            if feature == "l_median":
                ax[i].set_ylim(0, 85)
            if feature == "len_total":
                ax[i].set_ylim(50000, 330000)
                
        if label == 'Patient':
            if feature == "curv_mean":
                ax[i].set_ylim(-0.007, 0.0015)
            if feature == "pearson":
                ax[i].set_ylim(-1, 1)
            if feature == "circ":
                ax[i].set_ylim(0, 1.05)
            if feature == "sol":
                ax[i].set_ylim(0.5, 1.05)
            if feature == "l_mean":
                ax[i].set_ylim(0, 85)
            if feature == "l_median":
                ax[i].set_ylim(0, 85)
            if feature == "len_total":
                ax[i].set_ylim(00, 330000)
        
    fig.suptitle(f'{feature}_{label}', fontsize=16)

    fig.tight_layout()
    
    root, name = os.path.split(filename)
    
    name = name.replace(".xlsx", f"_{feature}_{label}.png")
    
    today = str(date.today())
    filename = f'{root}/{today}/{label}/{name}'
    if not os.path.exists(f'{root}/{today}/{label}'):
        os.makedirs(f'{root}/{today}/{label}')
    
    fig.savefig(filename)

    

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
                        sg.In('3216', size=(5, 1), key="-WIDTH-", enable_events=True),
                        sg.Text("Height"),
                        sg.In('3216',size=(5, 1), key="-HEIGHT-", enable_events=True),
                        sg.Text("Day"),
                        sg.In('0',size=(3, 1), key="-DAY-", enable_events=True),
                        sg.Text("Line Width"),
                        sg.In('10',size=(3, 1), key="-WIDTH_LINE-", enable_events=True),
                        sg.Text("Rotation angle"),
                        sg.In('0',size=(3, 1), key="-ANGLE-", enable_events=True),
                        sg.Text("Scale"),
                        sg.In('0.55',size=(5, 1), key="-SCALE-", enable_events=True),
                       
                    ],
                    [
                        sg.Text("Resolution"),
                        sg.In('70', size=(5, 1), key="-RES-", enable_events=True),
                        sg.Checkbox('Days',key="-DAYS-", default=False),
                        sg.Text("Colormap"),
                        sg.In('rainbow', size=(10, 1), key="-COLORMAP-", enable_events=True),
                        sg.Text("C-min"),
                        sg.In('-0.005', size=(5, 1), key="-C_MIN-", enable_events=True),
                        sg.Text("C-max"),
                        sg.In('0.005', size=(5, 1), key="-C_MAX-", enable_events=True),    
                        sg.Checkbox('Autorange',key="-AUTO-", default=False),
                        sg.Text("Sim growth rate"),
                        sg.In('-1', size=(5, 1), key="-GR_SIM-", enable_events=True),  
                        
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
                        sg.Button('Remove days'),
                    ]
                    ]


    layout_buttons = [
                    [          
                        sg.Button('Add curve'),
                        sg.Button('Automated analysis'),
                        sg.Text(""),
                        sg.Button('Simulation of growth'),
                    ],
                    [
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
            res, w_line, colormap, c_min, c_max, auto_flag, days_flag, scale = int(values["-RES-"]), int(values["-WIDTH_LINE-"]), values["-COLORMAP-"], float(values["-C_MIN-"]), float(values["-C_MAX-"]), values["-AUTO-"], values["-DAYS-"], float(values["-SCALE-"])
            
            if len(curves)>1:
                curves.pop()    
            if len(curves)>0:
                img = update_img(curves, res, width, height, w_line, colormap, c_min, c_max, auto_flag, True, days_flag, scale, angle = angle)
                kin = create_dataset(curves)
                figure, data = plot_results(kin, img, window['-CANVAS-'].TKCanvas, curves[0]) 
                window['-PLOT-'].update(visible = True)
            else:
                window['-PLOT-'].update(visible = False)
                
            #clean canvas
        
        elif event == "Update settings": 
            width, height, angle = int(values["-WIDTH-"]), int(values["-HEIGHT-"]), float(values["-ANGLE-"])
            res, w_line, colormap, c_min, c_max, auto_flag, days_flag, scale = int(values["-RES-"]), int(values["-WIDTH_LINE-"]), values["-COLORMAP-"], float(values["-C_MIN-"]), float(values["-C_MAX-"]), values["-AUTO-"], values["-DAYS-"], float(values["-SCALE-"])
            img = update_img(curves, res, width, height, w_line, colormap, c_min, c_max, auto_flag, True, days_flag, scale, angle = angle)
            
            kin = create_dataset(curves)
            figure, data = plot_results(kin, img, window['-CANVAS-'].TKCanvas, curves[0]) 
            window['-PLOT-'].update(visible = True)
            
        
        elif event == "Add curve": 
                
            day, width, height, angle = int(values["-DAY-"]), int(values["-WIDTH-"]), int(values["-HEIGHT-"]), float(values["-ANGLE-"])
            res, w_line, colormap, c_min, c_max, auto_flag, days_flag, scale = int(values["-RES-"]), int(values["-WIDTH_LINE-"]), values["-COLORMAP-"], float(values["-C_MIN-"]), float(values["-C_MAX-"]), values["-AUTO-"], values["-DAYS-"], float(values["-SCALE-"])
            
            curves = add_curve(curves, csv_filename, day, width, height, angle, res, scale)
            img = update_img(curves, res, width, height, w_line, colormap, c_min, c_max, auto_flag, True, days_flag, scale, angle = angle)
            
            kin = create_dataset(curves)
            figure, data = plot_results(kin, img, window['-CANVAS-'].TKCanvas, curves[0]) 
            window['-PLOT-'].update(visible = True)

        elif event == "Automated analysis":
            root_dir = sg.popup_get_folder('Please enter a folder name')
            
            width, height, angle = int(values["-WIDTH-"]), int(values["-HEIGHT-"]), int(values["-ANGLE-"])
            res, w_line, colormap, c_min, c_max, auto_flag, days_flag, scale = int(values["-RES-"]), int(values["-WIDTH_LINE-"]), values["-COLORMAP-"], float(values["-C_MIN-"]), float(values["-C_MAX-"]), values["-AUTO-"], values["-DAYS-"], float(values["-SCALE-"])
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
                                
                                curves = add_curve(curves, csv_filename, day, width, height, angle, res, scale)
                            img = update_img(curves, res, width, height, w_line, colormap, c_min, c_max, auto_flag, True, days_flag, scale, angle = angle)
                            kin = create_dataset(curves)
                            figure, data = plot_results(kin, img, window['-CANVAS-'].TKCanvas, curves[0])
                            
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
                for i in range(len(data.index)):
                    time = data.iloc[i]['time']

                    rep = data.iloc[i]['dataset']
                    rep.to_csv(name + '_'+str(time) +'_data.csv',index=True)
                    rep_inv = data.iloc[i]['dataset_inv']
                    rep_inv.to_csv(name + '_'+str(time) +'_data_inv.csv',index=True)
            except:
                pass
            
        elif event == "Auto save": 
            auto_save(csv_filename, img, figure, data)

        elif event == "Simulation of growth": 
            
            if len(kin.index)>1:

                width, height = int(values["-WIDTH-"]), int(values["-HEIGHT-"])
                data0 = data.iloc[0]['dataset'].copy()
                res = data.iloc[0]['res']
            
                if int(values["-GR_SIM-"]) < 0 :
                    l_sim = data.iloc[-1]['l_median']
                    #kin.to_csv('_data.csv',index=True)
                else:
                    l_sim = values["-GR_SIM-"]

                #data1 = kin.iloc[-1]['dataset'].copy()
                sim_normals(curves, data0, (width, height), l_sim = l_sim, res = res, folder = values["-FOLDER-"])

            else:
                sg.popup("Cancel", f'Please load more curves')
            
            
        
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
        elif event == "Summary report":
            filename = sg.popup_get_file('Open file', file_types = (('Summary file','.xlsx'), ))
            
            Shapes = ["Circ", "LR", "MR", "U"]
            Features = ["curv_mean", "pearson","circ", "sol", "l_mean", "l_median", "len_total"]            
            
            sheet_names = ['MCF7', 'Patient']
            
            try:
                for sheet_name in sheet_names:
                    sum_df =  pd.read_excel(filename, sheet_name=sheet_name)
                    
                    for feature in Features:
                        plot_feature(sum_df, feature, Shapes, sheet_name, filename)
                    sg.popup("Cancel", f'Report files for {sheet_name} were saved')
            except:
                sg.popup("Cancel", 'Please check the names of sheets')
            
        
        elif event == "Remove days":
            filename = sg.popup_get_file('Open file', file_types = (('Report file *_data.xlsx','.xlsx'), ))
            kin = pd.read_excel(filename, 0)
            kin['dataset'] = 0
            for i in range(len(kin.index)):
                kin['dataset'].iloc[i] = pd.read_excel(filename, i+1)
            
            day_to_remove = 3
            
            i = kin[(kin['time'] == day_to_remove)].index
            kin = kin.drop(i)
            
            img = Image.new("RGBA", (width, height), (255, 255, 255,0))
            
            figure, data = plot_results(kin, img, window['-CANVAS-'].TKCanvas, curves[0])
            window['-PLOT-'].update(visible = True)
            filename = filename.replace("_data.xlsx", "_rem 3.csv" )

            auto_save(filename, img, figure, data)
            
            
            
        
            
         
        
            
if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 16:36:08 2022

@author: vipro
"""
import pandas as pd
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageOps
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from scipy.interpolate import UnivariateSpline, CubicSpline
import math

class MplColorHelper:
  def __init__(self, cmap_name, start_val, stop_val):
    self.start_val = start_val
    self.stop_val = stop_val
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)

  def color(self, val):
    return self.cmap(self.norm(val))

  def renorm(self, start_val, stop_val):
      self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
      self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
      



class Shape:
    
    def __init__(self, data_name, time, dim = (500,500), angle = 0, res_l = 100, scale = 1):
        self.time = time
        self.data_name = data_name
        
        self.dim = dim
        self.res_l = res_l
        self.scale = scale
        self.angle = np.deg2rad(angle)
        
        self.raw_data = self.load_data(self.data_name)
        self.scale_data = self.raw_data.copy()
        
        self.scale_data['x'] = self.raw_data['x'] / self.scale
        self.scale_data['y'] = self.raw_data['y'] / self.scale
        
        self.scale_data['x'].round(0)
        self.scale_data['y'].round(0)
        print(self.scale_data)
        

        
        self.perimeter = self.calc_perimeter()
        self.res = int(self.perimeter/self.res_l)
        self.rotate()
        
        self.data = self.smooth(self.scale_data, k = 3, s = self.res-10)
        
        self.area, self.cx, self.cy, self.solidity = self.moments(self.data)
        
        self.offset_x, self.offset_y = self.offsets()
        
        self.move_start()
               
        
        self.width = self.data['x'].max() - self.data['x'].min()
        self.height = self.data['y'].max() - self.data['y'].min()
        
        self.circ = (4*math.pi*self.area)/(self.perimeter*self.perimeter)
        
        self.data = self.curvature(self.data)
        
        self.data = self.filter_outliers(self.data, 'c')
        

    def set_scale(self, scale):
        self.scale = scale
        self.scale_data = self.raw_data.copy()
        
        self.scale_data['x'] = self.raw_data['x'] / self.scale
        self.scale_data['y'] = self.raw_data['y'] / self.scale
        
        self.scale_data['x'].round(0)
        self.scale_data['y'].round(0)

        

        
        self.perimeter = self.calc_perimeter()
        self.res = int(self.perimeter/self.res_l)
        self.rotate()
        
        self.data = self.smooth(self.scale_data, k = 3, s = self.res-1)
        
        self.area, self.cx, self.cy, self.solidity = self.moments(self.data)
        
        self.offset_x, self.offset_y = self.offsets()
        
        self.move_start()
               
        
        self.width = self.data['x'].max() - self.data['x'].min()
        self.height = self.data['y'].max() - self.data['y'].min()
        
        self.circ = (4*math.pi*self.area)/(self.perimeter*self.perimeter)
        
        self.data = self.curvature(self.data)
        
        self.data = self.filter_outliers(self.data, 'c')
        
        

    def set_res(self, res_l):
        self.res_l = res_l

        self.scale_data = self.raw_data.copy()
        
        self.scale_data['x'] = self.raw_data['x'] / self.scale
        self.scale_data['y'] = self.raw_data['y'] / self.scale
        
        self.scale_data['x'].round(0)
        self.scale_data['y'].round(0)
        

        
        self.perimeter = self.calc_perimeter()
        self.res = int(self.perimeter/self.res_l)
        self.rotate()
        
        self.data = self.smooth(self.scale_data, k = 3, s = self.res-1)
        
        self.area, self.cx, self.cy, self.solidity = self.moments(self.data)
        
        self.offset_x, self.offset_y = self.offsets()
        
        self.move_start()
               
        
        self.width = self.data['x'].max() - self.data['x'].min()
        self.height = self.data['y'].max() - self.data['y'].min()
        
        self.circ = (4*math.pi*self.area)/(self.perimeter*self.perimeter)
        
        self.data = self.curvature(self.data)
        
        self.data = self.filter_outliers(self.data, 'c')
        
    def set_angle(self, angle):
        self.angle = np.deg2rad(angle)
        
        
        self.scale_data = self.raw_data.copy()
        
        self.scale_data['x'] = self.raw_data['x'] / self.scale
        self.scale_data['y'] = self.raw_data['y'] / self.scale
        
        self.scale_data['x'].round(0)
        self.scale_data['y'].round(0)
        

        
        self.perimeter = self.calc_perimeter()
        self.res = int(self.perimeter/self.res_l)
        self.rotate()
        
        self.data = self.smooth(self.scale_data, k = 3, s = self.res-1)
        
        self.area, self.cx, self.cy, self.solidity = self.moments(self.data)
        
        self.offset_x, self.offset_y = self.offsets()
        
        self.move_start()
               
        
        self.width = self.data['x'].max() - self.data['x'].min()
        self.height = self.data['y'].max() - self.data['y'].min()
        
        self.circ = (4*math.pi*self.area)/(self.perimeter*self.perimeter)
        
        self.data = self.curvature(self.data)
        
        self.data = self.filter_outliers(self.data, 'c')
    
    def calc_perimeter(self):
        data = self.scale_data.copy()
        
        
        #data['lenght'] = 0
        #data.loc[0, 'lenght'] = 0
        #for i in range(len(data.index)-1):       
            #data.loc[i+1, 'lenght'] = data.iloc[i]['lenght'] + math.dist((data.iloc[i]['x'], data.iloc[i]['y']), (data.iloc[i+1]['x'], data.iloc[i+1]['y']))
        points = data[['x', 'y']].to_numpy()
        points = np.append(points,points[:1], axis=0)
        
        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
        
        return distance[-1]
    
    def move_start(self):
        data = self.data.copy()
        
        data['x'] = data['x'] - self.offset_x
        data['y'] = data['y'] - self.offset_y
        
        points = data[['x', 'y']].to_numpy()
        points = np.append(points,points[:1], axis=0)
        a = points[:,1]-self.dim[1]/2
        zero_crossings = np.where(np.diff(np.sign(a)))[0]
        print(zero_crossings)
        roll = zero_crossings[0]

        for zero in zero_crossings:
            print(points[zero, 0])
            if points[zero, 0] < points[roll, 0]:
                roll = zero
        
        print(points)
        print(roll)
        points_rolled = np.roll(points, -roll-1, axis = 0)
        print(points_rolled)
        
        data_rolled_rolled = pd.DataFrame(points_rolled, columns = ['x', 'y'])
        self.data[['x', 'y']] = data_rolled_rolled[['x','y']]
        
        
    def load_data(self, name):
        df = pd.DataFrame(columns=['x', 'y', 'c'])
        try:
            data = pd.read_csv(name, index_col=None)
            df[['x', 'y']] = data[['X','Y']]
            print(df)
        except:
            pass
        
        return df
    
    def rotate(self):
        data1 = self.scale_data.copy()
        data = self.scale_data.copy()
        ox, oy = self.dim[0]/2, self.dim[1]/2
 
        data1['x'] = ox + math.cos(self.angle) * (data['x'] - ox) - math.sin(self.angle) * (data['y'] - oy)
        data1['y'] = oy + math.sin(self.angle) * (data['x'] - ox) + math.cos(self.angle) * (data['y'] - oy)
         
        self.scale_data[['x', 'y']] = data1[['x','y']]
        #return data
          
    def filter_outliers(self, df, col_name):
        q_low = df[col_name].quantile(0.01)
        q_hi  = df[col_name].quantile(0.99)
        
        df_filtered = df[(df[col_name] < q_hi) & (df[col_name] > q_low)]
 
        return df_filtered
    
    def moments(self, data):
        print(data)
        imageMoments = cv2.moments(np.array(data[['x', 'y']]).astype('int32'))
        print(imageMoments)
        cx = int(imageMoments['m10']/imageMoments['m00'])
        cy = int(imageMoments['m01']/imageMoments['m00'])
        area = imageMoments['m00']
        
       
        area_c = cv2.contourArea(np.array(data[['x', 'y']]).astype('int32'))
       
        
        hull = cv2.convexHull(np.array(data[['x', 'y']]).astype('int32'))
        hullArea = cv2.contourArea(hull)
        solidity = area_c / float(hullArea)
       
        
        
        mu_20 = imageMoments['m20']/imageMoments['m00'] - cx**2
        mu_02 = imageMoments['m20']/imageMoments['m00'] - cy**2
        mu_11 = imageMoments['m11']/imageMoments['m00'] - cx*cy
        
        cov = np.asarray([[mu_20, mu_11],[mu_11, mu_02]])
        eigvalues, eigvectors = np.linalg.eig(cov)
        eigvec_1, eigvec_2 = eigvectors[:,0], eigvectors[:,1]
        theta = np.arctan2(eigvec_1[1], eigvec_1[0])
        
        #theta = 1/2 * np.arctan((2 * mu_11) / (mu_20 - mu_02))
        angle = np.rad2deg(theta)
        if angle > 90:
            angle = angle - 180
        #print(angle)
        return area, cx, cy, solidity
    

    def offsets(self):
        return int(self.cx - self.dim[0]/2), int(self.cy - self.dim[1]/2)
        


    def make_binary(self, center = True):
        pts = self.data[['x', 'y']].to_numpy()
        pts = tuple(map(tuple, pts))
        
        out = Image.new("1", self.dim, 0)
        draw = ImageDraw.Draw(out)
        draw.polygon(pts, fill = 1)
        fin = Image.Image.rotate(out, self.angle)
        #plt.imshow(fin)
        #plt.show()
        
        return fin
    
    def curvature(self, data):
        
        points = data[['x', 'y']].to_numpy()
        points = np.append(points,points[:1], axis=0)
        
        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
        distance = distance/distance[-1]
        
        dx_dt = np.gradient(data['x'].to_numpy())
        dy_dt = np.gradient(data['y'].to_numpy())
    
        dx_dtdt = np.gradient(dx_dt)
        dy_dtdt = np.gradient(dy_dt)
    
        curv_up = dx_dt*dy_dtdt - dy_dt*dx_dtdt
        curv_down = np.power(dx_dt*dx_dt + dy_dt*dy_dt, 1.5)
    
        curv = np.divide(curv_up,curv_down)
        data['c'] = -curv
        data['u'] = distance
        
        return data
    
    def smooth(self, data, k=3, s= 0.2):
        
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
        alpha = np.linspace(0, 1, self.res)

        points_fitted = np.stack([spl(alpha) for spl in splines], axis=1)
        #np.stack([np.ones(3) for _ in range(3)]) 

        print(points_fitted)
        new_data = pd.DataFrame(points_fitted, columns = ['x', 'y'])
        
        return new_data
        
    def hist(self, center = True, norm = True):
        img_bin = self.make_binary(center)
            
        pix = np.array(img_bin)
    
        y_range, x_range = pix.shape
        x_hist = np.empty(x_range)
        y_hist = np.empty(y_range)
        s_x = 0
        s_y = 0
        for i in range(x_range):
            x_hist[i] = np.count_nonzero(pix[:,i])
            s_x += x_hist[i]
            
        for j in range(y_range):
            y_hist[j] = np.count_nonzero(pix[j,:])
            s_y += y_hist[j]
            
        if norm:
            x_hist = x_hist / s_x
            y_hist = y_hist / s_y
            
        return x_hist, y_hist     
    
    
    def scale_bar(self, out, size, offset):
        draw = ImageDraw.Draw(out)
        width = 50
        draw.rectangle([(offset, self.dim[1]-offset-size), (offset + width, self.dim[1]-offset)], fill=(0,0,0), width = 1)
        
        return out
    
    def make_img(self, w, colormap, c_min, c_max, auto, c_day = None, alpha = None):
        scale = 1
        r = 20
        flag_30 = False
        
        out = Image.new("RGBA", (self.dim[0], self.dim[1]), (255, 255, 255,0))
        if auto:
            c_min = self.data['c'].min()
            c_max = self.data['c'].max()

        
        print(self.data)
        self.COL = MplColorHelper(colormap, c_min, c_max)

        draw = ImageDraw.Draw(out)
        for i in range(self.data.shape[0]-1):
            x1, y1, c1, u1 = self.data.iloc[i]
            x2, y2, c2, u2 = self.data.iloc[i+1]
            
            if c_day is None:
                color = tuple([int(z * 255) for z in self.COL.get_rgb((c1+c2)/2)])
            else:
                color = c_day
            if i == 0:    
                draw.ellipse([(int(x1/scale)-r, int(y1/scale)-r), (int(x1/scale)+r, int(y1/scale)+r)], fill=(0,0,0), width = 1)
            if (u1 > 0.3) & (flag_30 == False):    
                draw.rectangle([(int(x1/scale)-r, int(y1/scale)-r), (int(x1/scale)+r, int(y1/scale)+r)], fill=(0,0,0), width = 1)
                flag_30 = True
            
            draw.line([(int(x1/scale), int(y1/scale)), (int(x2/scale), int(y2/scale))], fill=color, width = w, joint = 'curve')
        
        x1, y1, c1, u1 = self.data.iloc[-1]
        x2, y2, c2, u2 = self.data.iloc[0]
        draw.line([(int(x1/scale), int(y1/scale)), (int(x2/scale), int(y2/scale))], fill=color, width = w, joint = 'curve')
        
                
        
        out = self.scale_bar(out, 300, 100)
        sc = 100
        #draw.line = ([(int(self.dim[0]/2), int(self.dim[1]/2)), (int(dim[0]/2+sc*math.cos(self.angle)), int(dim[1]/2+sc*math.sin(self.angle)))], fill='tab:red', width = 20)
        out = out.rotate(90)
        #out.show()
        #out = ImageOps.flip(out)
        
        return out
        
      
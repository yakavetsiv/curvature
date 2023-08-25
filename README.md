# Curvature analysis script

Image processing script for calculation of local curvature and growth of tumoroids based on their shape. 
- [Installation](#installation)
- [Input data](#input-data)
- [Running the script](#running-the-script)
  * [Curvature analysis for single shape](#curvature-analysis-for-single-shape)
  * [Growth kinetics](#growth-kinetics)



# Installation
```
pip install -r requirements.txt
```
# Input data
Script requires *.csv files with the coordinates of the shapes. The shape should be closed. Please see exmaple in /examples

# Running the script
```
python curvature_GUI_stats.py
```
## Curvature analysis for single shape
Curvature analysis for single shape:
1.	Browse - Select the folder with the example file
2.	Select the *.csv datafile with the coordinates of shape
3.	Verify the import parameters (default: 1608x1608 px). If the datafile name fits the pattern (* 3.csv, where 3 is the day), the day will be recognized automatically
4.	Add the curve
   Note: If you want to change some of the settings, please enter the updated values and press "Update settings" button
5. Press "Save data" to save the analyzed datasets.
6. Press "Save image" to save the top left panel (color mapping the shape with curvature values).
7. Press "Save plot" to save all the generated layots of panels.
8. "Clear last" or "Clear all" to remove the last/all shapes

## Growth kinetics
Estimation of the absolute growth of tumoroid (requires two or more shapes):
1. Add multiple shapes assigning various days for each shape
   Note: If you want different colors for each day instead of color mapping, check in "DAYS" . The order of the addiing of files IS IMPROTANT!
2. Once two or more shapes were added, the script will automatically calculate the growth for each shape point (based on the resolution), splitting samples in pairs based on the set "day" property
  Note: If you want to change some of the settings, please enter the updated values and press "Update settings" button. The changes will be applied for all loaded shapes
3. Press "Save data" to save the analyzed datasets.
4. Press "Save image" to save the top left panel (color mapping the shape with curvature values).
5. Press "Save plot" to save all the generated layots of panels.
6. "Clear last" or "Clear all" to remove the last/all shapes

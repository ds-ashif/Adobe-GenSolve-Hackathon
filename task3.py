import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

def preview_csv(file_path):
    df = pd.read_csv(file_path)
    print("Column names in CSV file:")
    print(df.columns.tolist())
    print("\nFirst few rows of the CSV file:")
    print(df.head())

csv_file_path = 'frag0.csv'
preview_csv(csv_file_path)




def load_curves_from_csv(file_path):
    df = pd.read_csv(file_path)
    curves = {}
    
    for curve_id in df['id'].unique():  
        curve_data = df[df['id'] == curve_id]  
        points = curve_data[['x_coord', 'y_coord']].values  
        curves[curve_id] = points
    
    return curves

def detect_gaps(curves, gap_threshold=5.0):
    gaps = []
    
    curve_ids = sorted(curves.keys())
    
    for i in range(len(curve_ids) - 1):
        curve1 = curves[curve_ids[i]]
        curve2 = curves[curve_ids[i + 1]]
        
        end_of_curve1 = curve1[-1]
        start_of_curve2 = curve2[0]
        distance = np.linalg.norm(np.array(end_of_curve1) - np.array(start_of_curve2))
        
        if distance > gap_threshold:
            gaps.append(((curve_ids[i], len(curve1) - 1), (curve_ids[i + 1], 0)))
    
    return gaps

def interpolate_curve(segment1, segment2):
    x1, y1 = segment1[-1]
    x2, y2 = segment2[0]
    
    num_points = 100
    interpolated_x = np.linspace(x1, x2, num=num_points)
    interpolated_y = np.linspace(y1, y2, num=num_points)
    
    complete_curve = np.column_stack((interpolated_x, interpolated_y))
    return complete_curve

def smooth_curve(curve, s=0.1):
    x = curve[:, 0]
    y = curve[:, 1]
    
    spline_x = UnivariateSpline(range(len(x)), x, s=s)
    spline_y = UnivariateSpline(range(len(y)), y, s=s)
    
    smoothed_x = spline_x(range(len(x)))
    smoothed_y = spline_y(range(len(y)))
    
    smoothed_curve = np.column_stack((smoothed_x, smoothed_y))
    return smoothed_curve

def complete_curves(curves, gap_threshold=5.0):
    gaps = detect_gaps(curves, gap_threshold)
    completed_curves = []
    
    for curve_id, curve in curves.items():
        completed_curves.append(curve)
        if any([curve_id == gap[0][0] and gap[0][1] == len(curves[curve_id]) - 1 for gap in gaps]):
            next_curve_id = [gap[1][0] for gap in gaps if gap[0][0] == curve_id and gap[0][1] == len(curves[curve_id]) - 1][0]
            segment1 = curves[curve_id]
            segment2 = curves[next_curve_id]
            interpolated_curve = interpolate_curve(segment1, segment2)
            completed_curves.append(interpolated_curve)
    
    smoothed_curves = [smooth_curve(curve) for curve in completed_curves]
    
    return smoothed_curves

def plot_curves(curves):
    plt.figure()
    for curve in curves:
        plt.plot(curve[:, 0], curve[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Completed Curves')
    plt.show()

def main(csv_file_path):
   
    curves = load_curves_from_csv(csv_file_path)
    
  
    completed_curves = complete_curves(curves)
    
    
    plot_curves(completed_curves)

if __name__ == "__main__":
   
    csv_file_path = 'frag0.csv'
    main(csv_file_path)

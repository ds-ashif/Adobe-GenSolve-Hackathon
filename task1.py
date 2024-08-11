import numpy as np
import matplotlib.pyplot as plt
import svgwrite
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import csv

def smooth_path(path):
    x = [p[0] for p in path]
    y = [p[1] for p in path]
    x_smooth = savgol_filter(x, 11, 3)
    y_smooth = savgol_filter(y, 11, 3)
    return list(zip(x_smooth, y_smooth))


def read_path_from_csv(filename):
    path = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                if len(row) >= 2:
                    x, y = map(float, row[:2])
                    path.append((x, y))
                else:
                    print(f"Skipping row with insufficient columns: {row}")
            except ValueError as e:
                print(f"ValueError: {e} - Skipping row: {row}")
    return path

def detect_straight_lines(path):
    lines = []
    current_line = [path[0]]
    for i in range(1, len(path) - 1):
        p1, p2, p3 = path[i-1], path[i], path[i+1]
        if is_collinear(p1, p2, p3):
            current_line.append(p2)
        else:
            current_line.append(p2)
            lines.append(current_line)
            current_line = [p2]
    current_line.append(path[-1])
    lines.append(current_line)
    return lines

def is_collinear(p1, p2, p3, tolerance=1e-5):
    return abs((p2[1] - p1[1]) * (p3[0] - p2[0]) - (p3[1] - p2[1]) * (p2[0] - p1[0])) < tolerance


def fit_circle(x, y):
    def circle_model(params, x, y):
        xc, yc, R = params
        return np.sqrt((x - xc)**2 + (y - yc)**2) - R

    def residuals(params, x, y):
        return circle_model(params, x, y)

    center_estimate = np.mean(x), np.mean(y), np.mean(np.sqrt((x - np.mean(x))**2 + (y - np.mean(y))**2))
    

    try:
        params, _ = curve_fit(lambda x, xc, yc, R: residuals([xc, yc, R], x, y), x, y, p0=center_estimate)
        xc, yc, R = params
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        xc, yc, R = np.mean(x), np.mean(y), np.mean(np.sqrt((x - np.mean(x))**2 + (y - np.mean(y))**2))

    return xc, yc, R

def detect_circles_and_ellipses(path):
    x = np.array([p[0] for p in path])
    y = np.array([p[1] for p in path])
    xc, yc, R = fit_circle(x, y)
    return [(xc + R * np.cos(theta), yc + R * np.sin(theta)) for theta in np.linspace(0, 2 * np.pi, len(path))]


def fit_bezier_curves(path):
    n = len(path) - 1
    curves = []
    for i in range(n):
        if i + 3 < len(path):
            p0 = path[i]
            p3 = path[i+3]
            p1 = (2 * p0[0] + p3[0]) / 3, (2 * p0[1] + p3[1]) / 3
            p2 = (p0[0] + 2 * p3[0]) / 3, (p0[1] + 2 * p3[1]) / 3
            curves.append((p0, p1, p2, p3))
    return curves


def bezier_to_svg_path(bezier_curves):
    path = ""
    for curve in bezier_curves:
        p0, p1, p2, p3 = curve
        path_segment = f"M {p0[0]},{p0[1]} C {p1[0]},{p1[1]} {p2[0]},{p2[1]} {p3[0]},{p3[1]} "
        path += path_segment
        print(f"Path segment: {path_segment}")  
    return path

def generate_svg(bezier_curves, filename="output.svg"):
    dwg = svgwrite.Drawing(filename, profile='tiny')
    path_data = bezier_to_svg_path(bezier_curves)
    dwg.add(dwg.path(d=path_data, stroke="black", fill="none"))
    dwg.save()


def main():
    # Path to  CSV file
    csv_filename = 'frag0.csv'
    
    # Reading the path data from the CSV file
    path = read_path_from_csv(csv_filename)
    
    # Smooth the path
    smoothed_path = smooth_path(path)
    
    # Detect and regularize shapes
    straight_lines = detect_straight_lines(smoothed_path)
    circles = detect_circles_and_ellipses(smoothed_path)
    
    # Combine the regularized shapes
    regularized_path = straight_lines + [circles]
    
    # Fit Bezier curves to the regularized path
    bezier_curves = []
    for segment in regularized_path:
        bezier_curves.extend(fit_bezier_curves(segment))
    
    # Generate SVG
    generate_svg(bezier_curves)
    
    # Plot for visualization
    plt.figure(figsize=(8, 8))
    for segment in regularized_path:
        segment = np.array(segment)
        plt.plot(segment[:, 0], segment[:, 1], 'ro-')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()

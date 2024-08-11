import csv
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit
from itertools import combinations
from math import sqrt


def parse_path_segment(segment):
    commands = segment.split(' ')
    coords = []

    for command in commands:
        if command in 'MLC': 
            continue
        try:
            x, y = map(float, command.split(','))
            coords.append((x, y))
        except ValueError:
            print(f"Skipping invalid command: {command}")
            continue

    return np.array(coords)

def smooth_path(coords, window_size=5):
    smoothed_coords = []
    for i in range(len(coords)):
        start = max(0, i - window_size // 2)
        end = min(len(coords), i + window_size // 2 + 1)
        smoothed_coords.append(np.mean(coords[start:end], axis=0))
    return np.array(smoothed_coords)


def fit_circle(x, y):
    def residuals(params, x, y):
        xc, yc, R = params
        return np.sqrt((x - xc)**2 + (y - yc)**2) - R

 
    x_m = np.mean(x)
    y_m = np.mean(y)
    R_m = np.sqrt((x - x_m)**2 + (y - y_m)**2).mean()
    
    center_estimate = x_m, y_m, R_m
    
   
    params, _ = curve_fit(lambda x, y, xc, yc, R: residuals([xc, yc, R], x, y), x, y, p0=center_estimate)

    return params


def detect_circles_and_ellipses(coords, threshold=0.01):
    circles = []
    ellipses = []

    hull = ConvexHull(coords)
    hull_coords = coords[hull.vertices]

    x = hull_coords[:, 0]
    y = hull_coords[:, 1]

    try:
        xc, yc, R = fit_circle(x, y)
        residuals = np.sqrt((x - xc)**2 + (y - yc)**2) - R

        if np.all(np.abs(residuals) < threshold):
            circles.append((xc, yc, R))
        else:
            ellipses.append(hull_coords)

    except Exception as e:
        print(f"Error in circle/ellipse detection: {e}")

    return circles, ellipses


def check_symmetry(coords, line_point1, line_point2):
    def reflection(point, line_point1, line_point2):
        x0, y0 = point
        x1, y1 = line_point1
        x2, y2 = line_point2
        dx = x2 - x1
        dy = y2 - y1
        a = dy / dx
        b = -1
        c = (y1 - a * x1)
        d = (x0 + (y0 - a * x0 - c) * a / (a**2 + 1))
        e = a * d + c
        return (2 * d - x0, 2 * e - y0)
    
    symmetric_pairs = []
    for i, p1 in enumerate(coords):
        for j, p2 in enumerate(coords):
            if i >= j:
                continue
            reflected = reflection(p1, line_point1, line_point2)
            if np.allclose(reflected, p2):
                symmetric_pairs.append((p1, p2))

    return symmetric_pairs

def main():
   
    with open('frag1.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        path_segments = [row[0] for row in csv_reader]

    for segment in path_segments:
        coords = parse_path_segment(segment)
        if coords.size > 0:
            smoothed_path = smooth_path(coords)

           
            line_point1 = np.min(smoothed_path, axis=0)
            line_point2 = np.max(smoothed_path, axis=0)

            print(f"Path Segment: {segment}")
            print(f"Smoothed Path Coordinates: \n{smoothed_path}")

            symmetric_pairs = check_symmetry(smoothed_path, line_point1, line_point2)
            if symmetric_pairs:
                print(f"Symmetric Pairs Detected:")
                for p1, p2 in symmetric_pairs:
                    print(f"  Point 1: {p1}, Point 2: {p2}")
            else:
                print("No symmetric pairs detected.")

            circles, ellipses = detect_circles_and_ellipses(smoothed_path)

            if circles:
                print(f"Circles Detected:")
                for xc, yc, R in circles:
                    print(f"  Center: ({xc}, {yc}), Radius: {R}")
            else:
                print("No circles detected.")

            if ellipses:
                print(f"Ellipses Detected:")
                for ellipse in ellipses:
                    print(f"  Ellipse Points: \n{ellipse}")
            else:
                print("No ellipses detected.")

        else:
            print("No valid coordinates found in this segment.")

if __name__ == "__main__":
    main()

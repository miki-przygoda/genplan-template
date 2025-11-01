from datasets.packaged_modules.imagefolder.imagefolder import IMAGE_EXTENSIONS
from preprocessing_data_inspect import load_floorplans_dataset

import random
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def extract_room_polygons(pil_image, min_area_ratio: float = 0.001, corner_quality: float = 0.01):
    """Return filled room mask, room contours, metadata, and snapped corners."""

    # Convert PIL image to OpenCV-friendly arrays
    rgb_image = np.array(pil_image.convert("RGB"))
    gray = cv.cvtColor(rgb_image, cv.COLOR_RGB2GRAY)

    # Enhance structural edges and connect potential gaps in walls
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 40, 120)

    wall_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    thick_walls = cv.dilate(edges, wall_kernel, iterations=2)
    thick_walls = cv.morphologyEx(thick_walls, cv.MORPH_CLOSE, wall_kernel, iterations=2)

    # Invert to isolate open space, then flood fill the exterior to remove background
    open_space = cv.bitwise_not(thick_walls)
    floodfilled = open_space.copy()
    height, width = open_space.shape
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv.floodFill(floodfilled, flood_mask, (0, 0), 0)

    room_mask = cv.morphologyEx(floodfilled, cv.MORPH_OPEN, wall_kernel, iterations=2)

    contours, _ = cv.findContours(room_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * (height * width)

    filled_mask = np.zeros_like(room_mask)
    polygons = []
    metadata = []

    corner_candidates = cv.goodFeaturesToTrack(
        thick_walls,
        maxCorners=500,
        qualityLevel=corner_quality,
        minDistance=10,
        blockSize=7,
        useHarrisDetector=True,
        k=0.04,
    )

    deduplication_threshold = 6
    unique_corners = []
    if corner_candidates is not None:
        for candidate in corner_candidates:
            x, y = candidate.ravel()
            x, y = int(x), int(y)
            matched = False
            for stored in unique_corners:
                if abs(stored[0] - x) <= deduplication_threshold and abs(stored[1] - y) <= deduplication_threshold:
                    stored[0] = int((stored[0] * stored[2] + x) / (stored[2] + 1))
                    stored[1] = int((stored[1] * stored[2] + y) / (stored[2] + 1))
                    stored[2] += 1
                    matched = True
                    break
            if not matched:
                unique_corners.append([x, y, 1])

    corner_points = [(corner[0], corner[1]) for corner in unique_corners]
    for contour in contours:
        area = cv.contourArea(contour)
        if area < min_area:
            continue

        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        polygon = [(int(point[0][0]), int(point[0][1])) for point in approx]

        cv.drawContours(filled_mask, [contour], contourIdx=-1, color=255, thickness=cv.FILLED)

        moments = cv.moments(contour)
        if moments["m00"] != 0:
            centroid = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
        else:
            centroid = (0, 0)

        corner_ids = []
        if corner_points:
            for vertex in polygon:
                distances = [np.linalg.norm(np.array(vertex) - np.array(corner)) for corner in corner_points]
                nearest = int(np.argmin(distances))
                corner_ids.append(nearest)

        polygons.append(contour)
        metadata.append(
            {
                "room_id": len(metadata),
                "area_px": int(area),
                "polygon": polygon,
                "centroid": centroid,
                "corner_ids": corner_ids,
            }
        )

    return filled_mask, polygons, metadata, corner_points


def colourise_rooms(rgb_image: np.ndarray, room_polygons, corner_points):
    """Create an overlay and blended visualisation for segmented rooms."""

    overlay = np.zeros_like(rgb_image)
    rng = np.random.default_rng(seed=42)

    for index, contour in enumerate(room_polygons):
        colour = rng.integers(0, 255, size=3).tolist()
        cv.fillPoly(overlay, [contour], colour)

    for corner in corner_points:
        cv.circle(overlay, corner, radius=4, color=(255, 255, 255), thickness=-1)

    blended = cv.addWeighted(rgb_image, 0.6, overlay, 0.4, gamma=0)
    return overlay, blended


def plot_results(original: np.ndarray, mask: np.ndarray, overlay: np.ndarray, blended: np.ndarray):
    """Visual helper to check segmentation quality during experimentation."""

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Room Mask")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(overlay)
    plt.title("Room Overlay")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(blended)
    plt.title("Blended")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    os.system("clear")

    dataset = load_floorplans_dataset()
    seed = random.randint(0, len(dataset) - 1)
    print(f"Using sample index: {seed}")

    pil_image = dataset[seed]["image"]
    mask, room_polygons, metadata, corner_points = extract_room_polygons(pil_image)

    rgb_image = np.array(pil_image.convert("RGB"))
    overlay, blended = colourise_rooms(rgb_image, room_polygons, corner_points)

    print("Extracted room metadata:")
    for room in metadata:
        print(room)

    print("Detected corners:")
    for corner_id, point in enumerate(corner_points):
        print({"corner_id": corner_id, "point": point})

    plot_results(rgb_image, mask, overlay, blended)


if __name__ == "__main__":
    main()

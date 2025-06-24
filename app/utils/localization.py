import os
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def localize_kidney(image_path):
    model = YOLO("models/yolov8_localizer.pt")
    results = model(image_path, conf=0.15)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    boxes = []

    if results and results[0].boxes is not None and results[0].boxes.xyxy is not None:
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for coords in xyxy:
            coords = coords.tolist()
            boxes.append(coords)

            # Draw rectangle only (remove label)
            draw.rectangle(coords, outline="red", width=3)

    return boxes, image

def map_coordinates_to_regions(boxes, image_width=512, image_height=512):
    """
    Convert bounding box coordinates to specific directional region names using a 3x3 grid.
    Avoid 'Middle Middle' by using more specific labels like 'Middle Left', 'Middle Right', 'Middle Down'.
    boxes: List of [x_min, y_min, x_max, y_max] for detected regions.
    Returns list of region names (e.g., ['Top Right', 'Middle Left']).
    """
    if not boxes:
        return []
    
    regions = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        # Divide image into a 3x3 grid
        # X-axis: Left (0 to width/3), Middle (width/3 to 2*width/3), Right (2*width/3 to width)
        # Y-axis: Top (0 to height/3), Middle (height/3 to 2*height/3), Bottom (2*height/3 to height)
        if x_center < image_width / 3:
            x_region = "Left"
        elif x_center < 2 * image_width / 3:
            x_region = "Middle"
        else:
            x_region = "Right"

        if y_center < image_height / 3:
            y_region = "Top"
        elif y_center < 2 * image_height / 3:
            y_region = "Middle"
        else:
            y_region = "Bottom"

        # Combine to form region name, avoiding "Middle Middle"
        if x_region == "Middle" and y_region == "Middle":
            # Determine the dominant direction to avoid "Middle Middle"
            # If x_center is closer to the left edge of the middle section, use "Middle Left"
            # If x_center is closer to the right edge, use "Middle Right"
            # If y_center is closer to the bottom edge, use "Middle Down" (interpreted as Bottom Middle)
            middle_x_start = image_width / 3
            middle_x_end = 2 * image_width / 3
            middle_y_start = image_height / 3
            middle_y_end = 2 * image_height / 3

            dist_to_left = x_center - middle_x_start
            dist_to_right = middle_x_end - x_center
            dist_to_bottom = middle_y_end - y_center

            if dist_to_left < dist_to_right and dist_to_left < dist_to_bottom:
                region = "Middle Left"
            elif dist_to_right <= dist_to_left and dist_to_right < dist_to_bottom:
                region = "Middle Right"
            else:
                region = "Middle Down"  # Interpreted as Bottom Middle
        else:
            region = f"{y_region} {x_region}"
        
        if region not in regions:
            regions.append(region)
    
    return regions
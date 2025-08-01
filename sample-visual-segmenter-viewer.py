#!/usr/bin/env python3
"""
Clarifai Visual Segmentation Demo with Polygon Overlay Visualization

This script demonstrates how to use Clarifai's image segmentation model to identify
and analyze different segments/regions within an image. Image segmentation is the
process of partitioning an image into multiple segments or regions, each representing
different objects or parts of objects.

Features:
- Downloads and analyzes images using Clarifai's segmentation API
- Displays detailed text results with confidence scores
- Shows visual comparison between original image and segmented regions
- Prints complete raw JSON response from Clarifai API
- Extracts and displays detailed polygon coordinate data
- Visualizes base64 mask data overlaid on original images (pixel-level segmentation)
- Creates semi-transparent colored polygon overlays for each detected region
- Supports masks, polygons, and bounding box data formats
- Color-coded visualization with legend for easy interpretation
- Preserves original image visibility through transparent overlays
- Labels positioned at polygon centroids for accurate placement
- Exports structured JSON data for further analysis
- Interactive session management with cleanup capabilities

For more information about Clarifai's visual models, visit:
https://docs.clarifai.com/getting-started/quickstart

Author: Clarifai
Last Updated: 2025
Requirements: clarifai>=11.6.0, matplotlib>=3.5.0, Pillow>=10.0.0, requests, opencv-python>=4.5.0
"""

# Import necessary libraries
import base64
import gc  # For garbage collection
import json  # For JSON processing
import os  # For accessing environment variables
import random
import requests  # For downloading images from URLs
from io import BytesIO  # For handling image data in memory

from clarifai.client.model import Model  # Clarifai's Model class for API interactions
from PIL import Image, ImageDraw, ImageFont  # For image processing and visualization
import numpy as np  # For numerical operations

# Configure matplotlib for different environments
import matplotlib
import matplotlib.pyplot as plt  # For displaying images
import matplotlib.patches as patches  # For drawing rectangles on images

# Try to import OpenCV, with fallback if not available
try:
    import cv2
    HAS_OPENCV = True
    print("✅ OpenCV available for mask visualization")
except ImportError:
    HAS_OPENCV = False
    print("⚠️  OpenCV not available - mask visualization will be limited")

# Try to use a GUI backend, fallback to non-interactive if not available
try:
    matplotlib.use('TkAgg')  # Try GUI backend first
except:
    try:
        matplotlib.use('Qt5Agg')  # Try Qt backend
    except:
        matplotlib.use('Agg')  # Use non-interactive backend as fallback
        print("ℹ️  Using non-interactive display mode (images will be saved instead of displayed)")

# =============================================================================
# IMAGE VISUALIZATION FUNCTIONS
# =============================================================================
def debug_region_structure(regions):
    """
    Debug function to inspect the structure of region data from Clarifai.
    This helps understand what data is available for visualization.
    
    Args:
        regions (list): List of region objects from Clarifai prediction
    """
    print(f"\n🔍 DEBUG: Analyzing {len(regions)} regions...")
    
    for i, region in enumerate(regions[:2]):  # Show first 2 regions as examples
        print(f"\n--- Region {i+1} Debug Info ---")
        print(f"Region type: {type(region)}")
        
        # Check region_info
        if hasattr(region, 'region_info'):
            print(f"✅ Has region_info: {type(region.region_info)}")
            
            # Check for mask (primary segmentation data)
            if hasattr(region.region_info, 'mask'):
                mask = region.region_info.mask
                print(f"✅ Has mask: {type(mask)}")
                if hasattr(mask, 'image') and hasattr(mask.image, 'base64'):
                    print(f"   Mask has base64 image data: {len(mask.image.base64)} characters")
                else:
                    print("❌ Mask has no base64 image data")
            else:
                print("❌ No mask attribute")
            
            # Check for polygon (secondary segmentation data)
            if hasattr(region.region_info, 'polygon'):
                polygon = region.region_info.polygon
                if polygon and hasattr(polygon, 'points'):
                    print(f"✅ Has polygon with {len(polygon.points)} points")
                    if len(polygon.points) > 0:
                        first_point = polygon.points[0]
                        print(f"   First point: col={first_point.col:.3f}, row={first_point.row:.3f}")
                else:
                    print("❌ Polygon is None or has no points")
            else:
                print("❌ No polygon attribute")
            
            # Check for bounding_box
            if hasattr(region.region_info, 'bounding_box'):
                bbox = region.region_info.bounding_box
                print(f"✅ Has bounding_box: ({bbox.left_col:.3f}, {bbox.top_row:.3f}) to ({bbox.right_col:.3f}, {bbox.bottom_row:.3f})")
            else:
                print("❌ No bounding_box attribute")
        else:
            print("❌ No region_info attribute")
        
        # Check concepts
        if hasattr(region, 'data') and hasattr(region.data, 'concepts'):
            print(f"✅ Has {len(region.data.concepts)} concepts")
            if len(region.data.concepts) > 0:
                top_concept = region.data.concepts[0]
                print(f"   Top concept: {top_concept.name} ({top_concept.value:.3f})")
        else:
            print("❌ No concepts data")
    
    print("--- End Debug Info ---\n")

def print_all_polygon_data(regions):
    """
    Print detailed region information including masks, polygons, and bounding boxes from Clarifai prediction.
    This shows the complete segmentation data received from the API.
    
    Args:
        regions (list): List of region objects from Clarifai prediction
    """
    print(f"\n📐 DETAILED SEGMENTATION DATA FROM CLARIFAI PREDICTION")
    print("=" * 70)
    print(f"Total regions found: {len(regions)}")
    
    for i, region in enumerate(regions, 1):
        print(f"\n🔷 REGION {i}")
        print("-" * 40)
        
        # Print concept information first
        if hasattr(region, 'data') and hasattr(region.data, 'concepts') and region.data.concepts:
            top_concept = region.data.concepts[0]
            print(f"📊 Primary Concept: {top_concept.name} (confidence: {top_concept.value:.4f})")
            
            # Print all concepts if there are multiple
            if len(region.data.concepts) > 1:
                print(f"📋 All Concepts:")
                for j, concept in enumerate(region.data.concepts):
                    print(f"   {j+1}. {concept.name} (confidence: {concept.value:.4f})")
        else:
            print("📊 Primary Concept: Unknown")
        
        # Check for mask data (primary segmentation format)
        if hasattr(region, 'region_info') and hasattr(region.region_info, 'mask'):
            mask = region.region_info.mask
            if hasattr(mask, 'image') and hasattr(mask.image, 'base64'):
                print(f"🎭 Segmentation Mask: Available")
                print(f"   Base64 data length: {len(mask.image.base64)} characters")
                print(f"   Image format: base64 encoded mask")
                print(f"   💡 Use PIL to decode: Image.open(BytesIO(base64.b64decode(mask.image.base64)))")
            else:
                print("❌ Mask has no base64 image data")
        else:
            print("❌ No mask data available")
        
        # Check for polygon data (alternative/additional format)
        if hasattr(region, 'region_info') and hasattr(region.region_info, 'polygon') and region.region_info.polygon:
            polygon = region.region_info.polygon
            if hasattr(polygon, 'points') and polygon.points:
                points = polygon.points
                print(f"🔺 Polygon Shape: {len(points)} vertices")
                print(f"📝 Raw Polygon Coordinates (normalized 0.0-1.0):")
                
                # Print all polygon points
                for j, point in enumerate(points):
                    print(f"   Point {j+1:2d}: col={point.col:.6f}, row={point.row:.6f}")
                
                # Calculate some basic polygon properties
                if len(points) >= 3:
                    # Calculate approximate area using shoelace formula
                    area = 0.0
                    n = len(points)
                    for j in range(n):
                        k = (j + 1) % n
                        area += points[j].col * points[k].row
                        area -= points[k].col * points[j].row
                    area = abs(area) / 2.0
                    
                    # Calculate approximate perimeter
                    perimeter = 0.0
                    for j in range(n):
                        k = (j + 1) % n
                        dx = points[k].col - points[j].col
                        dy = points[k].row - points[j].row
                        perimeter += (dx*dx + dy*dy)**0.5
                    
                    # Find bounding box of polygon
                    min_col = min(p.col for p in points)
                    max_col = max(p.col for p in points)
                    min_row = min(p.row for p in points)
                    max_row = max(p.row for p in points)
                    
                    print(f"📏 Calculated Properties:")
                    print(f"   Area (normalized): {area:.6f}")
                    print(f"   Perimeter (normalized): {perimeter:.6f}")
                    print(f"   Bounding Box: ({min_col:.4f}, {min_row:.4f}) to ({max_col:.4f}, {max_row:.4f})")
                    print(f"   Width: {max_col - min_col:.4f}, Height: {max_row - min_row:.4f}")
                
            else:
                print("❌ Polygon has no points")
        else:
            print("❌ No polygon data available")
        
        # Check for bounding box as fallback
        if hasattr(region, 'region_info') and hasattr(region.region_info, 'bounding_box'):
            bbox = region.region_info.bounding_box
            print(f"📦 Bounding Box:")
            print(f"   Top-left: ({bbox.left_col:.6f}, {bbox.top_row:.6f})")
            print(f"   Bottom-right: ({bbox.right_col:.6f}, {bbox.bottom_row:.6f})")
            print(f"   Width: {bbox.right_col - bbox.left_col:.6f}")
            print(f"   Height: {bbox.bottom_row - bbox.top_row:.6f}")
            print(f"   Area: {(bbox.right_col - bbox.left_col) * (bbox.bottom_row - bbox.top_row):.6f}")
        else:
            print("❌ No bounding box data available")
    
    print(f"\n" + "=" * 70)
    print("💡 Notes:")
    print("   • Coordinates are normalized (0.0 to 1.0) relative to image dimensions")
    print("   • col = x-coordinate (horizontal), row = y-coordinate (vertical)")
    print("   • To convert to pixels: x_pixel = col * image_width, y_pixel = row * image_height")
    print("   • Masks are base64-encoded binary images showing pixel-level segmentation")
    print("   • Polygons are simplified boundary representations of the segmented regions")
    print("   • Bounding boxes provide rectangular approximations of the regions")
    print("=" * 70)

def print_raw_prediction_json(model_prediction, output_filename="raw_prediction.json"):
    """
    Print and save the raw JSON response from Clarifai prediction.
    This shows the complete API response structure for debugging and analysis.
    
    Args:
        model_prediction: Raw prediction object from Clarifai API
        output_filename (str): Name of the output JSON file for raw response
    """
    import json
    from datetime import datetime
    
    print(f"\n🔍 RAW CLARIFAI PREDICTION JSON RESPONSE")
    print("=" * 70)
    
    try:
        # Convert the prediction object to dictionary/JSON format
        # The prediction object has a to_dict() method for serialization
        if hasattr(model_prediction, 'to_dict'):
            prediction_dict = model_prediction.to_dict()
        else:
            # Fallback: try to access the raw response if available
            prediction_dict = {
                "message": "Raw prediction object doesn't have to_dict() method",
                "type": str(type(model_prediction)),
                "available_attributes": [attr for attr in dir(model_prediction) if not attr.startswith('_')]
            }
        
        # Pretty print the JSON response to console
        json_str = json.dumps(prediction_dict, indent=2, ensure_ascii=False)
        print("📄 Complete API Response Structure:")
        print("-" * 50)
        print(json_str)
        
        # Save to file with metadata
        output_data = {
            "metadata": {
                "extraction_timestamp": datetime.now().isoformat(),
                "clarifai_model_prediction": "raw_api_response",
                "note": "This is the complete JSON response from Clarifai's prediction API"
            },
            "prediction_response": prediction_dict
        }
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Raw prediction JSON saved to: {output_filename}")
        print(f"📊 Response contains complete API structure including metadata, regions, and concepts")
        
    except Exception as e:
        print(f"❌ Error processing prediction JSON: {e}")
        print("🔄 Attempting alternative approach...")
        
        # Alternative approach: try to serialize key attributes
        try:
            basic_info = {
                "prediction_type": str(type(model_prediction)),
                "has_outputs": hasattr(model_prediction, 'outputs'),
                "outputs_count": len(model_prediction.outputs) if hasattr(model_prediction, 'outputs') else 0,
                "error_details": str(e)
            }
            
            if hasattr(model_prediction, 'outputs') and model_prediction.outputs:
                output = model_prediction.outputs[0]
                basic_info["first_output"] = {
                    "has_data": hasattr(output, 'data'),
                    "regions_count": len(output.data.regions) if hasattr(output, 'data') and hasattr(output.data, 'regions') else 0
                }
            
            print("📄 Basic Prediction Information:")
            print(json.dumps(basic_info, indent=2))
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(basic_info, f, indent=2)
            
            print(f"💾 Basic prediction info saved to: {output_filename}")
            
        except Exception as e2:
            print(f"❌ Could not extract prediction information: {e2}")

def process_mask_data(regions):
    """
    Process mask data from regions and provide instructions for visualization.
    
    Args:
        regions (list): List of region objects from Clarifai prediction
    """
    print(f"\n🎭 MASK DATA PROCESSING")
    print("=" * 50)
    
    mask_count = 0
    for i, region in enumerate(regions, 1):
        if hasattr(region, 'region_info') and hasattr(region.region_info, 'mask'):
            mask = region.region_info.mask
            if hasattr(mask, 'image') and hasattr(mask.image, 'base64'):
                mask_count += 1
                concept_name = "Unknown"
                if hasattr(region, 'data') and hasattr(region.data, 'concepts') and region.data.concepts:
                    concept_name = region.data.concepts[0].name
                
                print(f"🎭 Region {i} ({concept_name}): Mask available")
                print(f"   Base64 length: {len(mask.image.base64)} characters")
    
    if mask_count > 0:
        print(f"\n✅ Found {mask_count} regions with mask data")
        print(f"\n💡 To visualize masks in your own code:")
        print(f"```python")
        print(f"import base64")
        print(f"from PIL import Image")
        print(f"from io import BytesIO")
        print(f"")
        print(f"# For each region with mask:")
        print(f"mask_data = region.region_info.mask.image.base64")
        print(f"mask_image = Image.open(BytesIO(base64.b64decode(mask_data)))")
        print(f"mask_array = np.array(mask_image)")
        print(f"```")
        print(f"\n🎨 Mask visualization tips:")
        print(f"   • Masks are binary images (0 = background, 255 = object)")
        print(f"   • Use mask_array > 0 to get boolean mask")
        print(f"   • Apply masks to original image for precise segmentation")
        print(f"   • Combine multiple masks with different colors for overlay")
    else:
        print("❌ No mask data found in any regions")
        print("💡 This model may provide polygon or bounding box data instead")

def save_polygon_data_to_json(regions, image_filepath, output_filename="segmentation_data.json"):
    """
    Save complete segmentation data (masks, polygons, bounding boxes) to JSON file for further analysis.
    
    Args:
        regions (list): List of region objects from Clarifai prediction
        image_filepath (str): Path to the original image file
        output_filename (str): Name of the output JSON file
    """
    import json
    from datetime import datetime
    
    # Prepare data structure
    segmentation_data = {
        "metadata": {
            "image_file": image_filepath,
            "extraction_timestamp": datetime.now().isoformat(),
            "total_regions": len(regions),
            "coordinate_system": "normalized (0.0 to 1.0)",
            "data_types": ["masks", "polygons", "bounding_boxes"],
            "note": "Complete segmentation data from Clarifai visual segmentation model"
        },
        "regions": []
    }
    
    for i, region in enumerate(regions, 1):
        region_data = {
            "region_id": i,
            "concepts": [],
            "segmentation": {}
        }
        
        # Extract concepts
        if hasattr(region, 'data') and hasattr(region.data, 'concepts'):
            for concept in region.data.concepts:
                region_data["concepts"].append({
                    "name": concept.name,
                    "confidence": float(concept.value)
                })
        
        # Extract mask data (primary segmentation format)
        if hasattr(region, 'region_info') and hasattr(region.region_info, 'mask'):
            mask = region.region_info.mask
            if hasattr(mask, 'image') and hasattr(mask.image, 'base64'):
                region_data["segmentation"]["mask"] = {
                    "type": "base64_image",
                    "data_length": len(mask.image.base64),
                    "format": "base64 encoded binary mask",
                    "decode_instructions": "Use PIL: Image.open(BytesIO(base64.b64decode(mask_data)))",
                    # Note: base64_data excluded from JSON due to size - use API directly for processing
                }
        
        # Extract polygon data
        if hasattr(region, 'region_info') and hasattr(region.region_info, 'polygon') and region.region_info.polygon:
            polygon = region.region_info.polygon
            if hasattr(polygon, 'points') and polygon.points:
                points = []
                for point in polygon.points:
                    points.append({
                        "col": float(point.col),
                        "row": float(point.row)
                    })
                
                region_data["segmentation"]["polygon"] = {
                    "type": "polygon",
                    "points": points,
                    "point_count": len(points)
                }
                
                # Calculate properties
                if len(points) >= 3:
                    # Area calculation
                    area = 0.0
                    n = len(points)
                    for j in range(n):
                        k = (j + 1) % n
                        area += points[j]["col"] * points[k]["row"]
                        area -= points[k]["col"] * points[j]["row"]
                    area = abs(area) / 2.0
                    
                    # Bounding box
                    cols = [p["col"] for p in points]
                    rows = [p["row"] for p in points]
                    
                    region_data["segmentation"]["polygon"]["properties"] = {
                        "area_normalized": area,
                        "bounding_box": {
                            "min_col": min(cols),
                            "max_col": max(cols),
                            "min_row": min(rows),
                            "max_row": max(rows),
                            "width": max(cols) - min(cols),
                            "height": max(rows) - min(rows)
                        }
                    }
        
        # Extract bounding box data
        if hasattr(region, 'region_info') and hasattr(region.region_info, 'bounding_box'):
            bbox = region.region_info.bounding_box
            region_data["segmentation"]["bounding_box"] = {
                "type": "bounding_box",
                "left_col": float(bbox.left_col),
                "top_row": float(bbox.top_row),
                "right_col": float(bbox.right_col),
                "bottom_row": float(bbox.bottom_row),
                "width": float(bbox.right_col - bbox.left_col),
                "height": float(bbox.bottom_row - bbox.top_row),
                "area_normalized": float((bbox.right_col - bbox.left_col) * (bbox.bottom_row - bbox.top_row))
            }
        
        segmentation_data["regions"].append(region_data)
    
    # Save to JSON file
    try:
        with open(output_filename, 'w') as f:
            json.dump(segmentation_data, f, indent=2)
        print(f"💾 Complete segmentation data saved to: {output_filename}")
        print(f"📊 Data includes {len(regions)} regions with masks, polygons, and bounding boxes")
        print(f"🎭 Mask data: {sum(1 for r in segmentation_data['regions'] if 'mask' in r['segmentation'])} regions")
        print(f"🔺 Polygon data: {sum(1 for r in segmentation_data['regions'] if 'polygon' in r['segmentation'])} regions") 
        print(f"📦 Bounding box data: {sum(1 for r in segmentation_data['regions'] if 'bounding_box' in r['segmentation'])} regions")
    except Exception as e:
        print(f"❌ Error saving segmentation data: {e}")

def download_image(url):
    """
    Download an image from a URL and return it as a PIL Image object.
    
    Args:
        url (str): The URL of the image to download
        
    Returns:
        PIL.Image: The downloaded image
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"❌ Error downloading image: {e}")
        return None

def draw_segmentation_regions(image, regions, show_labels=True):
    """
    Draw colored polygon overlays and labels for segmentation regions on an image.
    
    Args:
        image (PIL.Image): The original image
        regions (list): List of region objects from Clarifai prediction
        show_labels (bool): Whether to show concept labels on the image
        
    Returns:
        tuple: (PIL.Image with regions drawn, list of color mappings)
    """
    # Create a copy of the image to draw on
    img_with_regions = image.copy().convert('RGBA')
    
    # Create a transparent overlay for colored regions
    overlay = Image.new('RGBA', img_with_regions.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Try to use a default font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    # Enhanced color palette with distinct colors for better visibility
    colors = [
        '#FF0000',  # Bright Red
        '#0000FF',  # Bright Blue  
        '#00FF00',  # Bright Green
        '#FF00FF',  # Magenta
        '#00FFFF',  # Cyan
        '#FFA500',  # Orange
        '#800080',  # Purple
        '#FFFF00',  # Yellow
        '#FF69B4',  # Hot Pink
        '#32CD32',  # Lime Green
        '#FFD700',  # Gold
        '#8A2BE2',  # Blue Violet
        '#DC143C',  # Crimson
        '#20B2AA',  # Light Sea Green
        '#FF6347',  # Tomato
        '#4682B4',  # Steel Blue
    ]
    
    # Store color mappings for legend
    color_mappings = []
    
    for i, region in enumerate(regions):
        # Get polygon coordinates if available
        if hasattr(region.region_info, 'polygon') and region.region_info.polygon:
            polygon_points = region.region_info.polygon.points
            
            # Convert normalized polygon coordinates to pixel coordinates
            img_width, img_height = image.size
            pixel_points = []
            
            for point in polygon_points:
                x = int(point.col * img_width)
                y = int(point.row * img_height)
                pixel_points.append((x, y))
            
            # Choose unique color for this region
            color_hex = colors[i % len(colors)]
            
            # Convert hex color to RGB tuple
            color_rgb = tuple(int(color_hex[j:j+2], 16) for j in (1, 3, 5))
            
            # Create semi-transparent colored overlay for this polygon region
            overlay_color = color_rgb + (80,)  # RGBA with 80/255 opacity (~30% transparent)
            
            if len(pixel_points) >= 3:  # Need at least 3 points for a polygon
                overlay_draw.polygon(pixel_points, fill=overlay_color, outline=color_rgb + (200,), width=2)
            
            # Get the top concept for this region
            if region.data.concepts:
                top_concept = region.data.concepts[0]
                concept_name = top_concept.name
                confidence = top_concept.value
                
                # Store color mapping
                color_mappings.append({
                    'region_id': i + 1,
                    'color': color_hex,
                    'concept': concept_name,
                    'confidence': confidence
                })
                
                if show_labels and pixel_points:
                    # Calculate centroid of polygon for label placement
                    centroid_x = sum(p[0] for p in pixel_points) // len(pixel_points)
                    centroid_y = sum(p[1] for p in pixel_points) // len(pixel_points)
                    
                    label = f"R{i+1}: {concept_name} ({confidence:.2f})"
                    
                    # Calculate text size for background
                    text_bbox = overlay_draw.textbbox((centroid_x, centroid_y-15), label, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    # Draw label background with some padding (semi-transparent)
                    bg_color = color_rgb + (180,)  # Semi-transparent background
                    label_x = centroid_x - text_width // 2
                    label_y = centroid_y - 15
                    
                    overlay_draw.rectangle([label_x-4, label_y-4, label_x+text_width+4, label_y+text_height+4], 
                                         fill=bg_color)
                    
                    # Draw label text in white for contrast
                    overlay_draw.text((label_x, label_y), label, fill='white', font=font)
            else:
                # Store color mapping even if no concepts
                color_mappings.append({
                    'region_id': i + 1,
                    'color': color_hex,
                    'concept': 'Unknown',
                    'confidence': 0.0
                })
        
        # Fallback to bounding box if polygon is not available
        elif hasattr(region.region_info, 'bounding_box'):
            bbox = region.region_info.bounding_box
            
            # Convert normalized coordinates to pixel coordinates
            img_width, img_height = image.size
            left = int(bbox.left_col * img_width)
            top = int(bbox.top_row * img_height)
            right = int(bbox.right_col * img_width)
            bottom = int(bbox.bottom_row * img_height)
            
            # Choose unique color for this region
            color_hex = colors[i % len(colors)]
            
            # Convert hex color to RGB tuple
            color_rgb = tuple(int(color_hex[j:j+2], 16) for j in (1, 3, 5))
            
            # Create semi-transparent colored overlay for this region
            overlay_color = color_rgb + (80,)  # RGBA with 80/255 opacity (~30% transparent)
            overlay_draw.rectangle([left, top, right, bottom], fill=overlay_color)
            
            # Draw border around the region
            border_color = color_rgb + (200,)  # More opaque border
            overlay_draw.rectangle([left, top, right, bottom], outline=border_color, width=3)
            
            # Get the top concept for this region
            if region.data.concepts:
                top_concept = region.data.concepts[0]
                concept_name = top_concept.name
                confidence = top_concept.value
                
                # Store color mapping
                color_mappings.append({
                    'region_id': i + 1,
                    'color': color_hex,
                    'concept': concept_name,
                    'confidence': confidence
                })
                
                if show_labels:
                    label = f"R{i+1}: {concept_name} ({confidence:.2f})"
                    
                    # Calculate text size for background
                    text_bbox = overlay_draw.textbbox((left, top-30), label, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    # Draw label background with some padding (semi-transparent)
                    bg_color = color_rgb + (180,)  # Semi-transparent background
                    overlay_draw.rectangle([left-2, top-32, left+text_width+4, top-2], fill=bg_color)
                    
                    # Draw label text in white for contrast
                    overlay_draw.text((left, top-30), label, fill='white', font=font)
            else:
                # Store color mapping even if no concepts
                color_mappings.append({
                    'region_id': i + 1,
                    'color': color_hex,
                    'concept': 'Unknown',
                    'confidence': 0.0
                })
    
    # Composite the overlay onto the original image
    img_with_regions = Image.alpha_composite(img_with_regions, overlay)
    
    # Convert back to RGB for matplotlib compatibility
    img_with_regions = img_with_regions.convert('RGB')
    
    return img_with_regions, color_mappings

def visualize_masks_on_image(image_source, regions, save_path=None, source_type="filepath"):
    """
    Visualize segmentation masks overlaid on the original image using base64 mask data.
    Based on the sample code from Clarifai documentation.
    
    Args:
        image_source (str): Either URL or filepath of the image
        regions (list): List of region objects from Clarifai prediction
        save_path (str, optional): Path to save the visualization. If None, auto-generates.
        source_type (str): Either "url" or "filepath" to indicate image source type
    
    Returns:
        matplotlib.figure.Figure: The created figure object for manual cleanup if needed
    """
    import base64
    import random
    
    print(f"\n🎭 CREATING MASK VISUALIZATION")
    print("=" * 50)
    
    # Check if OpenCV is available
    if not HAS_OPENCV:
        print("❌ OpenCV not available - falling back to polygon visualization")
        return visualize_segmentation_results(image_source, regions, save_path, True, source_type)
    
    # Import OpenCV only if available
    import cv2
    import matplotlib.patches as mpatches
    
    print(f"\n🎭 CREATING MASK VISUALIZATION")
    print("=" * 50)
    
    # Load the original image based on source type
    if source_type == "url":
        print("📥 Downloading image for mask visualization...")
        # Download image for URL
        response = requests.get(image_source)
        img_array = np.array(Image.open(BytesIO(response.content)))
    else:  # filepath
        print("📁 Loading local image for mask visualization...")
        try:
            # Load image using OpenCV for consistency with documentation sample
            img = cv2.imread(image_source)
            img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"✅ Loaded image from: {image_source}")
        except Exception as e:
            print(f"❌ Error loading local image: {e}")
            return None
    
    # Extract masks and concepts from regions using Clarifai's working sample code
    masks = []
    concepts = []
    confidences = []
    
    for region in regions:
        # Filter regions with sufficient confidence (following Clarifai documentation)
        if hasattr(region, 'data') and hasattr(region.data, 'concepts') and region.data.concepts:
            if region.data.concepts[0].value > 0.05:  # Minimum confidence threshold
                # Check if mask data is available
                if hasattr(region, 'region_info') and hasattr(region.region_info, 'mask'):
                    mask = region.region_info.mask
                    if hasattr(mask, 'image') and hasattr(mask.image, 'base64'):
                        # Use Clarifai's recommended approach: PIL directly from base64 without decoding
                        try:
                            # Direct PIL Image loading from base64 data (Clarifai's working method)
                            mask_image = Image.open(BytesIO(mask.image.base64))
                            mask_array = np.array(mask_image)
                            masks.append(mask_array)
                            concepts.append(region.data.concepts[0].name)
                            confidences.append(region.data.concepts[0].value)
                            print(f"✅ Processed mask for: {region.data.concepts[0].name} - Shape: {mask_array.shape}")
                        except Exception as mask_error:
                            print(f"⚠️  Could not process mask for {region.data.concepts[0].name}: {mask_error}")
                            # Skip this mask and continue with others
                            continue
    
    if not masks:
        print("❌ No usable mask data found in regions")
        print("💡 This may be due to incompatible mask format or decoding issues")
        print("🔄 Falling back to bounding box visualization...")
        return visualize_segmentation_results(image_source, regions, save_path, True, source_type)
    
    print(f"🎭 Found {len(masks)} masks to visualize")
    
    # Generate high-contrast colors for each mask for better visibility
    colors = []
    # Use predefined high-contrast colors first, then random if more masks
    high_contrast_colors = [
        (0, 0, 255),      # Bright Red (BGR)
        (255, 0, 0),      # Bright Blue (BGR) 
        (0, 255, 0),      # Bright Green (BGR)
        (255, 255, 0),    # Cyan (BGR)
        (255, 0, 255),    # Magenta (BGR)
        (0, 255, 255),    # Yellow (BGR)
        (128, 0, 255),    # Orange-Red (BGR)
        (255, 128, 0),    # Sky Blue (BGR)
    ]
    
    for i in range(len(masks)):
        if i < len(high_contrast_colors):
            colors.append(high_contrast_colors[i])
        else:
            # Fall back to random bright colors
            r = random.randint(128, 255)  # Ensure bright colors
            g = random.randint(128, 255)
            b = random.randint(128, 255)
            colors.append((b, g, r))  # BGR format for OpenCV compatibility
    
    # Create overlays for each mask
    overlays = []
    for i in range(len(masks)):
        mask = masks[i]
        color = colors[i]
        
        # Create overlay with same dimensions as original image
        overlay = np.zeros_like(img_array)
        overlay[mask > 0] = color  # Apply color where mask is positive
        overlays.append(overlay)
    
    # Overlay masks on original image with enhanced alpha blending for better contrast
    # Only apply high-confidence masks initially (≥0.6)
    overlayed = np.copy(img_array)
    
    for i, overlay in enumerate(overlays):
        if confidences[i] >= 0.6:  # Only show high-confidence masks initially
            # Apply stronger alpha blending (30% overlay, 70% original) for better visibility
            cv2.addWeighted(overlay, 0.30, overlayed, 0.70, 0, overlayed)
    
    # Enhance contrast and brightness more aggressively
    overlayed = cv2.convertScaleAbs(overlayed, alpha=1.6, beta=60)
    
    # Create matplotlib figure with enhanced layout for better legend visibility
    fig = plt.figure(figsize=(24, 10))  # Wider figure to accommodate better legend
    
    # Create a grid layout: 2 large columns for images, 1 smaller for legend
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.4], hspace=0.1, wspace=0.1)
    
    # Original image subplot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_array)
    ax1.set_title('Original Image', fontsize=16, fontweight='bold', pad=15)
    ax1.axis('off')
    
    # Overlaid image subplot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(overlayed)
    ax2.set_title('Image with High-Contrast Segmentation Masks', fontsize=16, fontweight='bold', pad=15)
    ax2.axis('off')
    
    # Legend subplot (dedicated space)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('Interactive Segmentation Legend\n(Click to toggle)', fontsize=14, fontweight='bold', pad=15)
    ax3.axis('off')
    
    # State management for interactive legend
    # Only show masks with confidence >= 0.6 by default
    mask_visibility = [confidences[i] >= 0.6 for i in range(len(overlays))]
    original_overlayed = np.copy(overlayed)   # Store original overlayed image
    
    def update_display():
        """Update the display based on current mask visibility state"""
        # Start with original image
        current_overlayed = np.copy(img_array)
        
        # Apply only visible overlays
        for i, overlay in enumerate(overlays):
            if mask_visibility[i]:
                cv2.addWeighted(overlay, 0.30, current_overlayed, 0.70, 0, current_overlayed)
        
        # Apply contrast enhancement
        current_overlayed = cv2.convertScaleAbs(current_overlayed, alpha=1.6, beta=60)
        
        # Update the display
        ax2.clear()
        ax2.imshow(current_overlayed)
        ax2.set_title('Image with High-Contrast Segmentation Masks', fontsize=16, fontweight='bold', pad=15)
        ax2.axis('off')
        
        # Update legend colors based on visibility state
        update_legend()
        
        # Refresh the figure
        fig.canvas.draw()
    
    def update_legend():
        """Update legend appearance based on mask visibility"""
        ax3.clear()
        ax3.set_title('Interactive Segmentation Legend\n(Click to toggle)', fontsize=14, fontweight='bold', pad=15)
        ax3.axis('off')
        
        # Create interactive legend patches
        legend_patches = []
        for i in range(len(overlays)):
            # Convert BGR to RGB for matplotlib and normalize to 0-1
            color_rgb = [colors[i][2]/255, colors[i][1]/255, colors[i][0]/255]
            concept = concepts[i]
            confidence = confidences[i]
            
            # Modify appearance based on visibility
            if mask_visibility[i]:
                # Visible: full color with "ON" indicator and confidence
                patch = mpatches.Patch(color=color_rgb, label=f"✅ {concept} ({confidence:.2f})")
                alpha = 1.0
            else:
                # Hidden: grayed out with "OFF" indicator and confidence 
                gray_color = [0.7, 0.7, 0.7]  # Light gray
                patch = mpatches.Patch(color=gray_color, label=f"❌ {concept} ({confidence:.2f})")
                alpha = 0.5
            
            legend_patches.append(patch)
        
        # Add legend with interactive properties
        if legend_patches:
            legend = ax3.legend(handles=legend_patches, loc='center left', 
                               fontsize=11, frameon=True, fancybox=True, 
                               shadow=True, borderpad=1, labelspacing=1.5)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.95)
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(1)
            
            # Store legend reference for click handling
            ax3.legend_obj = legend
        
        # Add usage instructions
        ax3.text(0.02, 0.05, "💡 Click legend items\nto toggle masks", 
                fontsize=10, va='bottom', ha='left', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Set legend subplot limits
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
    
    def on_legend_click(event):
        """Handle clicks on legend items to toggle mask visibility"""
        if event.inaxes == ax3:
            # Check if click is within legend area
            if hasattr(ax3, 'legend_obj') and ax3.legend_obj:
                legend = ax3.legend_obj
                
                # Get legend bounding box
                bbox = legend.get_window_extent()
                # Convert to axes coordinates
                bbox_axes = bbox.transformed(ax3.transAxes.inverted())
                
                # Check if click is within legend bounds
                if (bbox_axes.x0 <= event.xdata <= bbox_axes.x1 and 
                    bbox_axes.y0 <= event.ydata <= bbox_axes.y1):
                    
                    # Calculate which legend item was clicked
                    legend_height = bbox_axes.height
                    click_y_rel = (event.ydata - bbox_axes.y0) / legend_height
                    
                    # Reverse the calculation since legend items are top-to-bottom
                    item_index = int((1 - click_y_rel) * len(overlays))
                    
                    # Ensure index is within bounds
                    if 0 <= item_index < len(overlays):
                        # Toggle visibility
                        mask_visibility[item_index] = not mask_visibility[item_index]
                        
                        # Print feedback
                        status = "ON" if mask_visibility[item_index] else "OFF"
                        confidence = confidences[item_index]
                        print(f"🎭 Toggled {concepts[item_index]} ({confidence:.2f}) mask: {status}")
                        
                        # Update display
                        update_display()
    
    # Initialize the legend
    update_legend()
    
    # Connect click event handler
    fig.canvas.mpl_connect('button_press_event', on_legend_click)
    
    # Add main title
    fig.suptitle('Clarifai Interactive Segmentation Masks - Click Legend to Toggle', 
                fontsize=20, fontweight='bold', y=0.95)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Make room for title
    
    # Print enhanced mask information to console
    print("\n🎨 High-Contrast Interactive Mask Overlay Information:")
    print("=" * 70)
    for i, concept in enumerate(concepts):
        color_rgb = f"RGB({colors[i][2]}, {colors[i][1]}, {colors[i][0]})"
        confidence = confidences[i]
        print(f"🎭 Mask {i+1}: {concept} ({confidence:.3f}) - Color: {color_rgb}")
    print(f"🔥 Applied 30% mask opacity with enhanced contrast (alpha=1.6, beta=60)")
    print(f"📊 Interactive legend with click-to-toggle functionality and confidence scores")
    print(f"💡 Click legend items to show/hide individual masks")
    print(f"✅ High-confidence masks (≥0.6) are visible by default, low-confidence masks are hidden")
    print(f"🎯 Confidence threshold: Only masks with confidence ≥ 0.60 are shown initially")
    
    # Determine if we should save the image
    backend = matplotlib.get_backend()
    if save_path is None and backend == 'Agg':
        # Non-interactive backend, auto-save
        save_path = "mask_overlay_results.png"
    
    # Save if path provided or if using non-interactive backend
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Mask visualization saved to: {save_path}")
    
    # Try to display the plot
    try:
        if backend != 'Agg':
            print("🖼️  Interactive Mask visualization displayed!")
            print("   👆 You can interact with the visualization:")
            print("   • Press 'q' or ESC to close the window")
            print("   • Press 's' to save the current view")
            print("   • Click legend items to toggle individual masks ON/OFF")
            print("   • Close the window manually to continue")
            print("   • Green checkmark (✅) = mask visible, Red X (❌) = mask hidden")
            
            # Enable keyboard shortcuts
            def on_key_press(event):
                if event.key in ['q', 'escape']:
                    plt.close(fig)
                    print("🔄 Mask visualization closed by user")
                elif event.key == 's':
                    from datetime import datetime
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_name = f"mask_overlay_manual_save_{timestamp}.png"
                    fig.savefig(save_name, dpi=300, bbox_inches='tight')
                    print(f"💾 Manual save completed: {save_name}")
            
            # Connect the key press event (with fallback)
            try:
                fig.canvas.mpl_connect('key_press_event', on_key_press)
            except:
                pass  # Fallback if key events aren't supported
            
            # Force the figure to display
            plt.figure(fig.number)  # Make sure this figure is active
            plt.draw()  # Force draw
            plt.pause(0.1)  # Small pause to ensure rendering
            print("🔄 Mask figure should now be visible...")
            
            # Don't call plt.show() here as it will be called by the main script
            
        else:
            print("🖼️  Mask visualization saved as image file (display not available in this environment)")
    except Exception as e:
        print(f"⚠️  Could not display mask visualization: {e}")
        if not save_path:
            plt.savefig("mask_overlay_results.png", dpi=300, bbox_inches='tight')
            print("💾 Saved mask visualization to: mask_overlay_results.png")
    
    return fig

def visualize_segmentation_results(image_source, regions, save_path=None, interactive_mode=True, source_type="filepath"):
    """
    Create a visualization showing the original image and segmented regions with color legend.
    
    Args:
        image_source (str): Either URL or filepath of the image
        regions (list): List of region objects from Clarifai prediction
        save_path (str, optional): Path to save the visualization. If None, auto-generates.
        interactive_mode (bool): If True, waits for user interaction to close. If False, closes automatically.
        source_type (str): Either "url" or "filepath" to indicate image source type
    
    Returns:
        matplotlib.figure.Figure: The created figure object for manual cleanup if needed
    """
    # Download or load the original image based on source type
    if source_type == "url":
        print("📥 Downloading image for visualization...")
        original_image = download_image(image_source)
    else:  # filepath
        print("📁 Loading local image for visualization...")
        try:
            original_image = Image.open(image_source)
            print(f"✅ Loaded image from: {image_source}")
        except Exception as e:
            print(f"❌ Error loading local image: {e}")
            return None
    
    if original_image is None:
        print("❌ Could not download image for visualization")
        return
    
    # Create image with segmentation regions and get color mappings
    segmented_image, color_mappings = draw_segmentation_regions(original_image, regions)
    
    # Create a figure with three subplots: original, segmented, and legend
    fig = plt.figure(figsize=(20, 8))
    
    # Create a grid layout: 2 columns for images, 1 for legend
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.5], height_ratios=[1, 0.1])
    
    # Original image subplot
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(original_image)
    ax1.set_title('Original Image', fontsize=16, fontweight='bold', pad=20)
    ax1.axis('off')
    
    # Segmented image subplot
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.imshow(segmented_image)
    ax2.set_title('Segmented Regions', fontsize=16, fontweight='bold', pad=20)
    ax2.axis('off')
    
    # Legend subplot
    ax3 = fig.add_subplot(gs[:, 2])
    ax3.axis('off')
    ax3.set_title('Region Legend', fontsize=14, fontweight='bold', pad=20)
    
    # Create color legend
    legend_y_start = 0.95
    legend_y_step = 0.08
    
    for i, mapping in enumerate(color_mappings):
        y_pos = legend_y_start - (i * legend_y_step)
        
        # Draw color square
        color_square = plt.Rectangle((0.1, y_pos-0.02), 0.1, 0.04, 
                                   facecolor=mapping['color'], 
                                   edgecolor='black', linewidth=1)
        ax3.add_patch(color_square)
        
        # Add text description
        text = f"R{mapping['region_id']}: {mapping['concept']}\n     ({mapping['confidence']:.2f})"
        ax3.text(0.25, y_pos, text, fontsize=10, va='center', ha='left')
    
    # Set legend limits
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # Add main title
    fig.suptitle('Clarifai Image Segmentation Results with Polygon Overlay Regions', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    
    # Print color mapping to console
    print("\n🎨 Region Color Mapping:")
    print("=" * 60)
    for mapping in color_mappings:
        print(f"🔷 Region {mapping['region_id']} ({mapping['color']}): {mapping['concept']} "
              f"({mapping['confidence']:.2f} confidence)")
    
    # Determine if we should save the image
    backend = matplotlib.get_backend()
    if save_path is None and backend == 'Agg':
        # Non-interactive backend, auto-save
        save_path = "segmentation_results_with_legend.png"
    
    # Save if path provided or if using non-interactive backend
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved to: {save_path}")
    
    # Try to display the plot
    try:
        if backend != 'Agg':
            if interactive_mode:
                # Set up interactive mode with keyboard shortcuts
                print("🖼️  Visualization displayed!")
                print("   👆 You can interact with the visualization:")
                print("   • Press 'q' or ESC to close the window")
                print("   • Press 's' to save the current view")
                print("   • Close the window manually to continue")
                print("   • Use toolbar for zoom/pan functionality")
                
                # Enable keyboard shortcuts
                def on_key_press(event):
                    if event.key in ['q', 'escape']:
                        plt.close(fig)
                        print("🔄 Visualization closed by user")
                    elif event.key == 's':
                        from datetime import datetime
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        save_name = f"segmentation_manual_save_{timestamp}.png"
                        fig.savefig(save_name, dpi=300, bbox_inches='tight')
                        print(f"💾 Manual save completed: {save_name}")
                
                # Connect the key press event (with fallback)
                try:
                    fig.canvas.mpl_connect('key_press_event', on_key_press)
                except:
                    pass  # Fallback if key events aren't supported
                
                # Force the figure to display
                plt.figure(fig.number)  # Make sure this figure is active
                plt.draw()  # Force draw
                plt.pause(0.1)  # Small pause to ensure rendering
                print("🔄 Figure should now be visible...")
                
                # Don't use plt.show() here as it will be called by the main script
                
            else:
                # Non-interactive mode - display briefly and close
                plt.show(block=False)
                print("🖼️  Visualization displayed (non-interactive mode)")
        else:
            print("🖼️  Visualization saved as image file (display not available in this environment)")
    except Exception as e:
        print(f"⚠️  Could not display visualization: {e}")
        if not save_path:
            plt.savefig("segmentation_results_with_legend.png", dpi=300, bbox_inches='tight')
            print("💾 Saved visualization to: segmentation_results_with_legend.png")
    
    # Return figure for potential manual cleanup
    return fig

def cleanup_session():
    """
    Clean up matplotlib resources and close any open figures at the end of the session.
    This ensures proper resource cleanup and prevents memory leaks.
    """
    try:
        # Close all matplotlib figures
        plt.close('all')
        
        # Clear the matplotlib cache if available
        if hasattr(plt, 'clf'):
            plt.clf()
        
        # Force garbage collection for thorough cleanup
        import gc
        gc.collect()
        
        print("🧹 Session cleanup completed - all visualization windows closed")
        
    except Exception as e:
        print(f"⚠️  Minor cleanup issue (can be ignored): {e}")

def wait_for_user_exit():
    """
    Wait for user input to exit the session gracefully.
    Provides multiple exit options for user convenience.
    """
    print("\n" + "=" * 60)
    print("🎯 INTERACTIVE SESSION CONTROL")
    print("=" * 60)
    print("Choose how to end this session:")
    print("  1️⃣  Press ENTER to exit immediately")
    print("  2️⃣  Type 'save' to save visualization and exit")
    print("  3️⃣  Type 'help' for visualization tips")
    print("  4️⃣  Any visualization windows can be closed with 'q' or ESC")
    
    while True:
        try:
            user_input = input("\n➤ Your choice (or just press ENTER to exit): ").strip().lower()
            if not user_input:
                print("Exiting session.")
                break
            elif user_input == 'save':
                print("💾 'save' command acknowledged. In a real application, you would trigger a save here.")
                # The logic to save would depend on which visualization is active.
                # Since this function is generic, we just print a message and exit.
                break
            elif user_input == 'help':
                print("\n--- Visualization Tips ---")
                print(" • You can zoom and pan the image using the toolbar in the plot window.")
                print(" • Press 's' in the plot window to save a snapshot if enabled.")
                print(" • Close the plot window with 'q', ESC, or the window's close button.")
                print("--------------------------")
            else:
                print(f"Unrecognized command: '{user_input}'. Please try again or press ENTER to exit.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session due to user interrupt.")
            break

    cleanup_session()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # =============================================================================
    # AUTHENTICATION: Set up API credentials
    # =============================================================================
    # Get the Personal Access Token (PAT) from environment variables
    # Your PAT can be found under "Settings" -> "Security" in your Clarifai account
    # 
    # To set your environment variable, run:
    # export CLARIFAI_PAT="your_actual_api_key_here"
    pat = os.getenv('CLARIFAI_PAT')

    # Check if the PAT was successfully retrieved
    if not pat:
        raise ValueError(
            "❌ Please set the CLARIFAI_PAT environment variable\n"
            "Run: export CLARIFAI_PAT='your_actual_api_key_here'\n"
            "Get your PAT from: https://clarifai.com/settings/security"
        )

    # =============================================================================
    # INPUT CONFIGURATION: Define the image to analyze
    # =============================================================================
    # URL of the sample image we want to analyze
    # This image contains people walking, which will be segmented into different regions
    image_url = "https://tf-cmsv2-smithsonianmag-media.s3.amazonaws.com/filer/ea/21/ea2159df-9eec-4b05-a3de-356c57e23227/another_airplane_4676723312.jpg"
    
    # For local file testing, uncomment and modify this line:
    # image_path = "image.png"  # Local file path
    # source_type = "filepath"  # Change to "filepath" for local files

    print(f"🖼️  Analyzing image: {image_url}")
    print("📊 Running image segmentation analysis with visualization...")
    print(f"🎭 Matplotlib backend: {matplotlib.get_backend()}")

    # =============================================================================
    # MODEL SETUP: Initialize the Clarifai segmentation model
    # =============================================================================
    # Define the URL of Clarifai's general image segmentation model
    # This model can identify and segment various objects and regions in images
    model_url = "https://clarifai.com/clarifai/main/models/image-general-segmentation"

    try:
        # Create a Model instance with our credentials and make a prediction
        # The predict_by_url() method sends the image URL to Clarifai for analysis
        print("🔄 Sending request to Clarifai API...")
        model_prediction = Model(url=model_url, pat=pat).predict_by_url(image_url)
        print("✅ Received prediction from Clarifai API")

        # =============================================================================
        # RAW JSON RESPONSE: Print and save complete API response
        # =============================================================================
        print_raw_prediction_json(model_prediction, "clarifai_raw_prediction.json")

        # =============================================================================
        # RESULTS PROCESSING: Extract and display segmentation results
        # =============================================================================
        # The prediction results contain regions - different segments of the image
        # Each region has concepts (what was detected) with confidence values
        regions = model_prediction.outputs[0].data.regions

        # Debug: Show region structure to understand the data format
        debug_region_structure(regions)

        # Print detailed polygon and mask data
        print_all_polygon_data(regions)

        # Process and display mask information
        process_mask_data(regions)

        # Save complete data to JSON file
        save_polygon_data_to_json(regions, image_url, "complete_segmentation_data.json")

        # =============================================================================
        # VISUALIZATION: Create interactive visual displays
        # =============================================================================
        if len(regions) > 0:
            print(f"\n🎨 Creating visualizations for {len(regions)} regions...")
            
            # Try mask visualization first (primary method, uses OpenCV)
            print("🎭 Attempting mask visualization...")
            mask_fig = visualize_masks_on_image(image_url, regions, source_type="url")
            
            # If mask visualization failed, use polygon/bounding box visualization
            if mask_fig is None:
                print("🔄 Mask visualization failed, using bounding box visualization...")
                polygon_fig = visualize_segmentation_results(image_url, regions, source_type="url")
                active_fig = polygon_fig
                visualization_type = "Bounding Box"
            else:
                active_fig = mask_fig
                visualization_type = "Mask Overlay"
            
            # Display the successful visualization
            if active_fig:
                print(f"\n🖼️  Displaying {visualization_type} visualization...")
                
                # Show the active figure
                try:
                    plt.show()  # This will display the created figure
                    print("📺 Visualization window is now active")
                    print("💡 Close the window or use keyboard shortcuts to continue")
                except Exception as e:
                    print(f"⚠️  Display issue: {e}")
                    print("💾 Image may have been saved to file instead")
                
                # Wait for user interaction before cleanup
                wait_for_user_exit()
            else:
                print("❌ No visualizations were created")
                print("💡 Check that the required packages are installed and the image is accessible")
        else:
            print("❌ No regions found in the image")
            print("💡 The model may not have detected any segmentable regions")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("💡 Please check your API key, internet connection, and image accessibility")
        raise
    
    finally:
        # Always clean up resources
        cleanup_session()
        print("\n✅ Visual segmentation analysis complete!")
        print("🎉 Thank you for using Clarifai's image segmentation with visualization!")
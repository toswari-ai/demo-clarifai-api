#!/usr/bin/env python3
"""
Clarifai Visual Object Detection Demo with Interactive Zoom and Detail Views

This script demonstrates how to use Clarifai's visual detection models to identify
and locate specific objects within images. Unlike classification (which tells you
what's in an image), detection tells you WHERE objects are located using bounding
boxes and coordinates.

Features:
- Downloads and analyzes images using Clarifai's object detection API
- Displays detailed text results with confidence scores and coordinates
- Shows visual comparison between original image and detected objects
- Creates colored bounding box overlays for each detected object
- Interactive zoom and pan functionality for detailed inspection
- Interactive legend with click-to-toggle detection visibility
- Large, high-resolution visualization (24x12 inches)
- Individual detail views showing cropped regions around each detection
- Color-coded visualization with comprehensive legend
- Enhanced formatting with object size and position analysis
- Preserves original image visibility with semi-transparent overlays
- Real-time detection filtering for focused analysis

This example uses Clarifai's face detection model to find and locate faces
in an image, providing both identification and precise positioning information.

For more information about Clarifai's visual detection models, visit:
https://docs.clarifai.com/getting-started/quickstart

Author: Clarifai
Last Updated: 2025
Requirements: clarifai>=11.6.0, matplotlib>=3.5.0, Pillow>=10.0.0, requests
"""

# Import necessary libraries
from clarifai.client.model import Model  # Clarifai's Model class for API interactions
import os  # For accessing environment variables
import requests  # For downloading images from URLs
from PIL import Image, ImageDraw, ImageFont  # For image processing and visualization
import numpy as np  # For numerical operations
from io import BytesIO  # For handling image data in memory

# Configure matplotlib for different environments
import matplotlib
import matplotlib.pyplot as plt  # For displaying images
import matplotlib.patches as patches  # For drawing rectangles on images

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

def draw_detection_boxes(image, regions, show_labels=True, visible_detections=None):
    """
    Draw colored bounding boxes and labels for detected objects on an image.
    
    Args:
        image (PIL.Image): The original image
        regions (list): List of region objects from Clarifai prediction
        show_labels (bool): Whether to show concept labels on the image
        visible_detections (list): List of detection indices that should be visible (None means all visible)
        
    Returns:
        tuple: (PIL.Image with bounding boxes drawn, list of color mappings)
    """
    # Create a copy of the image to draw on
    img_with_boxes = image.copy().convert('RGBA')
    
    # Create a transparent overlay for colored regions
    overlay = Image.new('RGBA', img_with_boxes.size, (0, 0, 0, 0))
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
    
    # Default to showing all detections if not specified
    if visible_detections is None:
        visible_detections = list(range(len(regions)))
    
    for i, region in enumerate(regions):
        # Choose unique color for this detection
        color_hex = colors[i % len(colors)]
        
        # Only draw if this detection is visible
        if i not in visible_detections:
            # Still store color mapping but don't draw
            if region.data.concepts:
                top_concept = region.data.concepts[0]
                concept_name = top_concept.name
                confidence = top_concept.value
            else:
                concept_name = 'Unknown'
                confidence = 0.0
                
            if hasattr(region.region_info, 'bounding_box'):
                color_mappings.append({
                    'detection_id': i + 1,
                    'color': color_hex,
                    'concept': concept_name,
                    'confidence': confidence,
                    'bbox': region.region_info.bounding_box,
                    'visible': False
                })
            continue
        
        # Get bounding box coordinates
        if hasattr(region.region_info, 'bounding_box'):
            bbox = region.region_info.bounding_box
            
            # Convert normalized coordinates to pixel coordinates
            img_width, img_height = image.size
            left = int(bbox.left_col * img_width)
            top = int(bbox.top_row * img_height)
            right = int(bbox.right_col * img_width)
            bottom = int(bbox.bottom_row * img_height)
            
            # Choose unique color for this detection
            color_hex = colors[i % len(colors)]
            
            # Convert hex color to RGB tuple
            color_rgb = tuple(int(color_hex[j:j+2], 16) for j in (1, 3, 5))
            
            # Create semi-transparent colored overlay for this bounding box
            overlay_color = color_rgb + (60,)  # RGBA with 60/255 opacity (~23% transparent)
            overlay_draw.rectangle([left, top, right, bottom], fill=overlay_color)
            
            # Draw border around the detection
            border_color = color_rgb + (255,)  # Fully opaque border
            overlay_draw.rectangle([left, top, right, bottom], outline=border_color, width=4)
            
            # Get the top concept for this detection
            if region.data.concepts:
                top_concept = region.data.concepts[0]
                concept_name = top_concept.name
                confidence = top_concept.value
                
                # Store color mapping
                color_mappings.append({
                    'detection_id': i + 1,
                    'color': color_hex,
                    'concept': concept_name,
                    'confidence': confidence,
                    'bbox': bbox,
                    'visible': True
                })
                
                if show_labels:
                    label = f"D{i+1}: {concept_name} ({confidence:.2f})"
                    
                    # Calculate text size for background
                    text_bbox = overlay_draw.textbbox((left, top-30), label, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    # Draw label background with some padding (semi-transparent)
                    bg_color = color_rgb + (200,)  # Semi-transparent background
                    overlay_draw.rectangle([left-4, top-34, left+text_width+8, top-2], fill=bg_color)
                    
                    # Draw label text in white for contrast
                    overlay_draw.text((left, top-30), label, fill='white', font=font)
            else:
                # Store color mapping even if no concepts
                color_mappings.append({
                    'detection_id': i + 1,
                    'color': color_hex,
                    'concept': 'Unknown',
                    'confidence': 0.0,
                    'bbox': bbox,
                    'visible': True
                })
    
    # Composite the overlay onto the original image
    img_with_boxes = Image.alpha_composite(img_with_boxes, overlay)
    
    # Convert back to RGB for matplotlib compatibility
    img_with_boxes = img_with_boxes.convert('RGB')
    
    return img_with_boxes, color_mappings

def visualize_detection_results(image_url, regions, save_path=None):
    """
    Create a visualization showing the original image and detected objects with bounding boxes.
    Features larger, zoomable images for better inspection and interactive legend for toggling detections.
    
    Args:
        image_url (str): URL of the original image
        regions (list): List of region objects from Clarifai prediction
        save_path (str, optional): Path to save the visualization. If None, auto-generates.
    """
    # Download the original image
    print("📥 Downloading image for visualization...")
    original_image = download_image(image_url)
    
    if original_image is None:
        print("❌ Could not download image for visualization")
        return
    
    # Initialize visibility state for all detections
    visible_detections = list(range(len(regions)))
    
    # Create image with detection bounding boxes and get color mappings
    detected_image, color_mappings = draw_detection_boxes(original_image, regions, visible_detections=visible_detections)
    
    # Create a larger figure with enhanced layout for better viewing
    fig = plt.figure(figsize=(24, 12))  # Much larger figure
    
    # Create a grid layout: 2 columns for images, 1 for legend
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 0.6], height_ratios=[1, 0.15])
    
    # Original image subplot with zoom capability
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(original_image)
    ax1.set_title('Original Image (Zoom/Pan Available)', fontsize=18, fontweight='bold', pad=25)
    ax1.axis('on')  # Keep axes for zoom/pan functionality
    ax1.set_xticks([])
    ax1.set_yticks([])
    # Add grid for better reference when zooming
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Detected image subplot with zoom capability
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.imshow(detected_image)
    ax2.set_title('Detected Objects (Zoom/Pan Available)', fontsize=18, fontweight='bold', pad=25)
    ax2.axis('on')  # Keep axes for zoom/pan functionality
    ax2.set_xticks([])
    ax2.set_yticks([])
    # Add grid for better reference when zooming
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend subplot
    ax3 = fig.add_subplot(gs[:, 2])
    ax3.axis('off')
    ax3.set_title('Detection Legend (Click to Toggle)', fontsize=16, fontweight='bold', pad=25)
    
    # Add interactive functionality
    def update_visualization():
        """Update the visualization when detections are toggled"""
        nonlocal detected_image
        detected_image, _ = draw_detection_boxes(original_image, regions, visible_detections=visible_detections)
        ax2.clear()
        ax2.imshow(detected_image)
        ax2.set_title('Detected Objects (Zoom/Pan Available)', fontsize=18, fontweight='bold', pad=25)
        ax2.axis('on')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        update_legend()
        fig.canvas.draw()
    
    def update_legend():
        """Update the legend display with current visibility states"""
        ax3.clear()
        ax3.axis('off')
        ax3.set_title('Detection Legend (Click to Toggle)', fontsize=16, fontweight='bold', pad=25)
        
        legend_y_start = 0.95
        legend_y_step = 0.08
        
        for i, mapping in enumerate(color_mappings):
            y_pos = legend_y_start - (i * legend_y_step)
            detection_idx = mapping['detection_id'] - 1
            is_visible = detection_idx in visible_detections
            
            # Use different styling for visible/hidden detections
            alpha = 1.0 if is_visible else 0.3
            edge_width = 2 if is_visible else 1
            
            # Draw color square with visibility indicator
            color_square = plt.Rectangle((0.05, y_pos-0.025), 0.12, 0.05, 
                                       facecolor=mapping['color'], alpha=alpha,
                                       edgecolor='black' if is_visible else 'gray', 
                                       linewidth=edge_width)
            ax3.add_patch(color_square)
            
            # Add visibility indicator
            status_icon = "👁️" if is_visible else "👁️‍🗨️"
            ax3.text(0.01, y_pos, status_icon, fontsize=12, va='center', ha='left')
            
            # Add enhanced text description
            bbox = mapping['bbox']
            width = bbox.right_col - bbox.left_col
            height = bbox.bottom_row - bbox.top_row
            confidence_pct = mapping['confidence'] * 100
            
            # Multi-line text with visibility styling
            text_color = 'black' if is_visible else 'gray'
            font_weight = 'bold' if is_visible else 'normal'
            
            text_lines = [
                f"D{mapping['detection_id']}: {mapping['concept']}",
                f"     Conf: {confidence_pct:.1f}%",
                f"     Size: {width:.2f}×{height:.2f}"
            ]
            
            # Draw each line with proper spacing and styling
            for j, line in enumerate(text_lines):
                ax3.text(0.2, y_pos - j*0.015, line, fontsize=10, va='center', ha='left',
                        fontweight='bold' if j == 0 and is_visible else font_weight,
                        color=text_color, alpha=alpha)
        
        # Add instructions
        ax3.text(0.5, 0.05, "💡 Click on legend items to\nshow/hide detections", 
                ha='center', va='bottom', fontsize=10, style='italic',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
    
    def on_legend_click(event):
        """Handle clicks on the legend to toggle detection visibility"""
        if event.inaxes == ax3:
            legend_y_start = 0.95
            legend_y_step = 0.08
            
            # Calculate which legend item was clicked
            for i, mapping in enumerate(color_mappings):
                y_pos = legend_y_start - (i * legend_y_step)
                
                # Check if click is within the legend item area
                if (0.01 <= event.xdata <= 0.95 and 
                    y_pos - 0.04 <= event.ydata <= y_pos + 0.025):
                    
                    detection_idx = mapping['detection_id'] - 1
                    
                    # Toggle visibility
                    if detection_idx in visible_detections:
                        visible_detections.remove(detection_idx)
                        print(f"🔍 Hidden detection {mapping['detection_id']}: {mapping['concept']}")
                    else:
                        visible_detections.append(detection_idx)
                        print(f"👁️ Showing detection {mapping['detection_id']}: {mapping['concept']}")
                    
                    # Update the visualization
                    update_visualization()
                    break
    
    # Initialize the legend display
    update_legend()
    
    # Connect the click event handler
    fig.canvas.mpl_connect('button_press_event', on_legend_click)
    
    # Add main title with enhanced styling
    fig.suptitle('Clarifai Object Detection Results - Interactive View (Mouse: Zoom/Pan, Legend: Click to Toggle)', 
                fontsize=20, fontweight='bold', y=0.96)
    
    # Add instruction text
    fig.text(0.5, 0.02, 'Instructions: Mouse wheel to zoom, click and drag to pan. Click legend items to show/hide detections. Right-click to reset view.', 
             ha='center', va='bottom', fontsize=12, style='italic', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # Enable interactive navigation toolbar
    try:
        # Add navigation toolbar for zooming and panning
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        print("🔍 Interactive zoom and pan enabled - use mouse wheel and drag!")
        print("👆 Click on legend items to show/hide specific detections!")
    except ImportError:
        print("📝 Basic zoom available - matplotlib navigation may be limited")
        print("👆 Click on legend items to show/hide specific detections!")
    
    # Adjust layout with more spacing
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    
    # Print color mapping to console with enhanced formatting
    print("\n🎨 Detection Color Mapping:")
    print("=" * 70)
    for mapping in color_mappings:
        bbox = mapping['bbox']
        width = bbox.right_col - bbox.left_col
        height = bbox.bottom_row - bbox.top_row
        confidence_pct = mapping['confidence'] * 100
        area = width * height
        
        print(f"🔷 Detection {mapping['detection_id']} ({mapping['color']}):")
        print(f"   📋 Object: {mapping['concept']}")
        print(f"   📊 Confidence: {confidence_pct:.1f}%")
        print(f"   📐 Size: {width:.3f} × {height:.3f} (Area: {area:.4f})")
        print(f"   📍 Position: ({bbox.left_col:.3f}, {bbox.top_row:.3f}) to ({bbox.right_col:.3f}, {bbox.bottom_row:.3f})")
        print()
    
    # Determine if we should save the image
    backend = matplotlib.get_backend()
    if save_path is None and backend == 'Agg':
        # Non-interactive backend, auto-save
        save_path = "detection_results_large_view.png"
    
    # Save if path provided or if using non-interactive backend
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 High-resolution visualization saved to: {save_path}")
    
    # Try to display the plot
    try:
        if backend != 'Agg':
            # Enable interactive features
            plt.subplots_adjust(bottom=0.1)  # Make room for toolbar
            mngr = plt.get_current_fig_manager()
            
            # Try to maximize window for better viewing experience
            try:
                if hasattr(mngr, 'window'):
                    if hasattr(mngr.window, 'wm_state'):
                        mngr.window.wm_state('zoomed')  # Windows/Linux
                    elif hasattr(mngr.window, 'showMaximized'):
                        mngr.window.showMaximized()  # Qt backend
                    elif hasattr(mngr, 'full_screen_toggle'):
                        pass  # Keep normal size for stability
            except:
                print("📱 Using default window size")
            
            plt.show()
            print("🖼️  Interactive visualization displayed!")
            print("🔍 Use mouse wheel to zoom, click and drag to pan")
            print("� Click on legend items to show/hide specific detections")
            print("�🔄 Right-click on image to reset zoom")
            print("💾 Use the toolbar save button to export current view")
        else:
            print("🖼️  Large visualization saved as image file (display not available in this environment)")
    except Exception as e:
        print(f"⚠️  Could not display visualization: {e}")
        if not save_path:
            plt.savefig("detection_results_large_view.png", dpi=300, bbox_inches='tight', facecolor='white')
            print("💾 Saved large visualization to: detection_results_large_view.png")
    
    # Don't close immediately to allow interaction
    if backend != 'Agg':
        print("🔄 Close the window when you're done exploring to continue...")
        print("💡 Try clicking legend items to toggle detection visibility!")
        # plt.close() will be called when user closes the window

def create_detection_detail_view(image_url, regions, save_path=None):
    """
    Create individual detail views of each detected object with zoom and context.
    
    Args:
        image_url (str): URL of the original image
        regions (list): List of region objects from Clarifai prediction
        save_path (str, optional): Base path for saving detail views
    """
    if not regions:
        print("No detections to create detail views for.")
        return
    
    # Download the original image
    original_image = download_image(image_url)
    if original_image is None:
        return
    
    # Create detail view for each detection
    num_detections = len(regions)
    cols = min(3, num_detections)
    rows = (num_detections + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
    if num_detections == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = ['#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#00FFFF', '#FFA500', 
              '#800080', '#FFFF00', '#FF69B4', '#32CD32', '#FFD700', '#8A2BE2',
              '#DC143C', '#20B2AA', '#FF6347', '#4682B4']
    
    for i, region in enumerate(regions):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if hasattr(region.region_info, 'bounding_box'):
            bbox = region.region_info.bounding_box
            
            # Calculate crop area with padding
            img_width, img_height = original_image.size
            left = max(0, int((bbox.left_col - 0.1) * img_width))
            top = max(0, int((bbox.top_row - 0.1) * img_height))
            right = min(img_width, int((bbox.right_col + 0.1) * img_width))
            bottom = min(img_height, int((bbox.bottom_row + 0.1) * img_height))
            
            # Crop the image around the detection
            cropped_image = original_image.crop((left, top, right, bottom))
            
            # Draw bounding box on cropped image
            draw = ImageDraw.Draw(cropped_image)
            
            # Adjust coordinates for cropped image
            crop_left = int((bbox.left_col * img_width) - left)
            crop_top = int((bbox.top_row * img_height) - top)
            crop_right = int((bbox.right_col * img_width) - left)
            crop_bottom = int((bbox.bottom_row * img_height) - top)
            
            color_hex = colors[i % len(colors)]
            color_rgb = tuple(int(color_hex[j:j+2], 16) for j in (1, 3, 5))
            
            # Draw thick bounding box
            draw.rectangle([crop_left, crop_top, crop_right, crop_bottom], 
                         outline=color_rgb, width=4)
            
            # Display the cropped image
            ax.imshow(cropped_image)
            
            # Get concept info
            concept_name = "Unknown"
            confidence = 0.0
            if region.data.concepts:
                concept_name = region.data.concepts[0].name
                confidence = region.data.concepts[0].value
            
            ax.set_title(f'Detection {i+1}: {concept_name}\nConfidence: {confidence:.2f} ({confidence*100:.1f}%)', 
                        fontsize=14, fontweight='bold', color=color_hex)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'Detection {i+1}\nNo bounding box data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_detections, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle('Detailed Detection Views - Individual Object Close-ups', 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save detail view if requested
    if save_path:
        detail_path = save_path.replace('.png', '_details.png')
        plt.savefig(detail_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Detail views saved to: {detail_path}")
    
    # Show detail view
    backend = matplotlib.get_backend()
    if backend != 'Agg':
        plt.show()
        print("🔍 Detail view displayed - examine each detection closely!")
    else:
        detail_path = "detection_details.png"
        plt.savefig(detail_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Detail views saved to: {detail_path}")
    
    plt.close()

# =============================================================================
# SECURITY SETUP: Get API credentials from environment variables
# =============================================================================
# Security best practice: Get PAT from environment variable instead of hardcoding it
pat = os.getenv('CLARIFAI_PAT')

# Check if the PAT was successfully retrieved
if not pat:
    raise ValueError(
        "❌ Please set the CLARIFAI_PAT environment variable\n"
        "Linux/Mac: export CLARIFAI_PAT='your_actual_api_key_here'\n"
        "Windows: set CLARIFAI_PAT=your_actual_api_key_here\n"
        "Get your PAT from: https://clarifai.com/settings/security"
    )

# =============================================================================
# INPUT CONFIGURATION: Define the image to analyze
# =============================================================================
# URL of the sample image we want to analyze for object detection
# This image contains people walking and potentially has logos or brands visible
image_url = "https://s3.amazonaws.com/samples.clarifai.com/people_walking2.jpeg"

print(f"🔍 Visual Object Detection Analysis")
print(f"🖼️  Image: {image_url}")
print("🎯 Detecting and locating objects with bounding boxes...")

# =============================================================================
# MODEL SETUP: Initialize the Clarifai object detection model
# =============================================================================
# Using Clarifai's logo detection model v2, which specializes in finding
# brand logos and commercial identifiers in images
# This model not only identifies logos but also provides their exact locations
model_url = "https://clarifai.com/clarifai/main/models/face-detection"

print("🧠 Initializing logo detection model v2...")

# Create a Model instance and make a prediction
# The predict_by_url() method sends the image URL to Clarifai for analysis
model_prediction = Model(url=model_url, pat=pat).predict_by_url(image_url)

# =============================================================================
# RESULTS PROCESSING: Extract and display detection results
# =============================================================================
print("\n" + "=" * 60)
print("🎯 Object Detection Results:")
print("=" * 60)

try:
    # Extract the regions (detected objects) from the prediction results
    # Each region contains information about a detected object and its location
    regions = model_prediction.outputs[0].data.regions
    
    if regions and len(regions) > 0:
        print(f"Found {len(regions)} detected objects/logos:\n")
        
        # Process each detected region
        for i, region in enumerate(regions, 1):
            print(f"🔍 Detection #{i}:")
            
            # Extract bounding box information (object location)
            if hasattr(region.region_info, 'bounding_box'):
                bbox = region.region_info.bounding_box
                print(f"   📍 Location (bounding box):")
                print(f"      • Top-left: ({bbox.left_col:.3f}, {bbox.top_row:.3f})")
                print(f"      • Bottom-right: ({bbox.right_col:.3f}, {bbox.bottom_row:.3f})")
                print(f"      • Width: {bbox.right_col - bbox.left_col:.3f}")
                print(f"      • Height: {bbox.bottom_row - bbox.top_row:.3f}")
            
            # Extract detected concepts (what was found)
            if hasattr(region, 'data') and hasattr(region.data, 'concepts'):
                print(f"   🏷️  Detected concepts:")
                for concept in region.data.concepts:
                    confidence = concept.value * 100  # Convert to percentage
                    print(f"      • {concept.name}: {confidence:.2f}% confidence")
            
            print()  # Add spacing between detections
        
        # =============================================================================
        # VISUAL DISPLAY: Show original image and detected objects
        # =============================================================================
        print("\n🎨 Creating visual display of detection results...")
        try:
            # Display the original image alongside the detected objects
            # Optionally save the visualization (uncomment the line below to save)
            # visualize_detection_results(image_url, regions, save_path="detection_with_bounding_boxes.png")
            visualize_detection_results(image_url, regions)
            
            # Create detailed views of each detection
            print("\n🔍 Creating detailed views of individual detections...")
            create_detection_detail_view(image_url, regions)
            
        except Exception as e:
            print(f"⚠️  Could not display visualization: {e}")
            print("   This might happen if you don't have a display available (e.g., in a server environment)")
            print("   The text results above still show all the detection information!")
            
    else:
        print("⚠️  No objects/logos detected in this image.")
        print("   This might mean:")
        print("   • The image doesn't contain recognizable logos")
        print("   • The objects are too small or unclear")
        print("   • Try a different image or detection model")

    # Display raw regions data for debugging
    print("🔧 Raw Detection Data (for developers):")
    print(f"   Number of regions: {len(regions) if regions else 0}")
    print(f"   Raw regions: {regions}")

except Exception as e:
    print(f"❌ Error processing detection results: {e}")
    print("🔧 Troubleshooting tips:")
    print("   • Check if the image URL is accessible")
    print("   • Verify your internet connection")
    print("   • Try a different image or model")
    print("   • Check if the model response contains region data")

# =============================================================================
# EDUCATIONAL INFORMATION AND NEXT STEPS
# =============================================================================
print("\n" + "=" * 60)
print("✅ Object detection analysis complete!")

print(f"\n💡 Understanding object detection:")
print("   • Identifies WHAT objects are in an image (classification)")
print("   • Determines WHERE objects are located (bounding boxes)")
print("   • Provides confidence scores for each detection")
print("   • Uses coordinates relative to image dimensions (0.0 to 1.0)")
print("   • Colored bounding boxes show detected object locations")
print("   • Semi-transparent overlays preserve original image visibility")
print("   • Interactive zoom allows detailed inspection of detection accuracy")
print("   • Detail views show cropped regions around each detected object")
print("   • Interactive legend enables selective viewing of detection classes")
print("   • Click legend items to focus on specific types of objects")
print("   • Toggle detection visibility for cleaner analysis")

print(f"\n🧠 About logo detection:")
print("   • Specialized model for finding brand logos and commercial identifiers")
print("   • Version 2 offers improved accuracy and coverage")
print("   • Useful for brand monitoring, copyright detection, advertising analysis")
print("   • Can detect multiple logos in a single image")

print(f"\n📐 Understanding bounding boxes:")
print("   • Coordinates are normalized (0.0 to 1.0) relative to image size")
print("   • left_col, top_row: top-left corner of the detected object")
print("   • right_col, bottom_row: bottom-right corner of the detected object")
print("   • To get pixel coordinates: multiply by actual image width/height")

print(f"\n🚀 Try modifying this script:")
print("   • Change 'image_url' to analyze your own images")
print("   • Try different detection models:")
print("     - General object detection")
print("     - Face detection")
print("     - Text detection (OCR)")
print("     - Custom trained detection models")
print("   • Adjust bounding box overlay transparency")
print("   • Customize the color palette for better visualization")
print("   • Use mouse wheel to zoom into specific regions")
print("   • Click and drag to pan around large images")
print("   • Click legend items to show/hide specific detection classes")
print("   • Save high-resolution views of interesting areas")
print("   • Modify detail view cropping padding for closer/wider views")
print("   • Save the detection results by uncommenting the save_path parameter")
print("   • Filter results by confidence threshold")
print("   • Add object size analysis and statistics")
print("   • Create batch processing for multiple images")
print("   • Customize legend appearance and positioning")
print("   • Add keyboard shortcuts for bulk toggle operations")

print(f"\n📚 Learn more at: https://docs.clarifai.com/getting-started/quickstart")
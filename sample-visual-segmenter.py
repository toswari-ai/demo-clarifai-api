#!/usr/bin/env python3
"""
Clarifai Visual Segmentation Demo with Image Viewer

This script demonstrates how to use Clarifai's image segmentation model to identify
and analyze different segments/regions within an image. Image segmentation is the
process of partitioning an image into multiple segments or regions, each representing
different objects or parts of objects.

Features:
- Downloads and analyzes images using Clarifai's segmentation API
- Displays detailed text results with confidence scores
- Shows visual comparison between original image and segmented regions
- Draws bounding boxes with labels for each detected region
- Color-coded visualization for easy interpretation

For more information about Clarifai's visual models, visit:
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

def draw_segmentation_regions(image, regions, show_labels=True):
    """
    Draw bounding boxes and labels for segmentation regions on an image.
    
    Args:
        image (PIL.Image): The original image
        regions (list): List of region objects from Clarifai prediction
        show_labels (bool): Whether to show concept labels on the image
        
    Returns:
        PIL.Image: Image with segmentation regions drawn
    """
    # Create a copy of the image to draw on
    img_with_regions = image.copy()
    draw = ImageDraw.Draw(img_with_regions)
    
    # Try to use a default font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    # Color palette for different regions
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    
    for i, region in enumerate(regions):
        # Get bounding box coordinates if available
        if hasattr(region.region_info, 'bounding_box'):
            bbox = region.region_info.bounding_box
            
            # Convert normalized coordinates to pixel coordinates
            img_width, img_height = image.size
            left = int(bbox.left_col * img_width)
            top = int(bbox.top_row * img_height)
            right = int(bbox.right_col * img_width)
            bottom = int(bbox.bottom_row * img_height)
            
            # Choose color for this region
            color = colors[i % len(colors)]
            
            # Draw bounding box
            draw.rectangle([left, top, right, bottom], outline=color, width=3)
            
            if show_labels and region.data.concepts:
                # Get the top concept for this region
                top_concept = region.data.concepts[0]
                label = f"{top_concept.name} ({top_concept.value:.2f})"
                
                # Draw label background
                text_bbox = draw.textbbox((left, top-25), label, font=font)
                draw.rectangle(text_bbox, fill=color)
                
                # Draw label text
                draw.text((left, top-25), label, fill='white', font=font)
    
    return img_with_regions

def visualize_segmentation_results(image_url, regions, save_path=None):
    """
    Create a visualization showing the original image and segmented regions.
    
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
    
    # Create image with segmentation regions
    segmented_image = draw_segmentation_regions(original_image, regions)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Display original image
    ax1.imshow(original_image)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Display image with segmentation regions
    ax2.imshow(segmented_image)
    ax2.set_title('Segmented Regions', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add a main title
    fig.suptitle('Clarifai Image Segmentation Results', fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Determine if we should save the image
    backend = matplotlib.get_backend()
    if save_path is None and backend == 'Agg':
        # Non-interactive backend, auto-save
        save_path = "segmentation_results.png"
    
    # Save if path provided or if using non-interactive backend
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved to: {save_path}")
    
    # Try to display the plot
    try:
        if backend != 'Agg':
            plt.show()
            print("🖼️  Visualization displayed! Close the window to continue.")
        else:
            print("🖼️  Visualization saved as image file (display not available in this environment)")
    except Exception as e:
        print(f"⚠️  Could not display visualization: {e}")
        if not save_path:
            plt.savefig("segmentation_results.png", dpi=300, bbox_inches='tight')
            print("💾 Saved visualization to: segmentation_results.png")
    
    plt.close()  # Clean up the figure

# =============================================================================
# SECURITY SETUP: Get API credentials from environment variables
# =============================================================================
# Instead of hardcoding your API key in the script (which is insecure), 
# we retrieve it from an environment variable. This is a security best practice.
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
image_url = "https://s3.amazonaws.com/samples.clarifai.com/people_walking2.jpeg"

print(f"🖼️  Analyzing image: {image_url}")
print("📊 Running image segmentation analysis...")

# =============================================================================
# MODEL SETUP: Initialize the Clarifai segmentation model
# =============================================================================
# Define the URL of Clarifai's general image segmentation model
# This model can identify and segment various objects and regions in images
model_url = "https://clarifai.com/clarifai/main/models/image-general-segmentation"

# Create a Model instance with our credentials and make a prediction
# The predict_by_url() method sends the image URL to Clarifai for analysis
model_prediction = Model(url=model_url, pat=pat).predict_by_url(image_url)

# =============================================================================
# RESULTS PROCESSING: Extract and display segmentation results
# =============================================================================
# The prediction results contain regions - different segments of the image
# Each region has concepts (what was detected) with confidence values
regions = model_prediction.outputs[0].data.regions

print(f"\n🎯 Found {len(regions)} segmented regions in the image:")
print("=" * 50)

# Iterate through each detected region
for region_index, region in enumerate(regions, 1):
    print(f"\n🔍 Region {region_index}:")
    
    # Each region can contain multiple concepts (detected objects/features)
    for concept in region.data.concepts:
        # Extract the concept name (what was detected)
        name = concept.name
        
        # Extract and round the confidence value (how certain the AI is)
        # Values range from 0.0 (not confident) to 1.0 (very confident)
        confidence = round(concept.value, 4)  # Round to 4 decimal places for readability
        
        # Display the results in a formatted way
        # The confidence represents how much of the region this concept covers
        print(f"   • {name}: {confidence} ({confidence * 100:.2f}% confidence)")

# =============================================================================
# VISUAL DISPLAY: Show original image and segmented regions
# =============================================================================
print("\n🎨 Creating visual display of segmentation results...")
try:
    # Display the original image alongside the segmented regions
    # Optionally save the visualization (uncomment the line below to save)
    # visualize_segmentation_results(image_url, regions, save_path="segmentation_results.png")
    visualize_segmentation_results(image_url, regions)
except Exception as e:
    print(f"⚠️  Could not display visualization: {e}")
    print("   This might happen if you don't have a display available (e.g., in a server environment)")
    print("   The text results above still show all the segmentation information!")

# =============================================================================
# SUMMARY AND NEXT STEPS
# =============================================================================
print("\n" + "=" * 50)
print("✅ Image segmentation analysis complete!")
print("\n💡 Understanding the results:")
print("   • Each region represents a different part of the image")
print("   • Concepts show what objects/features were detected in each region")
print("   • Confidence values indicate how certain the AI is about each detection")
print("   • Higher values (closer to 1.0) mean higher confidence")

# =============================================================================
# SUMMARY AND NEXT STEPS
# =============================================================================
print("\n" + "=" * 50)
print("✅ Image segmentation analysis complete!")
print("\n💡 Understanding the results:")
print("   • Each region represents a different part of the image")
print("   • Concepts show what objects/features were detected in each region")
print("   • Confidence values indicate how certain the AI is about each detection")
print("   • Higher values (closer to 1.0) mean higher confidence")
print("   • Colored bounding boxes in the visualization show detected regions")

print("\n🚀 Try modifying this script:")
print("   • Change the 'image_url' to analyze your own images")
print("   • Try different segmentation models from Clarifai's model gallery")
print("   • Modify the visualization colors or add more region details")
print("   • Save the visualized results by uncommenting the save_path parameter")
print("   • Experiment with different image formats (JPG, PNG, etc.)")
print("   • Add region area calculations or other analysis features")
print("   • Create batch processing for multiple images")

print(f"\n📚 Learn more at: https://docs.clarifai.com/getting-started/quickstart")
#!/usr/bin/env python3
"""
Clarifai Visual Segmentation Demo - Text Analysis Only

This script demonstrates how to use Clarifai's image segmentation model to identify
and analyze different segments/regions within an image. Image segmentation is the
process of partitioning an image into multiple segments or regions, each representing
different objects or parts of objects.

Features:
- Downloads and analyzes images using Clarifai's segmentation API
- Displays detailed text results with confidence scores
- Shows polygon and bounding box coordinate information
- Provides comprehensive region analysis in text format

For more information about Clarifai's visual models, visit:
https://docs.clarifai.com/getting-started/quickstart

Author: Clarifai
Last Updated: 2025
Requirements: clarifai>=11.6.0, requests
"""

# Import necessary libraries
from clarifai.client.model import Model  # Clarifai's Model class for API interactions
import os  # For accessing environment variables
import requests  # For downloading images from URLs
from io import BytesIO  # For handling image data in memory

# =============================================================================
# REGION ANALYSIS FUNCTIONS
# =============================================================================
def debug_region_structure(regions):
    """
    Debug function to inspect the structure of region data from Clarifai.
    This helps understand what data is available for analysis.
    
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
            
            # Check for polygon
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

def analyze_polygon_geometry(regions):
    """
    Analyze polygon geometry and provide detailed information about shapes.
    
    Args:
        regions (list): List of region objects from Clarifai prediction
    """
    print("\n📐 POLYGON GEOMETRY ANALYSIS")
    print("=" * 50)
    
    for i, region in enumerate(regions, 1):
        if hasattr(region.region_info, 'polygon') and region.region_info.polygon:
            polygon_points = region.region_info.polygon.points
            
            # Calculate polygon area (approximate using shoelace formula)
            if len(polygon_points) >= 3:
                area = 0.0
                n = len(polygon_points)
                for j in range(n):
                    k = (j + 1) % n
                    area += polygon_points[j].col * polygon_points[k].row
                    area -= polygon_points[k].col * polygon_points[j].row
                area = abs(area) / 2.0
                
                # Calculate perimeter (approximate)
                perimeter = 0.0
                for j in range(n):
                    k = (j + 1) % n
                    dx = polygon_points[k].col - polygon_points[j].col
                    dy = polygon_points[k].row - polygon_points[j].row
                    perimeter += (dx*dx + dy*dy)**0.5
                
                print(f"\n🔺 Region {i} Polygon Analysis:")
                print(f"   • Vertices: {len(polygon_points)}")
                print(f"   • Approximate Area: {area:.4f} (normalized units)")
                print(f"   • Approximate Perimeter: {perimeter:.4f} (normalized units)")
                
                # Show all vertices
                print(f"   • Vertices coordinates:")
                for j, point in enumerate(polygon_points):
                    print(f"     [{j+1}] ({point.col:.4f}, {point.row:.4f})")
                    
        elif hasattr(region.region_info, 'bounding_box'):
            bbox = region.region_info.bounding_box
            width = bbox.right_col - bbox.left_col
            height = bbox.bottom_row - bbox.top_row
            area = width * height
            
            print(f"\n📦 Region {i} Bounding Box Analysis:")
            print(f"   • Width: {width:.4f} (normalized units)")
            print(f"   • Height: {height:.4f} (normalized units)")
            print(f"   • Area: {area:.4f} (normalized units)")
            print(f"   • Top-Left: ({bbox.left_col:.4f}, {bbox.top_row:.4f})")
            print(f"   • Bottom-Right: ({bbox.right_col:.4f}, {bbox.bottom_row:.4f})")

def download_image_info(url):
    """
    Get basic information about an image from URL without downloading the full content.
    
    Args:
        url (str): The URL of the image
        
    Returns:
        dict: Basic image information
    """
    try:
        # Make a HEAD request to get headers without downloading content
        response = requests.head(url)
        response.raise_for_status()
        
        info = {
            'url': url,
            'content_type': response.headers.get('content-type', 'unknown'),
            'content_length': response.headers.get('content-length', 'unknown'),
            'status': 'accessible'
        }
        
        print(f"� Image Information:")
        print(f"   • URL: {url}")
        print(f"   • Content Type: {info['content_type']}")
        if info['content_length'] != 'unknown':
            size_kb = int(info['content_length']) / 1024
            print(f"   • File Size: {size_kb:.1f} KB")
        print(f"   • Status: {info['status']}")
        
        return info
        
    except Exception as e:
        print(f"❌ Error accessing image: {e}")
        return {'url': url, 'status': 'error', 'error': str(e)}

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

# Debug: Show region structure to understand the data format
debug_region_structure(regions)

print(f"\n🎯 Found {len(regions)} segmented regions in the image:")
print("=" * 50)

# Enhanced color palette for region identification in text output
color_names = [
    'Bright Red', 'Bright Blue', 'Bright Green', 'Magenta', 'Cyan', 
    'Orange', 'Purple', 'Yellow', 'Hot Pink', 'Lime Green', 
    'Gold', 'Blue Violet', 'Crimson', 'Light Sea Green', 'Tomato', 'Steel Blue'
]

# Iterate through each detected region with color information
for region_index, region in enumerate(regions, 1):
    color_name = color_names[(region_index - 1) % len(color_names)]
    print(f"\n� Region {region_index} ({color_name}):")
    
    # Each region can contain multiple concepts (detected objects/features)
    for concept_index, concept in enumerate(region.data.concepts):
        # Extract the concept name (what was detected)
        name = concept.name
        
        # Extract and round the confidence value (how certain the AI is)
        # Values range from 0.0 (not confident) to 1.0 (very confident)
        confidence = round(concept.value, 4)  # Round to 4 decimal places for readability
        
        # Display the results in a formatted way with ranking
        # The confidence represents how much of the region this concept covers
        rank_symbol = "👑" if concept_index == 0 else "  "  # Crown for top prediction
        print(f"   {rank_symbol} {name}: {confidence} ({confidence * 100:.2f}% confidence)")
    
    # Show region location if available
    if hasattr(region.region_info, 'polygon') and region.region_info.polygon:
        polygon_points = region.region_info.polygon.points
        print(f"      🔺 Polygon with {len(polygon_points)} vertices")
        # Show first few points as example
        if len(polygon_points) > 0:
            first_point = polygon_points[0]
            print(f"         First vertex: ({first_point.col:.3f}, {first_point.row:.3f})")
        if len(polygon_points) > 1:
            second_point = polygon_points[1]
            print(f"         Second vertex: ({second_point.col:.3f}, {second_point.row:.3f})")
        if len(polygon_points) > 2:
            print(f"         ... and {len(polygon_points) - 2} more vertices")
    elif hasattr(region.region_info, 'bounding_box'):
        bbox = region.region_info.bounding_box
        print(f"      📍 Bounding box: ({bbox.left_col:.2f}, {bbox.top_row:.2f}) to "
              f"({bbox.right_col:.2f}, {bbox.bottom_row:.2f})")

# =============================================================================
# IMAGE ANALYSIS: Get basic image information
# =============================================================================
print("\n📋 Getting basic image information...")
image_info = download_image_info(image_url)

# =============================================================================
# POLYGON GEOMETRY ANALYSIS: Detailed shape analysis
# =============================================================================
analyze_polygon_geometry(regions)

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
print("   • Polygon coordinates show the exact segmented areas")
print("   • Normalized coordinates range from 0.0 to 1.0 (relative to image size)")
print("   • Polygon geometry analysis provides area and perimeter calculations")
print("   • Debug information shows the internal structure of API responses")

print("\n🚀 Try modifying this script:")
print("   • Change the 'image_url' to analyze your own images")
print("   • Try different segmentation models from Clarifai's model gallery")
print("   • Experiment with different image formats (JPG, PNG, etc.)")
print("   • Add additional polygon geometry calculations")
print("   • Create batch processing for multiple images")
print("   • Export results to JSON or CSV format")
print("   • Add statistical analysis of region sizes and confidence scores")
print("   • Implement region filtering based on confidence thresholds")

print(f"\n📚 Learn more at: https://docs.clarifai.com/getting-started/quickstart")

print("\n✅ Text-based segmentation analysis complete!")
print("� All region data has been processed and displayed in text format.")
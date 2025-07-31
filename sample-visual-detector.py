#!/usr/bin/env python3
"""
Clarifai Visual Object Detection Demo

This script demonstrates how to use Clarifai's visual detection models to identify
and locate specific objects within images. Unlike classification (which tells you
what's in an image), detection tells you WHERE objects are located using bounding
boxes and coordinates.

This example uses Clarifai's logo detection model to find and locate brand logos
in an image, providing both identification and precise positioning information.

For more information about Clarifai's visual detection models, visit:
https://docs.clarifai.com/getting-started/quickstart

Author: Clarifai
Last Updated: 2025
Requirements: clarifai>=11.6.0
"""

# Import necessary libraries
from clarifai.client.model import Model  # Clarifai's Model class for API interactions
import os  # For accessing environment variables

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
model_url = "https://clarifai.com/clarifai/main/models/logo-detection-v2"

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
print("   • Add visualization of bounding boxes")
print("   • Filter results by confidence threshold")

print(f"\n📚 Learn more at: https://docs.clarifai.com/getting-started/quickstart")
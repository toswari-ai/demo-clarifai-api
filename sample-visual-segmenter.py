#!/usr/bin/env python3
"""
Clarifai Visual Segmentation Demo

This script demonstrates how to use Clarifai's image segmentation model to identify
and analyze different segments/regions within an image. Image segmentation is the
process of partitioning an image into multiple segments or regions, each representing
different objects or parts of objects.

For more information about Clarifai's visual models, visit:
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
# SUMMARY AND NEXT STEPS
# =============================================================================
print("\n" + "=" * 50)
print("✅ Image segmentation analysis complete!")
print("\n💡 Understanding the results:")
print("   • Each region represents a different part of the image")
print("   • Concepts show what objects/features were detected in each region")
print("   • Confidence values indicate how certain the AI is about each detection")
print("   • Higher values (closer to 1.0) mean higher confidence")

print("\n🚀 Try modifying this script:")
print("   • Change the 'image_url' to analyze your own images")
print("   • Try different segmentation models from Clarifai's model gallery")
print("   • Add additional processing for specific use cases")

print(f"\n📚 Learn more at: https://docs.clarifai.com/getting-started/quickstart")
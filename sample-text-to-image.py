#!/usr/bin/env python3
"""
Clarifai Text-to-Image Generation Demo (Imagen-2)

This script demonstrates how to use Clarifai's Imagen-2 model to generate images from text descriptions.
Imagen-2 is Google's advanced text-to-image model that can create high-quality, realistic images
from detailed text prompts. It's particularly good at architectural designs, realistic scenes,
and detailed visual concepts.

For more information about Clarifai's image generation models, visit:
https://docs.clarifai.com/getting-started/quickstart

Author: Clarifai
Last Updated: 2025
Requirements: clarifai>=11.6.0, opencv-python, numpy
"""

# Import necessary libraries
from clarifai.client.model import Model  # Clarifai's Model class for API interactions
import numpy as np  # For numerical operations on image arrays
import cv2  # OpenCV for image processing and saving
import os  # For accessing environment variables
import base64  # For decoding base64 image data

# =============================================================================
# SECURITY SETUP: Get API credentials from environment variables
# =============================================================================
# Security best practice: Get PAT from environment variable instead of hardcoding it
# This prevents accidentally sharing your API key when sharing code
pat = os.getenv('CLARIFAI_PAT')
if not pat:
    raise ValueError(
        "❌ Please set the CLARIFAI_PAT environment variable\n"
        "Linux/Mac: export CLARIFAI_PAT='your_actual_api_key_here'\n"
        "Windows: set CLARIFAI_PAT=your_actual_api_key_here\n"
        "Get your PAT from: https://clarifai.com/settings/security"
    )

# =============================================================================
# INPUT CONFIGURATION: Define the text prompt for image generation
# =============================================================================
# The text description that will be converted into an image
# Note: Using bytes (b"...") format as required by the predict_by_bytes method
# Try modifying this prompt to generate different types of images
input_text = b"floor plan for 2 bedroom kitchen house"

print(f"🎨 Generating image from prompt: {input_text.decode('utf-8')}")
print("⏳ This may take 10-30 seconds depending on image complexity...")

# =============================================================================
# MODEL SETUP: Initialize the Imagen-2 model
# =============================================================================
# Imagen-2 is Google's advanced text-to-image generation model
# It's particularly good at:
# - Architectural designs and floor plans
# - Realistic photographs and scenes
# - Detailed artistic compositions
# - Technical diagrams and layouts
model_url = "https://clarifai.com/gcp/generate/models/imagen-2"

# Create model instance and generate image from text
# predict_by_bytes() sends the text prompt as bytes to the model
model_prediction = Model(url=model_url, pat=pat).predict_by_bytes(input_text)

print("🖼️  Image generation complete! Processing results...")

# =============================================================================
# IMAGE PROCESSING: Convert AI response to viewable image file
# =============================================================================
# The AI model returns the generated image as base64-encoded data
# We need to decode this and convert it to a standard image format

# Step 1: Extract the base64-encoded image data from the API response
# The response structure: outputs[0].data.image.base64
im_b = model_prediction.outputs[0].data.image.base64

print("🔧 Converting base64 data to image format...")

# Step 2: Convert base64 string to numpy array
# np.frombuffer() converts the base64 data to a byte array
# np.uint8 specifies the data type (unsigned 8-bit integers)
img_np = cv2.imdecode(np.frombuffer(im_b, np.uint8), cv2.IMREAD_COLOR)

# Step 3: Save the image to a file
# cv2.imwrite() saves the numpy array as a JPEG image file
output_filename = "imagen-2-output.jpg"
cv2.imwrite(output_filename, img_np)

# =============================================================================
# SUCCESS CONFIRMATION: Provide feedback to user
# =============================================================================
print(f"✅ Image successfully saved as '{output_filename}'")
print(f"📁 You can find the generated image in the current directory")

# Display image dimensions for reference
if img_np is not None:
    height, width, channels = img_np.shape
    print(f"📐 Image dimensions: {width}x{height} pixels ({channels} color channels)")

print("\n💡 Understanding the process:")
print("   1. Text prompt is sent to Imagen-2 model")
print("   2. AI generates image based on text description")
print("   3. Image is returned as base64-encoded data")
print("   4. Base64 data is decoded and saved as JPEG file")

print("\n🚀 Try modifying this script:")
print("   • Change 'input_text' to generate different images")
print("   • Try prompts like: 'modern kitchen design', 'sunset over mountains'")
print("   • Experiment with detailed descriptions for better results")
print("   • Change output filename to organize your generated images")

print(f"\n📚 Learn more at: https://docs.clarifai.com/getting-started/quickstart")
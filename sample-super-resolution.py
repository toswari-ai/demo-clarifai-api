#!/usr/bin/env python3
"""
Clarifai Super-Resolution Image Enhancement Demo

This script demonstrates how to use Clarifai's super-resolution models to enhance
image quality and increase resolution. Super-resolution uses AI to upscale images
while preserving and even improving detail quality, making it ideal for photo
enhancement and restoration.

This example uses a Latent Diffusion Model (LDM) for 4x super-resolution enhancement,
which can significantly improve image quality and detail.

NOTE: There may be an open bug with this model - check Clarifai's status if you encounter issues.

For more information about Clarifai's image enhancement models, visit:
https://docs.clarifai.com/getting-started/quickstart

Author: Clarifai
Last Updated: 2025
Requirements: clarifai>=11.6.0, Pillow>=10.0.0, requests
"""

# Import necessary libraries
import requests  # For downloading images from URLs
from PIL import Image  # Python Imaging Library for image processing
from io import BytesIO  # For handling byte streams
from clarifai.client.model import Model  # Clarifai's Model class for API interactions
import os  # For accessing environment variables
import base64  # For decoding base64-encoded image data

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

print("🎨 AI-Powered Super-Resolution Image Enhancement")
print("🔍 Enhancing image quality using advanced AI models...")

# =============================================================================
# MODEL CONFIGURATION: Set up super-resolution parameters
# =============================================================================
# Configure inference parameters for optimal super-resolution results
inference_params = dict(
    # Number of inference steps: More steps = better quality but slower processing
    # Recommended range: 50–100 steps for best balance of quality and speed
    num_inference_steps=50,
    
    # Eta parameter controls the randomness in the diffusion process
    # 1.0 = standard setting, provides good balance of detail and stability
    eta=1
)

print(f"⚙️  Super-resolution parameters:")
print(f"   • Inference steps: {inference_params['num_inference_steps']} (quality vs speed balance)")
print(f"   • Eta parameter: {inference_params['eta']} (diffusion randomness control)")

# =============================================================================
# INPUT CONFIGURATION: Download and prepare the image
# =============================================================================
# URL of the sample image to enhance
# This image shows a car dashboard - good for demonstrating detail enhancement
image_url = "https://samples.clarifai.com/car-dashboard-steering-wheel.jpg"

print(f"\n📷 Source image: {image_url}")
print("⬇️  Downloading image for processing...")

try:
    # Download the image from the URL
    response = requests.get(image_url)
    response.raise_for_status()  # Raise an exception for bad status codes
    image_bytes = response.content
    
    # Get original image dimensions for comparison
    original_image = Image.open(BytesIO(image_bytes))
    original_width, original_height = original_image.size
    
    print(f"✅ Original image downloaded successfully")
    print(f"📐 Original dimensions: {original_width} x {original_height} pixels")
    
except Exception as e:
    print(f"❌ Error downloading image: {e}")
    print("🔧 Try checking your internet connection or using a different image URL")
    exit(1)

# =============================================================================
# MODEL SETUP: Initialize the super-resolution model
# =============================================================================
print("\n🧠 Initializing LDM 4x Super-Resolution model...")

# Create model instance for super-resolution
# This model can increase image resolution by 4x while enhancing details
model = Model(
    "https://clarifai.com/comp-vis/super-resolution/models/ldm-super-resolution-4x-openimages",
    pat=pat
)

print("🚀 Processing image with AI super-resolution...")

# =============================================================================
# IMAGE PROCESSING: Apply super-resolution enhancement
# =============================================================================
try:
    # Apply super-resolution to the image
    # predict_by_bytes() processes the raw image data
    prediction = model.predict_by_bytes(
        image_bytes,                    # The downloaded image data
        input_type="image",             # Specify that we're processing an image
        inference_params=inference_params  # Use our configured parameters
    )
    
    print("✅ Super-resolution processing complete!")
    
    # =============================================================================
    # RESULTS PROCESSING: Extract and save the enhanced image
    # =============================================================================
    print("🔧 Extracting enhanced image data...")
    
    # The output is base64 encoded, so we need to decode it
    enhanced_image_b64 = prediction.outputs[0].data.image.base64
    enhanced_image_bytes = base64.b64decode(enhanced_image_b64)
    
    # Convert bytes to PIL Image object for processing
    enhanced_image = Image.open(BytesIO(enhanced_image_bytes))
    enhanced_width, enhanced_height = enhanced_image.size
    
    # Save the enhanced image
    output_filename = "super_resolution_output.jpg"
    enhanced_image.save(output_filename, quality=95)  # High quality JPEG
    
    print(f"💾 Enhanced image saved as '{output_filename}'")
    
    # =============================================================================
    # RESULTS SUMMARY: Display enhancement statistics
    # =============================================================================
    print("\n" + "=" * 60)
    print("📊 Super-Resolution Results Summary:")
    print("=" * 60)
    
    # Calculate enhancement metrics
    scale_factor_x = enhanced_width / original_width
    scale_factor_y = enhanced_height / original_height
    pixel_increase = (enhanced_width * enhanced_height) / (original_width * original_height)
    
    print(f"📐 Original dimensions: {original_width} x {original_height} pixels")
    print(f"📐 Enhanced dimensions: {enhanced_width} x {enhanced_height} pixels")
    print(f"📈 Scale factor: {scale_factor_x:.1f}x width, {scale_factor_y:.1f}x height")
    print(f"🔢 Total pixel increase: {pixel_increase:.1f}x more pixels")
    print(f"💾 Output file: {output_filename}")
    
    print(f"\n✅ Image enhancement complete!")

except Exception as e:
    print(f"❌ Error during super-resolution processing: {e}")
    print("🔧 Troubleshooting tips:")
    print("   • Check if the model is currently available")
    print("   • Try reducing num_inference_steps for faster processing")
    print("   • Verify your internet connection is stable")
    print("   • Check if your image is in a supported format")

# =============================================================================
# EDUCATIONAL INFORMATION AND NEXT STEPS
# =============================================================================
print(f"\n💡 Understanding super-resolution:")
print("   • Uses AI to increase image resolution while preserving/enhancing details")
print("   • Different from simple upscaling - actually reconstructs missing information")
print("   • Particularly effective for photographs with fine details")
print("   • 4x super-resolution means 4 times the width and height (16x more pixels)")

print(f"\n🧠 About Latent Diffusion Models (LDM):")
print("   • Advanced AI architecture for high-quality image generation and enhancement")
print("   • Works in a compressed 'latent space' for efficiency")
print("   • Uses iterative refinement process (controlled by inference_steps)")
print("   • Excellent for maintaining image quality during upscaling")

print(f"\n🚀 Try modifying this script:")
print("   • Change 'image_url' to enhance your own images")
print("   • Adjust 'num_inference_steps' (higher = better quality, slower)")
print("   • Try different super-resolution models from Clarifai")
print("   • Add before/after image comparison")
print("   • Process multiple images in batch")
print("   • Add image quality metrics calculation")

print(f"\n📚 Learn more at: https://docs.clarifai.com/getting-started/quickstart")
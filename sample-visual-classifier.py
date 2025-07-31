#!/usr/bin/env python3
"""
Clarifai Visual Classification Demo

This script demonstrates how to use Clarifai's visual classification models to analyze
and identify content in images. Visual classification can detect objects, scenes, emotions,
activities, and many other visual concepts with confidence scores.

In this example, we're using a face sentiment recognition model to analyze emotions
and facial expressions in an image.

For more information about Clarifai's visual classification models, visit:
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
# URL of the sample image we want to analyze
# This image contains the Statue of Liberty, which will be analyzed for faces and sentiments
image_url = "https://s3.amazonaws.com/samples.clarifai.com/featured-models/image-captioning-statue-of-liberty.jpeg"

print(f"🖼️  Analyzing image: {image_url}")
print("😊 Running face sentiment recognition analysis...")

# =============================================================================
# MODEL SETUP: Initialize the Clarifai visual classification model
# =============================================================================
# We're using a face sentiment recognition model that can detect:
# - Facial expressions (happy, sad, angry, surprised, etc.)
# - Emotional states and sentiments
# - Demographic information (if present)
model_url = "https://clarifai.com/clarifai/main/models/face-sentiment-recognition"

# Create a Model instance and make a prediction
# The predict_by_url() method sends the image URL to Clarifai for analysis
print("🔍 Processing image with AI model...")
model_prediction = Model(url=model_url, pat=pat).predict_by_url(image_url)

# =============================================================================
# RESULTS PROCESSING: Extract and display classification results
# =============================================================================
print("\n🎯 Analysis Results:")
print("=" * 50)

# Extract the concepts (detected features) from the prediction results
# Each concept has a name and a confidence value
concepts = model_prediction.outputs[0].data.concepts

if concepts:
    print(f"Found {len(concepts)} visual concepts:\n")
    
    # Sort concepts by confidence (highest first) for better readability
    sorted_concepts = sorted(concepts, key=lambda x: x.value, reverse=True)
    
    for concept in sorted_concepts:
        # Extract concept name and confidence value
        name = concept.name
        confidence = round(concept.value, 4)  # Round to 4 decimal places
        percentage = confidence * 100  # Convert to percentage
        
        # Create a visual confidence bar
        bar_length = int(confidence * 20)  # Scale to 20 characters
        confidence_bar = "█" * bar_length + "░" * (20 - bar_length)
        
        print(f"  {name:<20} | {confidence_bar} | {percentage:6.2f}%")
else:
    print("⚠️  No visual concepts detected in this image.")
    print("   This might mean:")
    print("   • The image doesn't contain faces (required for face sentiment recognition)")
    print("   • The image quality is too low for analysis")
    print("   • Try a different image or model")

# =============================================================================
# SUMMARY AND EDUCATIONAL INFORMATION
# =============================================================================
print("\n" + "=" * 50)
print("✅ Visual classification analysis complete!")

print("\n💡 Understanding the results:")
print("   • Each concept represents something the AI detected in the image")
print("   • Confidence values range from 0.0 (not confident) to 1.0 (very confident)")
print("   • Higher confidence values indicate more certain detections")
print("   • Results are sorted by confidence (most confident first)")

print("\n🧠 About Face Sentiment Recognition:")
print("   • Detects emotions and facial expressions in images")
print("   • Can identify: happiness, sadness, anger, surprise, fear, etc.")
print("   • Works best on clear, well-lit images with visible faces")
print("   • May also detect age groups, gender, and other demographic info")

print("\n🚀 Try modifying this script:")
print("   • Change 'image_url' to analyze your own images")
print("   • Try different visual classification models:")
print("     - General image classification")
print("     - NSFW content detection")
print("     - Celebrity recognition")
print("     - Custom trained models")
print("   • Add image preprocessing or additional analysis")

print(f"\n📚 Learn more at: https://docs.clarifai.com/getting-started/quickstart")

# Raw output for debugging (optional)
print(f"\n🔧 Raw API Response (for developers):")
print(f"   Type: {type(concepts)}")
print(f"   Length: {len(concepts) if concepts else 0}")
if concepts and len(concepts) > 0:
    print(f"   First concept structure: {concepts[0]}")

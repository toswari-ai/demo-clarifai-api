#!/usr/bin/env python3
"""
Clarifai Multimodal AI Demo

This script demonstrates how to use Clarifai's multimodal AI models that can process
both images and text together. Multimodal models can understand visual content and
answer questions about images, providing a more comprehensive AI analysis.

This example uses OpenBMB's MiniCPM-o model to analyze an image and answer text questions about it.

For more information about Clarifai's multimodal models, visit:
https://docs.clarifai.com/resources/api-overview/python-sdk

Author: Clarifai
Last Updated: 2025
Requirements: clarifai>=11.6.0
"""

# Import necessary libraries
from clarifai.client.model import Model  # Clarifai's Model class for API interactions
from clarifai.client.input import Inputs  # For creating multimodal inputs
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
# INPUT CONFIGURATION: Define the question and image to analyze
# =============================================================================
# The question we want to ask about the image
# This demonstrates the AI's ability to understand and analyze visual content
prompt = "What time of day is it?"

# URL of the image we want to analyze
# This image shows a metro-north train, and we're asking about the time of day
image_url = "https://samples.clarifai.com/metro-north.jpg"

print(f"🤖 Multimodal AI Analysis")
print(f"📷 Image: {image_url}")
print(f"❓ Question: '{prompt}'")
print("🔍 Processing image and text together...")

# =============================================================================
# MULTIMODAL INPUT SETUP: Combine image and text into a single input
# =============================================================================
# Create a multimodal input that combines both the image and text prompt
# This allows the AI to consider both the visual content and the question together
multi_inputs = Inputs.get_multimodal_input(
    input_id="",  # Empty input ID (auto-generated)
    image_url=image_url,  # The image to analyze
    raw_text=prompt  # The question to ask about the image
)

print("📊 Created multimodal input combining image and text")

# =============================================================================
# MODEL SETUP: Initialize the multimodal AI model
# =============================================================================
# Using OpenBMB's MiniCPM-o-2.6 model, which is a powerful multimodal model
# that can process both images and text simultaneously
# MiniCPM-o stands for "omni-modal" - designed for comprehensive understanding
# The 2.6 version is optimized for visual question answering and image analysis
model_url = "https://clarifai.com/openbmb/miniCPM/models/MiniCPM-o-2_6-language"

print("🧠 Initializing MiniCPM-o-2.6 multimodal model...")

# Create model instance and make prediction with multimodal input
# The predict() method processes both the image and text simultaneously
model_prediction = Model(url=model_url, pat=pat).predict(inputs=[multi_inputs])

# =============================================================================
# RESULTS PROCESSING: Extract and display the AI's analysis
# =============================================================================
print("\n" + "=" * 60)
print("🎯 AI Analysis Results:")
print("=" * 60)

try:
    # Extract the text response from the model's output
    ai_response = model_prediction.outputs[0].data.text.raw
    
    print(f"\n💬 AI Response:")
    print(f"   {ai_response}")
    
    print(f"\n✅ Multimodal analysis complete!")
    
except Exception as e:
    print(f"❌ Error processing results: {e}")
    print("🔧 Troubleshooting tips:")
    print("   • Check if the image URL is accessible")
    print("   • Verify your internet connection")
    print("   • Try a different image or question")
    print("   • Check if the model response contains text data")

# =============================================================================
# EDUCATIONAL INFORMATION AND NEXT STEPS
# =============================================================================
print(f"\n💡 Understanding multimodal AI:")
print("   • Processes both visual and textual information simultaneously")
print("   • Can answer questions about images, describe scenes, read text in images")
print("   • Useful for visual question answering, image captioning, document analysis")
print("   • More comprehensive than single-modality models")

print(f"\n🧠 About the MiniCPM-o-2.6 model:")
print("   • Developed by OpenBMB (Open Big Model Base) for multimodal understanding")
print("   • Omni-modal architecture that processes text, images, and their relationships")
print("   • Version 2.6 is optimized for visual question answering and image analysis")
print("   • Efficient design that balances performance with computational requirements")
print("   • Excellent at understanding context between visual and textual information")

print(f"\n🚀 Try modifying this script:")
print("   • Change the 'prompt' to ask different questions:")
print("     - 'What objects do you see in this image?'")
print("     - 'Describe the weather conditions'")
print("     - 'What colors are prominent in this image?'")
print("     - 'Count the number of people in this image'")
print("     - 'What is the main subject of this image?'")
print("   • Try different images by changing 'image_url'")
print("   • Experiment with more complex questions")
print("   • Try other multimodal models from Clarifai's gallery:")
print("     - GPT-4 Vision models")
print("     - Claude-3 with vision capabilities")
print("     - Other MiniCPM variants")

print(f"\n📚 Learn more at: https://docs.clarifai.com/resources/api-overview/python-sdk")
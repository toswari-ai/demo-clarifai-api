#!/usr/bin/env python3
"""
Clarifai Text-to-Speech (TTS) Demo

This script demonstrates how to use Clarifai's text-to-speech models to convert
written text into spoken audio. Text-to-speech is useful for accessibility,
voice assistants, audiobook creation, and interactive applications.

Note: This model may have some limitations - check Clarifai's status if you encounter issues.

For more information about Clarifai's audio models, visit:
https://docs.clarifai.com/getting-started/quickstart

Author: Clarifai
Last Updated: 2025
Requirements: clarifai>=11.6.0
"""

# Import necessary libraries
from clarifai.client.model import Model  # Clarifai's Model class for API interactions
import os  # For accessing environment variables
import base64  # For decoding base64 audio data

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
# INPUT CONFIGURATION: Define the text to convert to speech
# =============================================================================
# The text that will be converted to spoken audio
# Try modifying this text to generate different audio content
text = "Good morning. I think this is going to be a great presentation."

print(f"🎙️  Converting text to speech: '{text}'")
print("⏳ Generating audio... This may take a few seconds...")

# =============================================================================
# MODEL SETUP: Initialize the Text-to-Speech model
# =============================================================================
# Using OpenAI's TTS-1-HD model through Clarifai
# This model provides high-quality text-to-speech conversion
model_url = "https://clarifai.com/openai/tts/models/openai-tts-1-hd"

# Create model instance and generate audio from text
# predict_by_bytes() sends the text as bytes to the model
model_prediction = Model(url=model_url, pat=pat).predict_by_bytes(text.encode())

print("🔊 Audio generation complete! Saving to file...")

# =============================================================================
# AUDIO PROCESSING: Save the generated audio to a file
# =============================================================================
try:
    # The model returns audio data as base64-encoded format
    # We need to decode this and save it as a playable audio file
    
    # Extract the base64 audio data from the API response
    audio_base64 = model_prediction.outputs[0].data.audio.base64
    
    # Decode the base64 data to get the actual audio bytes
    audio_bytes = base64.b64decode(audio_base64)
    
    # Save the audio bytes to a WAV file
    # Using 'wb' mode (write binary) to properly save audio data
    output_filename = 'output_audio.wav'
    with open(output_filename, 'wb') as f:
        f.write(audio_bytes)
    
    # =============================================================================
    # SUCCESS CONFIRMATION: Provide feedback to user
    # =============================================================================
    print(f"✅ Audio successfully saved as '{output_filename}'")
    print(f"📁 You can find the audio file in the current directory")
    print(f"🎵 Play the audio file with any media player")
    
    # Display file size for reference
    file_size = len(audio_bytes)
    print(f"📊 Audio file size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

except Exception as e:
    print(f"❌ Error processing audio: {e}")
    print("🔧 Troubleshooting tips:")
    print("   • Check if the model response contains audio data")
    print("   • Verify your internet connection")
    print("   • Try a shorter text input")
    print("   • Check Clarifai service status")

print("\n💡 Understanding the process:")
print("   1. Text is sent to the TTS model")
print("   2. AI converts text to spoken audio")
print("   3. Audio is returned as base64-encoded data")
print("   4. Base64 data is decoded and saved as WAV file")

print("\n🚀 Try modifying this script:")
print("   • Change the 'text' variable to generate different speech")
print("   • Try longer texts like paragraphs or stories")
print("   • Experiment with different TTS models")
print("   • Add voice selection or audio format options")

print(f"\n📚 Learn more at: https://docs.clarifai.com/getting-started/quickstart")
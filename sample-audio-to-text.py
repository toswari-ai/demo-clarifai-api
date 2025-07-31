#!/usr/bin/env python3
"""
Clarifai Audio-to-Text (Speech Recognition) Demo

This script demonstrates how to use Clarifai's speech recognition models to convert
audio files into text transcriptions. This process is also known as speech-to-text
or automatic speech recognition (ASR).

The example uses AssemblyAI's audio transcription model available through Clarifai
to transcribe a sample audio file containing spoken words.

For more information about Clarifai's audio models, visit:
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
# This prevents accidentally sharing your API key when sharing code
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
# INPUT CONFIGURATION: Define the audio file to transcribe
# =============================================================================
# URL of the sample audio file we want to transcribe
# This audio file contains a "Good Morning" greeting that will be converted to text
audio_url = "https://s3.amazonaws.com/samples.clarifai.com/GoodMorning.wav"

print(f"🤖 Audio-to-Text Transcription")
print(f"🎧 Processing audio: {audio_url}")
print("📝 Converting speech to text...")

# =============================================================================
# MODEL SETUP: Initialize the speech recognition model
# =============================================================================
# Using AssemblyAI's audio transcription model, which is specifically designed
# for high-quality speech recognition and transcription
# This model can handle various audio formats and accents
model_url = "https://clarifai.com/assemblyai/speech-recognition/models/audio-transcription"

print("🧠 Initializing AssemblyAI speech recognition model...")

# Create model instance and make prediction with audio URL
# The predict_by_url() method sends the audio URL to the model for transcription
model_prediction = Model(url=model_url, pat=pat).predict_by_url(audio_url)

# =============================================================================
# RESULTS PROCESSING: Extract and display the transcription
# =============================================================================
print("\n" + "=" * 60)
print("🎯 Transcription Results:")
print("=" * 60)

try:
    # Extract the transcribed text from the model's response
    # The text is stored in the raw field of the text data
    transcribed_text = model_prediction.outputs[0].data.text.raw
    
    print(f"\n📝 Transcribed Text:")
    print(f'   "{transcribed_text}"')
    
    print(f"\n✅ Audio transcription complete!")
    
    # Display additional information about the transcription
    if transcribed_text:
        word_count = len(transcribed_text.split())
        char_count = len(transcribed_text)
        print(f"\n📊 Transcription Statistics:")
        print(f"   • Word count: {word_count}")
        print(f"   • Character count: {char_count}")
    
except Exception as e:
    print(f"❌ Error processing transcription: {e}")
    print("🔧 Troubleshooting tips:")
    print("   • Check if the audio URL is accessible")
    print("   • Verify the audio file format is supported (WAV, MP3, etc.)")
    print("   • Ensure your internet connection is stable")
    print("   • Try a different audio file or model")

# =============================================================================
# EDUCATIONAL INFORMATION AND NEXT STEPS
# =============================================================================
print(f"\n💡 Understanding speech recognition:")
print("   • Converts spoken language into written text")
print("   • Uses advanced AI models trained on vast amounts of audio data")
print("   • Can handle different accents, languages, and audio qualities")
print("   • Useful for transcription, voice commands, accessibility features")

print(f"\n🧠 About the AssemblyAI model:")
print("   • Developed by AssemblyAI, a leader in speech recognition technology")
print("   • Optimized for high accuracy across various audio conditions")
print("   • Supports multiple audio formats (WAV, MP3, FLAC, etc.)")
print("   • Can handle background noise and multiple speakers")

print(f"\n🚀 Try modifying this script:")
print("   • Change 'audio_url' to transcribe your own audio files")
print("   • Upload audio files to cloud storage and use their URLs")
print("   • Try different types of audio:")
print("     - Interviews or conversations")
print("     - Lectures or presentations")
print("     - Voice memos or recordings")
print("   • Experiment with different speech recognition models")
print("   • Add language detection or translation features")

print(f"\n📚 Learn more at: https://docs.clarifai.com/getting-started/quickstart")

# Raw output for debugging (optional)
print(f"\n🔧 Raw API Response (for developers):")
print(f"   Model used: AssemblyAI audio-transcription")
print(f"   Input audio URL: {audio_url}")
if 'model_prediction' in locals():
    print(f"   Response type: {type(model_prediction.outputs[0].data.text)}")
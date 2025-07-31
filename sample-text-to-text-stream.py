#!/usr/bin/env python3
"""
Clarifai Streaming Text Generation Demo

This script demonstrates how to use Clarifai's text generation models with streaming output.
Streaming means the AI's response appears word-by-word in real-time as it's being generated,
rather than waiting for the complete response. This creates a more interactive experience
similar to ChatGPT or other modern AI chat interfaces.

For more information about Clarifai's text models, visit:
https://docs.clarifai.com/resources/api-overview/python-sdk

Author: Clarifai
Last Updated: 2025
Requirements: clarifai>=11.6.0
"""

# Import necessary libraries
import os  # For accessing environment variables
from clarifai.client import Model  # Clarifai's Model class for API interactions

# =============================================================================
# SECURITY SETUP: Get API credentials from environment variables
# =============================================================================
# Instead of hardcoding your API key in the script (which is insecure), 
# we retrieve it from an environment variable. This is a security best practice.
# 
# To set your environment variable, run:
# Linux/Mac: export CLARIFAI_PAT="your_actual_api_key_here"
# Windows: set CLARIFAI_PAT=your_actual_api_key_here
pat = os.getenv('CLARIFAI_PAT')

# Check if the PAT was successfully retrieved
if not pat:
    raise ValueError(
        "❌ Please set the CLARIFAI_PAT environment variable\n"
        "Linux/Mac: export CLARIFAI_PAT='your_actual_api_key_here'\n"
        "Windows: set CLARIFAI_PAT=your_actual_api_key_here\n"
        "Get your PAT from: https://clarifai.com/settings/security"
    )

print("🤖 Initializing Clarifai AI model...")

# =============================================================================
# MODEL SETUP: Initialize the Clarifai text generation model
# =============================================================================
# We're using DeepSeek-R1, which is a powerful language model for text generation
# This model is particularly good at reasoning and providing detailed explanations
model = Model(
    url="https://clarifai.com/deepseek-ai/deepseek-chat/models/DeepSeek-R1-0528-Qwen3-8B",
    pat=pat  # Our authentication token
)

print("💭 Generating AI response with streaming output...")
print("=" * 60)

# =============================================================================
# TEXT GENERATION: Configure and execute the AI request
# =============================================================================
# The generate() method returns a streaming response - each iteration gives us
# a piece of the AI's response as it's being generated
response = model.generate(
    # The main question/request we're asking the AI
    prompt="What is the future of AI?", 
    
    # System prompt defines the AI's behavior and personality
    # This tells the AI how to respond (tone, style, expertise level)
    system_prompt="You are a helpful assistant.", 
    
    # Maximum number of tokens (words/parts of words) in the response
    # 1024 tokens ≈ 750-800 words (approximately)
    max_tokens=1024, 
    
    # Temperature controls creativity/randomness (0.0 = very focused, 1.0 = very creative)
    # 0.7 is a good balance between creativity and coherence
    temperature=0.7, 
    
    # Top-p controls diversity of word choices (0.1 = conservative, 1.0 = diverse)
    # 0.9 allows for diverse but still coherent responses
    top_p=0.9
)

# =============================================================================
# STREAMING OUTPUT: Process and display the response in real-time
# =============================================================================
print("🎯 AI Response:")
print("-" * 40)

# Process the streaming response
# Each iteration gives us a piece of the AI's response as it's being generated
for chunk in response:
    # Each chunk contains a piece of the AI's response text
    # We print it immediately to create the streaming effect
    if chunk:  # Only print non-empty chunks to avoid printing None values
        # Print parameters explained:
        # - chunk: the actual text content from the AI
        # - end='': don't add a newline after each chunk (so text flows together)
        # - flush=True: immediately display the text (don't wait for buffer to fill)
        print(chunk, end='', flush=True)

# =============================================================================
# COMPLETION: Clean up the output and provide feedback
# =============================================================================
# Add a clean ending to the output
print("\n" + "=" * 60)
print("✅ Streaming complete!")

print("\n💡 Understanding the parameters:")
print("   • prompt: The question or request sent to the AI")
print("   • system_prompt: Instructions that shape the AI's behavior")
print("   • max_tokens: Maximum length of the response (1024 ≈ 750-800 words)")
print("   • temperature: Creativity level (0.0 = focused, 1.0 = creative)")
print("   • top_p: Word choice diversity (0.1 = conservative, 1.0 = diverse)")

print("\n🚀 Try modifying this script:")
print("   • Change the 'prompt' to ask different questions")
print("   • Modify 'system_prompt' to change the AI's personality")
print("   • Adjust 'temperature' and 'top_p' to see how responses change")
print("   • Try different models from Clarifai's model gallery")

print(f"\n📚 Learn more at: https://docs.clarifai.com/resources/api-overview/python-sdk")
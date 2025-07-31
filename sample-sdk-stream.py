#!/usr/bin/env python3
"""
Clarifai SDK Streaming Text Generation Demo

This script demonstrates how to use Clarifai's native Python SDK for streaming text generation.
Streaming means the AI's response appears word-by-word in real-time as it's being generated,
creating a more interactive and responsive user experience similar to ChatGPT.

This example uses Google's Gemma-3-12B-IT model through Clarifai's native SDK,
showcasing the power of large language models for conversational AI.

For more information about Clarifai's Python SDK, visit:
https://docs.clarifai.com/resources/api-overview/python-sdk

Author: Clarifai
Last Updated: 2025
Requirements: clarifai>=11.6.0
"""

# Import necessary libraries
import os  # For accessing environment variables
from clarifai.client import Model  # Clarifai's native Model class for API interactions

# =============================================================================
# SECURITY SETUP: Get API credentials from environment variables
# =============================================================================
# Security best practice: Get PAT from environment variable instead of hardcoding it
# This keeps your secret API key safe and prevents accidental exposure in code sharing
pat = os.getenv('CLARIFAI_PAT')  # Retrieve the API key from environment variable

# Validate that the PAT was successfully retrieved
if not pat:  # Check if the API key was found
    # If no API key is found, stop the program with a helpful error message
    raise ValueError(
        "❌ Please set the CLARIFAI_PAT environment variable\n"
        "Linux/Mac: export CLARIFAI_PAT='your_actual_api_key_here'\n"
        "Windows: set CLARIFAI_PAT=your_actual_api_key_here\n"
        "Get your PAT from: https://clarifai.com/settings/security"
    )

print("🤖 Initializing Clarifai SDK for streaming text generation...")

# =============================================================================
# MODEL SETUP: Initialize the Clarifai model
# =============================================================================
# Create a Clarifai model instance using the native SDK
# This connects directly to the specified AI model hosted on Clarifai's platform
model = Model(
    # Google's Gemma-3-12B-IT model: A 12-billion parameter instruction-tuned model
    # "IT" stands for "Instruction Tuned" - optimized for following user prompts
    url="https://clarifai.com/gcp/generate/models/gemma-3-12b-it",
    pat=pat  # Use our securely retrieved Personal Access Token
)

print("🧠 Connected to Gemma-3-12B-IT model")
print("💭 Generating streaming AI response...")

# =============================================================================
# STREAMING TEXT GENERATION: Configure and execute the AI request
# =============================================================================
print("\n" + "=" * 60)
print("🎯 AI Response (streaming):")
print("=" * 60)

# Make a streaming request to the AI model using Clarifai's native generate() method
# The generate() method returns an iterator that yields text chunks as they're generated
response = model.generate(
    # The main question or prompt we want the AI to respond to
    prompt="What is the future of AI?",
    
    # System prompt defines the AI's behavior, personality, and response style
    # This instruction tells the AI how to act and respond
    system_prompt="You are a helpful assistant.",
    
    # Maximum number of tokens (words/word pieces) in the response
    # 1024 tokens ≈ 750-800 words approximately
    max_tokens=1024,
    
    # Temperature controls randomness/creativity in responses
    # 0.0 = very focused and deterministic, 1.0 = very creative and random
    # 0.7 provides a good balance between coherence and creativity
    temperature=0.7,
    
    # Top-p controls diversity of word choices (nucleus sampling)
    # 0.1 = very conservative word choices, 1.0 = considers all possible words
    # 0.9 allows for diverse but still coherent responses
    top_p=0.9
)

# =============================================================================
# STREAMING OUTPUT: Process and display the response in real-time
# =============================================================================
try:
    # Process the streaming response chunk by chunk
    # Each iteration gives us a piece of the AI's response as it's being generated
    for chunk in response:
        # Each chunk contains a piece of the AI's response text
        # We print it immediately to create the streaming effect
        if chunk:  # Only process non-empty chunks to avoid printing None values
            # Print parameters explained:
            # - chunk: the actual text content from the AI
            # - end='': don't add a newline after each chunk (so text flows together)
            # - flush=True: immediately display the text (don't wait for buffer to fill)
            print(chunk, end='', flush=True)

except Exception as e:
    print(f"\n❌ Error during streaming: {e}")
    print("🔧 Troubleshooting tips:")
    print("   • Check your internet connection")
    print("   • Verify your API key is valid and active")
    print("   • Try a simpler prompt or different model")

# =============================================================================
# COMPLETION: Clean up the output and provide feedback
# =============================================================================
# Add proper ending formatting to the streamed output
print("\n" + "=" * 60)
print("✅ Streaming complete!")

print("\n💡 Understanding streaming vs. non-streaming:")
print("   • Streaming: Text appears gradually as it's generated (like typing)")
print("   • Non-streaming: Complete response appears all at once")
print("   • Streaming provides better user experience for long responses")
print("   • Useful for chat interfaces, real-time applications")

print(f"\n🧠 About the Gemma-3-12B-IT model:")
print("   • Developed by Google for instruction following and conversation")
print("   • 12 billion parameters for high-quality text generation")
print("   • Instruction-tuned for better understanding of user requests")
print("   • Excellent balance of performance, speed, and response quality")

print(f"\n🚀 Try modifying this script:")
print("   • Change the 'prompt' to ask different questions")
print("   • Modify 'system_prompt' to change the AI's personality:")
print("     - 'You are a creative writer'")
print("     - 'You are a technical expert'")
print("     - 'You are a friendly teacher'")
print("   • Adjust parameters:")
print("     - Increase 'temperature' for more creative responses")
print("     - Lower 'temperature' for more focused answers")
print("     - Change 'max_tokens' to control response length")
print("   • Try different models from Clarifai's model gallery")

print(f"\n📚 Learn more at: https://docs.clarifai.com/resources/api-overview/python-sdk")
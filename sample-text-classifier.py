#!/usr/bin/env python3
"""
Clarifai Text Classification Demo

This script demonstrates how to use Clarifai's text classification models to analyze
and categorize text content. Text classification can identify sentiment, emotions,
topics, intentions, and other meaningful patterns in written text.

This example uses FinBERT, a specialized financial sentiment analysis model,
to analyze the sentiment of text input and classify it as positive, negative, or neutral.

For more information about Clarifai's text classification models, visit:
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
# INPUT CONFIGURATION: Define the text to analyze
# =============================================================================
# Sample text for sentiment analysis
# This positive message will be analyzed for emotional tone and sentiment
text = "Have a great day!"

print(f"📝 Text Classification Analysis")
print(f"💬 Text to analyze: '{text}'")
print("🧠 Analyzing sentiment and emotional tone...")

# =============================================================================
# MODEL SETUP: Initialize the text classification model
# =============================================================================
# Using FinBERT (Financial BERT), a specialized model for sentiment analysis
# Originally designed for financial text but works well for general sentiment analysis
# BERT stands for "Bidirectional Encoder Representations from Transformers"
model_url = "https://clarifai.com/clarifai/sentiment-analysis/models/finbert"

print("🤖 Initializing FinBERT sentiment analysis model...")

# Create model instance and make prediction with text input
# predict_by_bytes() requires text to be encoded as bytes
model_prediction = Model(url=model_url, pat=pat).predict_by_bytes(text.encode())

# =============================================================================
# RESULTS PROCESSING: Extract and display classification results
# =============================================================================
print("\n" + "=" * 60)
print("🎯 Text Classification Results:")
print("=" * 60)

try:
    # Extract the classification concepts from the prediction results
    # Using [-1] to get the last output, which typically contains the final results
    concepts = model_prediction.outputs[-1].data.concepts
    
    if concepts and len(concepts) > 0:
        print(f"📊 Sentiment Analysis Results:\n")
        
        # Sort concepts by confidence (highest first) for better readability
        sorted_concepts = sorted(concepts, key=lambda x: x.value, reverse=True)
        
        # Find the highest confidence classification
        top_sentiment = sorted_concepts[0]
        
        # Display all classifications with confidence scores
        for concept in sorted_concepts:
            confidence = concept.value * 100  # Convert to percentage
            sentiment = concept.name
            
            # Create a visual confidence bar
            bar_length = int(concept.value * 20)  # Scale to 20 characters
            confidence_bar = "█" * bar_length + "░" * (20 - bar_length)
            
            # Add emoji based on sentiment
            emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(sentiment.lower(), "📊")
            
            print(f"  {emoji} {sentiment:<12} | {confidence_bar} | {confidence:6.2f}%")
        
        # Provide interpretation
        print(f"\n🎯 Primary Classification: {top_sentiment.name.upper()}")
        print(f"   Confidence: {top_sentiment.value * 100:.2f}%")
        
        # Add contextual interpretation
        if top_sentiment.value > 0.8:
            certainty = "Very confident"
        elif top_sentiment.value > 0.6:
            certainty = "Confident"
        elif top_sentiment.value > 0.4:
            certainty = "Somewhat confident"
        else:
            certainty = "Low confidence"
        
        print(f"   Certainty level: {certainty}")
    
    else:
        print("⚠️  No classification results found.")
        print("   This might indicate:")
        print("   • The text is too ambiguous to classify")
        print("   • The model couldn't process the input")
        print("   • Try different text or a different model")

    # Display raw concepts for debugging
    print(f"\n🔧 Raw Classification Data (for developers):")
    print(f"   Number of concepts: {len(concepts) if concepts else 0}")
    print(f"   Raw concepts: {concepts}")

except Exception as e:
    print(f"❌ Error processing classification results: {e}")
    print("🔧 Troubleshooting tips:")
    print("   • Check if the text is properly encoded")
    print("   • Verify your internet connection")
    print("   • Try simpler text or a different model")
    print("   • Ensure the model response contains concept data")

# =============================================================================
# EDUCATIONAL INFORMATION AND NEXT STEPS
# =============================================================================
print("\n" + "=" * 60)
print("✅ Text classification analysis complete!")

print(f"\n💡 Understanding text classification:")
print("   • Analyzes text content to identify patterns, themes, or categories")
print("   • Sentiment analysis determines emotional tone (positive/negative/neutral)")
print("   • Confidence scores indicate how certain the AI is about each classification")
print("   • Multiple classifications can be returned with different confidence levels")

print(f"\n🧠 About FinBERT:")
print("   • Based on BERT (Bidirectional Encoder Representations from Transformers)")
print("   • Originally trained on financial text for financial sentiment analysis")
print("   • Works well for general sentiment analysis tasks")
print("   • Understands context and nuanced language patterns")

print(f"\n📊 Understanding confidence scores:")
print("   • Values range from 0.0 (not confident) to 1.0 (very confident)")
print("   • Higher scores indicate more certain classifications")
print("   • Multiple classifications may be returned for nuanced text")
print("   • Consider the top-scoring classification as the primary result")

print(f"\n🚀 Try modifying this script:")
print("   • Change the 'text' variable to analyze different messages:")
print("     - 'I love this product! It works perfectly.'")
print("     - 'This service is terrible and disappointing.'")
print("     - 'The weather is okay today.'")
print("   • Try different classification models:")
print("     - Topic classification")
print("     - Language detection")
print("     - Intent recognition")
print("   • Process multiple texts in a batch")
print("   • Add confidence threshold filtering")

print(f"\n📚 Learn more at: https://docs.clarifai.com/getting-started/quickstart")
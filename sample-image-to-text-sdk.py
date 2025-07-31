
# Clarifai Image Recognition Script
# This script analyzes an image and identifies objects/concepts in it
import os
from clarifai.client.model import Model

# Security: Get PAT from environment variable instead of hardcoding it
pat = os.getenv('CLARIFAI_PAT')
if not pat:
    raise ValueError("Please set the CLARIFAI_PAT environment variable")

# Image to analyze - Statue of Liberty example
image_url = "https://s3.amazonaws.com/samples.clarifai.com/featured-models/image-captioning-statue-of-liberty.jpeg"

# Clarifai's general image recognition model
model_url = "https://clarifai.com/clarifai/main/models/general-image-recognition"

# Create model instance and make prediction
model = Model(url=model_url, pat=pat)
model_prediction = model.predict_by_url(image_url, input_type="image")

# Pretty print the concepts and their confidence values
print("🖼️  Image Analysis Results:")
print("=" * 60)
print(f"📷 Image: {image_url}")
print("=" * 60)

# Extract concepts from the prediction
concepts = model_prediction.outputs[0].data.concepts

print("🎯 Detected Concepts:")
print("-" * 40)

# Sort concepts by confidence value (highest first)
sorted_concepts = sorted(concepts, key=lambda x: x.value, reverse=True)

# Print each concept with its confidence percentage
for concept in sorted_concepts:
    confidence_percent = concept.value * 100  # Convert to percentage
    print(f"   • {concept.name:<15} {confidence_percent:6.2f}%")

print("-" * 40)
print(f"✅ Found {len(concepts)} concepts total")






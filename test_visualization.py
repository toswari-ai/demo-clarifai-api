#!/usr/bin/env python3
"""
Quick test for visualization functions
"""

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import requests
from io import BytesIO

def test_image_download():
    """Test downloading and displaying an image"""
    image_url = "https://s3.amazonaws.com/samples.clarifai.com/people_walking2.jpeg"
    
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        
        # Display the image
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title('Test Image Download')
        plt.axis('off')
        plt.show()
        
        print("✅ Image download and display test successful!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing visualization components...")
    test_image_download()

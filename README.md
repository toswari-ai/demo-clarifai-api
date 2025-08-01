# Clarifai AI Models Demo Collection

This project demonstrates how to use various Clarifai AI models for different tasks including text generation, image processing, audio processing, and multimodal AI interactions. Perfect for beginners learning AI integration!

## What's in this project? 📁

### Text Processing Models

- `sample-text-to-text-stream.py` - Streaming text generation with AI chat models
- `sample-text-to-text-openai-api-stream.py` - Text generation using OpenAI-compatible API
- `sample-text-classifier.py` - Classify text into categories
- `sample-llm-sdk.py` - Large Language Model interactions
- `sample-sdk-stream.py` - Streaming SDK demonstrations
- `sample-openai-api.py` - OpenAI API compatibility examples

### Image Processing Models

- `sample-text-to-image.py` - Generate images from text descriptions (Imagen-2)
- `sample-text-to-image-dall-e.py` - Generate images using DALL-E models
- `sample-image-to-text-sdk.py` - Convert images to text descriptions
- `sample-visual-classifier.py` - Classify objects and scenes in images
- `sample-visual-detector.py` - Detect and locate objects in images
- `sample-visual-detector-viewer.py` - Interactive object detection visualization
- `sample-visual-segmenter.py` - Segment different parts of images
- `sample-visual-segmenter-viewer.py` - **NEW!** Interactive mask visualization with click-to-toggle legend
- `sample-super-resolution.py` - Enhance image quality and resolution

### Audio Processing Models

- `sample-text-to-audio.py` - Convert text to speech
- `sample-audio-to-text.py` - Convert speech to text (transcription)

### Multimodal Models

- `sample-multimodal-to-text.py` - Process both images and text together
- `sample-image-to-text.py` - Advanced image analysis and description

### Configuration Files

- `requirements.txt` - Lists all the Python packages you need
- `setup-env.sh` - Environment setup script
- `README.md` - This comprehensive guide

## Prerequisites 🛠️

Before you start, you'll need:

1. **Python 3.8+** installed on your computer
2. **A Clarifai account** and Personal Access Token (PAT)
   - Sign up at [clarifai.com](https://clarifai.com)
   - Get your PAT from your account settings at [clarifai.com/settings/security](https://clarifai.com/settings/security)
3. **Basic understanding of Python** (variables, functions, imports)
4. **A code editor** like VS Code (recommended) or any text editor

## Quick Start Guide 🚀

### 1. Clone or Download this project

```bash
# If using git
git clone <your-repo-url>
cd demo-clarifai-api

# Or just download the files to a folder called demo-clarifai-api
```

### 2. Install Python packages

The easiest way to install all required packages is using the `requirements.txt` file:

```bash
# Install all packages from requirements.txt (recommended)
pip install -r requirements.txt
```

**What this installs:**

- `clarifai>=11.6.0` - Official Clarifai Python SDK
- `openai>=1.0.0` - For OpenAI-compatible API access
- `Pillow>=10.0.0` - Image processing library
- `numpy` - Numerical computing (for image arrays)
- `opencv-python>=4.5.0` - Computer vision library (for advanced mask visualization)
- `matplotlib>=3.5.0` - Plotting library (for interactive visualizations)
- `requests` - HTTP library (for downloading images)

**Alternative installation methods:**

```bash
# If you prefer to install packages one by one
pip install clarifai>=11.6.0 openai>=1.0.0 Pillow>=10.0.0 numpy>=1.21.0 opencv-python>=4.5.0 matplotlib>=3.5.0 requests>=2.25.0

# For conda users (recommended for better compatibility)
conda create -n clarifai-demo python=3.11
conda activate clarifai-demo
pip install -r requirements.txt
```

### 3. Set up your API key (Important for Security!) 🔐

Instead of putting your API key directly in the code (which is unsafe), we use environment variables.

**Get your PAT:**

1. Go to [clarifai.com/settings/security](https://clarifai.com/settings/security)
2. Create a new Personal Access Token
3. Copy the token (it starts with something like `Key_...`)

**Set the environment variable:**

**On Linux/Mac (Terminal):**

```bash
export CLARIFAI_PAT="your_actual_api_key_here"

# To make it permanent, add to your ~/.bashrc or ~/.zshrc:
echo 'export CLARIFAI_PAT="your_actual_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

**On Windows (Command Prompt):**

```cmd
set CLARIFAI_PAT=your_actual_api_key_here
```

**On Windows (PowerShell):**

```powershell
$env:CLARIFAI_PAT="your_actual_api_key_here"
```

⚠️ **Replace `your_actual_api_key_here` with your real Clarifai PAT!**

### 4. Test your setup

Run a simple example to make sure everything works:

```bash
python sample-text-to-text-stream.py
```

You should see an AI response about the future of AI, appearing word by word!

## Model Examples & Usage 🤖

### Text Generation Models

#### Streaming Text Generation (`sample-text-to-text-stream.py`)

**What it does:** Generates text responses in real-time, showing each word as it's created.

```bash
python sample-text-to-text-stream.py
```

**Best for:**

- Chat applications
- Real-time Q&A systems
- Interactive AI assistants

**Key features:**

- Uses DeepSeek-R1 model
- Streaming output (words appear as generated)
- Customizable prompts and system messages

#### OpenAI-Compatible API (`sample-text-to-text-openai-api-stream.py`)

**What it does:** Same streaming text generation but using OpenAI's familiar API format.

```bash
python sample-text-to-text-openai-api-stream.py
```

**Best for:**

- Developers familiar with OpenAI API
- Easy migration from OpenAI to Clarifai
- ChatGPT-style applications

### Image Generation Models

#### Text-to-Image with Imagen-2 (`sample-text-to-image.py`)

**What it does:** Creates images from text descriptions using Google's Imagen-2 model.

```bash
python sample-text-to-image.py
```

**Example input:** "floor plan for 2 bedroom kitchen house"
**Output:** Saves `output.jpg` with the generated image

**Best for:**

- Architectural visualization
- Creative image generation
- Prototyping visual concepts

**Note:** Requires `opencv-python` for image processing.

#### Text-to-Image with DALL-E (`sample-text-to-image-dall-e.py`)

**What it does:** Creates images using DALL-E models.

```bash
python sample-text-to-image-dall-e.py
```

**Best for:**

- Artistic image creation
- Creative illustrations
- Marketing visuals

### Image Analysis Models

#### Visual Classification (`sample-visual-classifier.py`)

**What it does:** Identifies objects, scenes, and concepts in images.

```bash
python sample-visual-classifier.py
```

**Example:** Analyzes emotions in faces, identifies objects, classifies scenes.

**Best for:**

- Content moderation
- Automatic image tagging
- Scene understanding

#### Object Detection (`sample-visual-detector.py`)

**What it does:** Finds and locates specific objects in images with bounding boxes.

```bash
python sample-visual-detector.py
```

**Best for:**

- Security systems
- Inventory management
- Autonomous vehicles

#### Interactive Object Detection Viewer (`sample-visual-detector-viewer.py`) 🆕

**What it does:** Enhanced interactive visualization of object detection results with detailed analysis and export capabilities.

```bash
python sample-visual-detector-viewer.py
```

**Key Features:**

- **Visual Bounding Boxes**: Clear object boundaries with confidence scores
- **Detailed Object Analysis**: Comprehensive detection statistics and metrics
- **Export Capabilities**: Save results in multiple formats (JSON, images)
- **High-Quality Visualization**: Professional presentation of detection results
- **Real-time Analysis**: Immediate feedback on detection performance

**Best for:**

- Research and development in computer vision
- Quality assessment of detection models
- Educational demonstrations
- Professional analysis presentations

#### Image Segmentation (`sample-visual-segmenter.py`)

**What it does:** Separates different parts of an image (like separating people from background).

```bash
python sample-visual-segmenter.py
```

**Example output:**

```text
person: 0.9876
background: 0.8765
clothing: 0.7654
```

**Best for:**

- Image editing applications
- Medical imaging
- Photo manipulation

#### Interactive Mask Visualization (`sample-visual-segmenter-viewer.py`) 🆕

**What it does:** Advanced interactive visualization of image segmentation with clickable legend, confidence-based filtering, and real-time mask toggling.

```bash
python sample-visual-segmenter-viewer.py
```

**Key Features:**

- **Interactive Legend**: Click legend items to show/hide individual masks
- **Confidence Filtering**: Only shows masks with confidence ≥ 0.6 by default
- **High-Contrast Visualization**: Enhanced colors and contrast for better visibility
- **Real-time Updates**: Dynamic mask toggling with visual feedback (✅/❌)
- **Professional Styling**: Clean interface with confidence scores displayed
- **Keyboard Shortcuts**: Press 'q' to close, 's' to save current view

**Example output:**

```text
🎭 Mask 1: sky-other (0.853) - Color: RGB(255, 0, 0)
🎭 Mask 2: airplane (0.116) - Color: RGB(0, 0, 255)
✅ High-confidence masks (≥0.6) are visible by default, low-confidence masks are hidden
💡 Click legend items to show/hide individual masks
```

**Best for:**

- Research and analysis of segmentation results
- Quality assessment of AI predictions
- Educational demonstrations of computer vision
- Professional presentation of segmentation data

### Audio Processing Models

#### Text-to-Speech (`sample-text-to-audio.py`)

**What it does:** Converts written text into spoken audio.

```bash
python sample-text-to-audio.py
```

**Best for:**

- Accessibility applications
- Voice assistants
- Audio content creation

#### Speech-to-Text (`sample-audio-to-text.py`)

**What it does:** Transcribes spoken audio into written text.

```bash
python sample-audio-to-text.py
```

**Best for:**

- Meeting transcription
- Voice commands
- Accessibility features

### Advanced Models

#### Super Resolution (`sample-super-resolution.py`)

**What it does:** Enhances image quality and increases resolution.

```bash
python sample-super-resolution.py
```

**Input:** Low-resolution image
**Output:** High-resolution enhanced image saved as `tmp.jpg`

**Best for:**

- Photo enhancement
- Old photo restoration
- Image quality improvement

#### Multimodal Processing (`sample-multimodal-to-text.py`)

**What it does:** Processes both images and text together for comprehensive understanding.

```bash
python sample-multimodal-to-text.py
```

**Best for:**

- Document analysis with images
- Visual question answering
- Complex scene understanding

## Troubleshooting 🔧

### Common Installation Issues

#### "Import could not be resolved" or "ModuleNotFoundError"

**Problem:** Python can't find the required packages.

**Solutions:**

```bash
# 1. Make sure you installed the packages
pip install -r requirements.txt

# 2. Check if packages are installed
pip list | grep -E "(clarifai|openai|opencv|numpy)"

# 3. If using conda, make sure you're in the right environment
conda activate your_environment_name
pip install -r requirements.txt

# 4. Try upgrading pip first
pip install --upgrade pip
pip install -r requirements.txt
```

#### "ModuleNotFoundError: No module named 'cv2'"

**Problem:** OpenCV (cv2) is not installed properly.

**Solutions:**

```bash
# Try these installation methods in order:
pip install opencv-python
# or
pip install opencv-contrib-python
# or
conda install -c conda-forge opencv
```

### API and Authentication Issues

#### "Please set the CLARIFAI_PAT environment variable"

**Problem:** Your API key is not set as an environment variable.

**Solutions:**

```bash
# Linux/Mac
export CLARIFAI_PAT="your_actual_api_key_here"

# Windows Command Prompt
set CLARIFAI_PAT=your_actual_api_key_here

# Windows PowerShell
$env:CLARIFAI_PAT="your_actual_api_key_here"

# Make it permanent by adding to your shell config file
echo 'export CLARIFAI_PAT="your_key_here"' >> ~/.bashrc
source ~/.bashrc
```

#### "Authentication failed" or "Invalid token"

**Problem:** Your API key is incorrect or expired.

**Solutions:**

1. Get a new PAT from [clarifai.com/settings/security](https://clarifai.com/settings/security)
2. Make sure you copied the entire key (starts with `Key_...`)
3. Check for extra spaces or characters when setting the environment variable
4. Verify your Clarifai account is active and has API access

### Runtime Errors

#### "IndexError: list index out of range"

**Problem:** The AI model didn't return expected results.

**Solutions:**

1. Check if the image URL is accessible
2. Try a different image or model
3. Verify your internet connection
4. Look at the raw API response for debugging

#### "Network errors" or "Connection timeout"

**Problem:** Can't connect to Clarifai's servers.

**Solutions:**

1. Check your internet connection
2. Try again in a few minutes (server might be busy)
3. Check if your firewall is blocking the connection
4. Verify Clarifai's service status

### Image Processing Issues

#### Generated images are corrupted or won't open

**Problem:** Image conversion failed during processing.

**Solutions:**

```python
# Add error checking to your image processing:
if img_np is not None and img_np.size > 0:
    cv2.imwrite("output.jpg", img_np)
    print("✅ Image saved successfully")
else:
    print("❌ Failed to process image data")
```

#### "Permission denied" when saving files

**Problem:** Don't have write permissions in the current directory.

**Solutions:**

```bash
# Check current directory permissions
ls -la

# Change to a directory where you have write permissions
cd ~/Documents  # or any folder you own

# Run the script from there
python /path/to/your/script.py
```

### Performance Issues

#### Scripts running very slowly

**Causes and Solutions:**

1. **Large images:** Resize images before processing
2. **Slow internet:** Use local images when possible
3. **Model complexity:** Some models take longer than others
4. **Server load:** Try running at different times

#### Out of memory errors

**Solutions:**

```bash
# For large image processing, try:
pip install opencv-python-headless  # Lighter version
# or reduce image size in your code before processing
```

### Getting Help

If you're still having issues:

1. **Check the error message carefully** - it often tells you exactly what's wrong
2. **Read the full traceback** - the error location helps identify the problem
3. **Try the examples in order** - start with simple text generation before complex image processing
4. **Check Clarifai's status page** - services might be temporarily down
5. **Visit the documentation** - [docs.clarifai.com](https://docs.clarifai.com)

### Debugging Tips

#### Enable verbose output

```python
# Add print statements to see what's happening:
print(f"PAT loaded: {'Yes' if pat else 'No'}")
print(f"Image URL: {image_url}")
print(f"Model prediction type: {type(model_prediction)}")
```

#### Check API responses

```python
# Before processing results, check if they exist:
if model_prediction.outputs and len(model_prediction.outputs) > 0:
    # Process results
    pass
else:
    print("No outputs received from the model")
```

## Code References & Examples 💻

### Quick Start Code Snippets

#### Basic Text Generation

```python
import os
from clarifai.client.model import Model

# Initialize the model
model = Model(
    url="https://clarifai.com/meta/Llama-2/models/llama2-7b-chat",
    pat=os.environ.get("CLARIFAI_PAT")
)

# Generate text
model_prediction = model.predict_by_bytes(
    b"Explain quantum computing in simple terms",
    input_type="text"
)

print(model_prediction.outputs[0].data.text.raw)
```

#### Image Classification

```python
from clarifai.client.model import Model

model = Model(
    url="https://clarifai.com/clarifai/main/models/general-image-recognition",
    pat=os.environ.get("CLARIFAI_PAT")
)

# Classify image from URL
model_prediction = model.predict_by_url(
    "https://samples.clarifai.com/metro-north.jpg",
    input_type="image"
)

# Print concepts
for concept in model_prediction.outputs[0].data.concepts:
    print(f"{concept.name}: {concept.value:.4f}")
```

#### Streaming Text Generation

```python
import os
from clarifai.client.model import Model

model = Model(
    url="https://clarifai.com/deepseek-ai/deepseek-r1/models/deepseek-r1",
    pat=os.environ.get("CLARIFAI_PAT")
)

# Stream text generation
for chunk in model.stream_predict_by_bytes(
    b"Write a haiku about AI",
    input_type="text"
):
    if chunk.outputs and chunk.outputs[0].data.text.raw:
        print(chunk.outputs[0].data.text.raw, end='', flush=True)
```

### Advanced Integration Examples

#### Custom Error Handling

```python
import os
from clarifai.client.model import Model
from clarifai.client.auth.helper import ClarifaiAuthHelper

def safe_predict(model_url, input_data, input_type="text"):
    """Wrapper function with comprehensive error handling"""
    try:
        # Verify authentication
        auth = ClarifaiAuthHelper.from_env()
        if not auth.pat:
            raise ValueError("CLARIFAI_PAT environment variable not set")
        
        model = Model(url=model_url, pat=auth.pat)
        
        if input_type == "text":
            prediction = model.predict_by_bytes(input_data.encode(), input_type)
        elif input_type == "image":
            prediction = model.predict_by_url(input_data, input_type)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
            
        return prediction
        
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        return None

# Usage example
result = safe_predict(
    "https://clarifai.com/meta/Llama-2/models/llama2-7b-chat",
    "What is machine learning?",
    "text"
)

if result and result.outputs:
    print(result.outputs[0].data.text.raw)
```

#### Batch Processing Images

```python
import os
from clarifai.client.model import Model
from concurrent.futures import ThreadPoolExecutor
import time

def process_image_batch(image_urls, model_url, max_workers=5):
    """Process multiple images concurrently"""
    model = Model(url=model_url, pat=os.environ.get("CLARIFAI_PAT"))
    
    def classify_single_image(url):
        try:
            prediction = model.predict_by_url(url, input_type="image")
            concepts = prediction.outputs[0].data.concepts[:3]  # Top 3
            return {
                'url': url,
                'concepts': [(c.name, c.value) for c in concepts],
                'status': 'success'
            }
        except Exception as e:
            return {'url': url, 'error': str(e), 'status': 'failed'}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(classify_single_image, image_urls))
    
    return results

# Example usage
image_urls = [
    "https://samples.clarifai.com/metro-north.jpg",
    "https://samples.clarifai.com/wedding.jpg",
    "https://samples.clarifai.com/dog.tiff"
]

results = process_image_batch(
    image_urls,
    "https://clarifai.com/clarifai/main/models/general-image-recognition"
)

for result in results:
    if result['status'] == 'success':
        print(f"✅ {result['url']}")
        for name, confidence in result['concepts']:
            print(f"   - {name}: {confidence:.3f}")
    else:
        print(f"❌ {result['url']}: {result['error']}")
```

#### Configuration Management

```python
import os
import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class ClarifaiConfig:
    """Configuration management for Clarifai applications"""
    pat: str
    base_url: str = "https://api.clarifai.com"
    timeout: int = 30
    max_retries: int = 3
    
    @classmethod
    def from_env(cls) -> 'ClarifaiConfig':
        """Load configuration from environment variables"""
        pat = os.environ.get("CLARIFAI_PAT")
        if not pat:
            raise ValueError("CLARIFAI_PAT environment variable required")
        
        return cls(
            pat=pat,
            base_url=os.environ.get("CLARIFAI_BASE_URL", cls.base_url),
            timeout=int(os.environ.get("CLARIFAI_TIMEOUT", cls.timeout)),
            max_retries=int(os.environ.get("CLARIFAI_MAX_RETRIES", cls.max_retries))
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ClarifaiConfig':
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)
    
    def to_file(self, config_path: str) -> None:
        """Save configuration to JSON file (excluding PAT for security)"""
        config_data = {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

# Usage examples
config = ClarifaiConfig.from_env()
print(f"Using PAT: {config.pat[:10]}...")
```

### Model-Specific Examples

#### Working with Different Model Types

**Text Models:**

```python
# Chat/Conversation Models
CHAT_MODELS = {
    "llama2": "https://clarifai.com/meta/Llama-2/models/llama2-7b-chat",
    "deepseek": "https://clarifai.com/deepseek-ai/deepseek-r1/models/deepseek-r1",
    "gpt4": "https://clarifai.com/openai/chat-completion/models/GPT-4"
}

# Text Classification Models
CLASSIFICATION_MODELS = {
    "sentiment": "https://clarifai.com/clarifai/main/models/sentiment-analysis-twitter-roberta-base",
    "toxicity": "https://clarifai.com/martin-danelljan/text_classification/models/toxic-text-detection-roberta"
}
```

**Vision Models:**

```python
# Image Analysis Models
VISION_MODELS = {
    "general": "https://clarifai.com/clarifai/main/models/general-image-recognition",
    "faces": "https://clarifai.com/clarifai/main/models/face-detection",
    "objects": "https://clarifai.com/clarifai/main/models/general-image-detection",
    "segmentation": "https://clarifai.com/clarifai/main/models/image-segmentation"
}

# Image Generation Models
GENERATION_MODELS = {
    "stable_diffusion": "https://clarifai.com/stability-ai/stable-diffusion-2/models/stable-diffusion-xl",
    "dall_e": "https://clarifai.com/openai/dall-e/models/dall-e-3",
    "imagen": "https://clarifai.com/google/imagen/models/imagen-2"
}
```

#### Interactive Visualization Template

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import cv2

class InteractiveVisualizer:
    """Template for creating interactive visualizations"""
    
    def __init__(self, image_path: str, predictions: list):
        self.image = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.predictions = predictions
        self.visible_masks = [True] * len(predictions)
        
        self.setup_plot()
    
    def setup_plot(self):
        """Initialize the matplotlib plot with interactive elements"""
        self.fig, (self.ax_img, self.ax_legend) = plt.subplots(
            1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [3, 1]}
        )
        
        # Display image
        self.ax_img.imshow(self.image_rgb)
        self.ax_img.set_title("Interactive Visualization")
        self.ax_img.axis('off')
        
        # Setup legend with buttons
        self.setup_legend()
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def setup_legend(self):
        """Create interactive legend with toggle buttons"""
        self.ax_legend.clear()
        self.ax_legend.set_xlim(0, 1)
        self.ax_legend.set_ylim(0, len(self.predictions))
        
        for i, pred in enumerate(self.predictions):
            y_pos = len(self.predictions) - i - 1
            status = "✅" if self.visible_masks[i] else "❌"
            confidence = pred.get('confidence', 0.0)
            
            self.ax_legend.text(
                0.05, y_pos + 0.3,
                f"{status} {pred['name']} ({confidence:.3f})",
                fontsize=10,
                transform=self.ax_legend.transData
            )
        
        self.ax_legend.set_title("Click to Toggle")
        self.ax_legend.axis('off')
    
    def on_click(self, event):
        """Handle click events for toggling visibility"""
        if event.inaxes == self.ax_legend:
            # Calculate which item was clicked
            y_click = event.ydata
            item_index = len(self.predictions) - int(y_click) - 1
            
            if 0 <= item_index < len(self.predictions):
                self.visible_masks[item_index] = not self.visible_masks[item_index]
                self.update_display()
    
    def update_display(self):
        """Update the visualization based on current visibility settings"""
        self.setup_legend()
        self.fig.canvas.draw()
    
    def show(self):
        """Display the interactive visualization"""
        plt.show()

# Usage example
# visualizer = InteractiveVisualizer("image.jpg", predictions)
# visualizer.show()
```

### API Integration Patterns

#### REST API Client

```python
import requests
import json
import base64
from typing import Dict, Any, Optional

class ClarifaiAPIClient:
    """Direct REST API client for advanced use cases"""
    
    def __init__(self, pat: str, base_url: str = "https://api.clarifai.com"):
        self.pat = pat
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Key {pat}",
            "Content-Type": "application/json"
        }
    
    def predict_text(self, model_id: str, text: str, user_id: str = "clarifai") -> Dict[Any, Any]:
        """Make text prediction using REST API"""
        url = f"{self.base_url}/v2/models/{model_id}/outputs"
        
        payload = {
            "inputs": [{
                "data": {
                    "text": {
                        "raw": text
                    }
                }
            }]
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()
    
    def predict_image_url(self, model_id: str, image_url: str) -> Dict[Any, Any]:
        """Make image prediction using image URL"""
        url = f"{self.base_url}/v2/models/{model_id}/outputs"
        
        payload = {
            "inputs": [{
                "data": {
                    "image": {
                        "url": image_url
                    }
                }
            }]
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()
    
    def predict_image_bytes(self, model_id: str, image_bytes: bytes) -> Dict[Any, Any]:
        """Make image prediction using image bytes"""
        url = f"{self.base_url}/v2/models/{model_id}/outputs"
        
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        payload = {
            "inputs": [{
                "data": {
                    "image": {
                        "base64": base64_image
                    }
                }
            }]
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

# Usage example
client = ClarifaiAPIClient(os.environ.get("CLARIFAI_PAT"))
result = client.predict_text("llama2-7b-chat", "Hello, world!")
print(json.dumps(result, indent=2))
```

#### Webhook Integration

```python
from flask import Flask, request, jsonify
import hmac
import hashlib
import json

app = Flask(__name__)
WEBHOOK_SECRET = os.environ.get("CLARIFAI_WEBHOOK_SECRET")

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    """Handle Clarifai webhook notifications"""
    
    # Verify webhook signature
    signature = request.headers.get('X-Clarifai-Signature')
    if not verify_webhook_signature(request.data, signature):
        return jsonify({'error': 'Invalid signature'}), 401
    
    # Process webhook data
    webhook_data = request.get_json()
    
    if webhook_data.get('type') == 'model.training.completed':
        handle_training_completed(webhook_data)
    elif webhook_data.get('type') == 'input.uploaded':
        handle_input_uploaded(webhook_data)
    
    return jsonify({'status': 'success'})

def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """Verify webhook signature for security"""
    if not WEBHOOK_SECRET or not signature:
        return False
    
    expected_signature = hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(f"sha256={expected_signature}", signature)

def handle_training_completed(data: dict):
    """Handle model training completion"""
    model_id = data.get('model_id')
    print(f"✅ Model {model_id} training completed!")
    
    # Add your custom logic here
    # e.g., send notification, update database, trigger next workflow

def handle_input_uploaded(data: dict):
    """Handle new input upload"""
    input_id = data.get('input_id')
    print(f"📁 New input uploaded: {input_id}")
    
    # Add your custom logic here
    # e.g., trigger automatic processing, update UI

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### Performance Optimization Tips

#### Caching Predictions

```python
import functools
import hashlib
import pickle
import os
from typing import Any, Callable

def cache_predictions(cache_dir: str = ".cache"):
    """Decorator to cache model predictions"""
    os.makedirs(cache_dir, exist_ok=True)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            cache_key = hashlib.md5(
                str(args).encode() + str(kwargs).encode()
            ).hexdigest()
            
            cache_file = os.path.join(cache_dir, f"{func.__name__}_{cache_key}.pkl")
            
            # Try to load from cache
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    pass  # Cache corrupted, will regenerate
            
            # Generate new result
            result = func(*args, **kwargs)
            
            # Save to cache
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except Exception:
                pass  # Failed to cache, continue anyway
            
            return result
        return wrapper
    return decorator

# Usage example
@cache_predictions()
def expensive_prediction(model_url: str, input_text: str):
    model = Model(url=model_url, pat=os.environ.get("CLARIFAI_PAT"))
    return model.predict_by_bytes(input_text.encode(), input_type="text")
```

#### Async Processing

```python
import asyncio
import aiohttp
import json
from typing import List, Dict, Any

class AsyncClarifaiClient:
    """Asynchronous Clarifai client for high-throughput applications"""
    
    def __init__(self, pat: str, base_url: str = "https://api.clarifai.com"):
        self.pat = pat
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Key {pat}",
            "Content-Type": "application/json"
        }
    
    async def predict_batch_async(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple predictions concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._single_prediction(session, req) 
                for req in requests
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
    
    async def _single_prediction(self, session: aiohttp.ClientSession, req: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single prediction request"""
        url = f"{self.base_url}/v2/models/{req['model_id']}/outputs"
        
        async with session.post(url, headers=self.headers, json=req['payload']) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {
                    'error': f"HTTP {response.status}",
                    'model_id': req['model_id']
                }

# Usage example
async def process_large_batch():
    client = AsyncClarifaiClient(os.environ.get("CLARIFAI_PAT"))
    
    requests = [
        {
            'model_id': 'general-image-recognition',
            'payload': {
                'inputs': [{'data': {'image': {'url': f'https://example.com/image{i}.jpg'}}}]
            }
        }
        for i in range(100)  # Process 100 images concurrently
    ]
    
    results = await client.predict_batch_async(requests)
    
    # Process results
    successful = [r for r in results if not isinstance(r, Exception) and 'error' not in r]
    failed = [r for r in results if isinstance(r, Exception) or 'error' in r]
    
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    return successful, failed

# Run the async function
# asyncio.run(process_large_batch())
```

## Learning Resources 📚

### Official Documentation

- **Clarifai Docs:** [docs.clarifai.com](https://docs.clarifai.com/)
- **Python SDK API Reference:** [Python Client Guide](https://docs.clarifai.com/resources/api-references/python)
- **Compute & Inference:** [Model Inference Guide](https://docs.clarifai.com/compute/inference/)
- **Model Explorer:** [Browse Available Models](https://clarifai.com/explore/models) - Discover thousands of pre-trained models
- **Community Resources:** [Clarifai Community](https://community.clarifai.com/)
- **Stack Overflow:** [Clarifai Questions](https://stackoverflow.com/questions/tagged/clarifai)

### Tutorials and Examples

- **Quick Start:** [Getting Started with Clarifai](https://docs.clarifai.com/)
- **Model Usage:** [Working with Models](https://docs.clarifai.com/resources/api-references/python/#model)
- **Dataset Management:** [Dataset Operations](https://docs.clarifai.com/resources/api-references/python/#dataset)
- **Input Processing:** [Input Handling](https://docs.clarifai.com/resources/api-references/python/#input)

### Community and Support

- **Community Forum:** [community.clarifai.com](https://community.clarifai.com/)
- **GitHub Issues:** Report bugs and request features
- **Discord Server:** Join the Clarifai developer community

### Additional Resources

#### Books and Courses

- **"Hands-On Machine Learning"** by Aurélien Géron - Comprehensive ML guide
- **"Deep Learning with Python"** by François Chollet - Neural networks fundamentals
- **Coursera AI Courses** - University-level AI education
- **Fast.ai Practical Deep Learning** - Hands-on deep learning approach

#### Research Papers and Articles

- **Vision Transformers (ViT)** - Understanding modern computer vision
- **CLIP: Learning Transferable Visual Models** - Multimodal AI foundations
- **GPT Architecture Papers** - Language model fundamentals
- **Stable Diffusion Research** - Image generation techniques

#### Development Tools and Libraries

- **Hugging Face Transformers** - Pre-trained model library
- **LangChain** - LLM application framework
- **Streamlit** - Rapid web app development for AI demos
- **Gradio** - Interactive ML model interfaces
- **Weights & Biases** - Experiment tracking and visualization

## Project Structure & Best Practices 🏗️

### Recommended Project Organization

```
clarifai-project/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py          # Configuration management
│   │   └── models.py            # Model URLs and settings
│   ├── clients/
│   │   ├── __init__.py
│   │   ├── clarifai_client.py   # Custom client wrapper
│   │   └── cache.py             # Caching utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── text_models.py       # Text processing models
│   │   ├── vision_models.py     # Image processing models
│   │   └── audio_models.py      # Audio processing models
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_utils.py       # Image preprocessing
│   │   ├── text_utils.py        # Text preprocessing
│   │   └── visualization.py     # Plotting utilities
│   └── main.py                  # Main application entry
├── tests/
│   ├── __init__.py
│   ├── test_clients.py
│   ├── test_models.py
│   └── test_utils.py
├── examples/                    # Demo scripts (like this repo)
├── data/
│   ├── input/                   # Input files
│   ├── output/                  # Generated results
│   └── cache/                   # Cached predictions
├── docs/
│   ├── API.md                   # API documentation
│   ├── SETUP.md                 # Setup instructions
│   └── EXAMPLES.md              # Usage examples
├── requirements.txt
├── requirements-dev.txt         # Development dependencies
├── .env.example                 # Environment template
├── .gitignore
└── README.md
```

### Essential Development Patterns

#### Configuration Management Example

```python
# config/settings.py
import os
from typing import Optional
from pydantic import BaseSettings, Field

class ClarifaiSettings(BaseSettings):
    """Application settings with validation"""
    
    # API Configuration
    pat: str = Field(..., env="CLARIFAI_PAT")
    base_url: str = Field("https://api.clarifai.com", env="CLARIFAI_BASE_URL")
    timeout: int = Field(30, env="CLARIFAI_TIMEOUT")
    max_retries: int = Field(3, env="CLARIFAI_MAX_RETRIES")
    
    # Processing Configuration
    confidence_threshold: float = Field(0.6, env="CONFIDENCE_THRESHOLD")
    max_concurrent: int = Field(5, env="MAX_CONCURRENT")
    
    class Config:
        env_file = ".env"

settings = ClarifaiSettings()
```

#### Production Error Handling

```python
# utils/error_handling.py
import logging
from functools import wraps
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)

def handle_api_errors(default_return: Any = None):
    """Decorator for robust API error handling"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"API error in {func.__name__}: {str(e)}")
                if default_return is not None:
                    return default_return
                raise
        return wrapper
    return decorator

# Usage
@handle_api_errors(default_return=[])
def get_image_classifications(image_url: str) -> list:
    # Your classification logic here
    pass
```

#### Testing Best Practices

```python
# tests/test_models.py
import pytest
from unittest.mock import Mock, patch
from src.clients.clarifai_client import ClarifaiClient

class TestClarifaiIntegration:
    """Comprehensive test suite for Clarifai integration"""
    
    @pytest.fixture
    def mock_client(self):
        return Mock(spec=ClarifaiClient)
    
    @pytest.mark.parametrize("confidence,expected", [
        (0.95, True),
        (0.5, False),
        (0.6, True),
    ])
    def test_confidence_filtering(self, confidence, expected):
        """Test confidence threshold filtering"""
        result = confidence >= 0.6
        assert result == expected
    
    @patch('clarifai.client.model.Model')
    def test_model_prediction_success(self, mock_model):
        """Test successful model prediction"""
        # Setup mock
        mock_response = Mock()
        mock_response.outputs = [Mock()]
        mock_response.outputs[0].data.text.raw = "Test response"
        mock_model.return_value.predict_by_bytes.return_value = mock_response
        
        # Test
        client = ClarifaiClient("test_pat")
        result = client.predict_text("test_model", "test_input")
        
        # Assert
        assert result == "Test response"
        mock_model.return_value.predict_by_bytes.assert_called_once()
```

## What's Next? 🚀

Once you understand these examples, you can:

### Build Real Applications

- **Chatbot:** Use text generation models for conversational AI
- **Content Moderation:** Automatically detect inappropriate content
- **Image Search:** Build visual search engines
- **Voice Assistant:** Combine text-to-speech and speech-to-text
- **Creative Tools:** Build image generation applications
- **Document Processing:** Extract text and analyze documents

### Advanced Features to Explore

- **Custom Model Training:** Train models on your own data
- **Workflow Automation:** Chain multiple models together
- **Real-time Processing:** Build streaming applications
- **Mobile Integration:** Use Clarifai in mobile apps
- **Edge Deployment:** Run models locally

### Best Practices

- **Security:** Always use environment variables for API keys
- **Error Handling:** Add proper error checking to your applications
- **Performance:** Optimize for your specific use case
- **Monitoring:** Track usage and performance metrics
- **Testing:** Test with various inputs and edge cases

## Contributing 🤝

Found a bug or want to improve these examples?

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting 🛠️

### Common Issues and Solutions

#### **ImportError: No module named 'clarifai'**

**Problem:** Python packages not installed correctly.

**Solutions:**

```bash
# Make sure you're in the right environment
conda activate clarifai-demo  # if using conda

# Reinstall requirements
pip install -r requirements.txt

# Or install clarifai specifically
pip install clarifai>=11.6.0
```

#### **Authentication Error: Invalid PAT**

**Problem:** API key issues.

**Solutions:**

```bash
# Check if environment variable is set
echo $CLARIFAI_PAT  # Linux/Mac
echo %CLARIFAI_PAT%  # Windows

# Make sure PAT starts with "Key_"
# Get a new PAT from https://clarifai.com/settings/security
export CLARIFAI_PAT="your_new_pat_here"
```

#### **matplotlib/GUI Backend Issues**

**Problem:** Visualization windows not showing.

**Solutions:**

```bash
# For Linux systems
sudo apt-get install python3-tk

# For conda environments
conda install tk

# For headless servers, use non-interactive backend
export MPLBACKEND=Agg
```

#### **OpenCV Installation Issues**

**Problem:** `cv2` import errors.

**Solutions:**

```bash
# Uninstall conflicting versions
pip uninstall opencv-python opencv-contrib-python opencv-headless

# Reinstall clean version
pip install opencv-python>=4.5.0
```

#### **Memory Issues with Large Images**

**Problem:** Out of memory errors.

**Solutions:**

- Use smaller images (< 2048x2048 recommended)
- Close visualization windows when done
- Restart Python session between large image processing

### Getting Help

- **Clarifai Documentation**: [docs.clarifai.com](https://docs.clarifai.com)
- **Community Forum**: [community.clarifai.com](https://community.clarifai.com)
- **GitHub Issues**: Report bugs in this repository
- **API Status**: [status.clarifai.com](https://status.clarifai.com)

## Security Note 🔒

**Never commit API keys to version control!**

Always use environment variables or secure config files that are excluded from your repository. Add your API keys to `.gitignore`:

```gitignore
# API keys and secrets
.env
config.py
secrets.json
```

## Happy Coding with Clarifai! 🎉

This demo collection provides everything you need to get started with Clarifai's powerful AI models. Each script is self-contained and well-documented, making it easy to understand and modify for your specific needs.

**Key Features of This Collection:**

- ✅ **Production-ready code** with proper error handling
- ✅ **Interactive visualizations** for computer vision tasks
- ✅ **Comprehensive documentation** and examples
- ✅ **Security best practices** with environment variables
- ✅ **Cross-platform compatibility** (Windows, macOS, Linux)
- ✅ **Easy setup** with conda/pip support

If you build something cool with these examples, we'd love to see it! Share your projects with the community and help others learn AI development.

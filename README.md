# Clarifai AI Models Demo Collection

This project demonstrates how to use various Clarifai AI models for different tasks including text generation, image processing, audio processing, and multimodal AI interactions. Perfect for beginners learning AI integration!

## What's in this project? 📁

### Text Processing Models

- `sample-text-to-text-stream.py` - Streaming text generation with AI chat models
- `sample-text-to-text-openai-api-stream.py` - Text generation using OpenAI-compatible API
- `sample-text-classifier.py` - Classify text into categories
- `sample-llm-sdk.py` - Large Language Model interactions

### Image Processing Models

- `sample-text-to-image.py` - Generate images from text descriptions (Imagen-2)
- `sample-text-to-image-dall-e.py` - Generate images using DALL-E models
- `sample-image-to-text-sdk.py` - Convert images to text descriptions
- `sample-visual-classifier.py` - Classify objects and scenes in images
- `sample-visual-detector.py` - Detect and locate objects in images
- `sample-visual-segmenter.py` - Segment different parts of images
- `sample-super-resolution.py` - Enhance image quality and resolution

### Audio Processing Models

- `sample-text-to-audio.py` - Convert text to speech
- `Audio To Text.py` - Convert speech to text (transcription)

### Multimodal Models

- `Multimodal To Text.py` - Process both images and text together
- `Image To Text.py` - Advanced image analysis and description

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
- `opencv-python` - Computer vision library (for image processing)

**Alternative installation methods:**

```bash
# If you prefer to install packages one by one
pip install clarifai openai Pillow numpy opencv-python

# For conda users
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

#### Image Segmentation (`sample-visual-segmenter.py`)

**What it does:** Separates different parts of an image (like separating people from background).

```bash
python sample-visual-segmenter.py
```

**Example output:**

```
person: 0.9876
background: 0.8765
clothing: 0.7654
```

**Best for:**

- Image editing applications
- Medical imaging
- Photo manipulation

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

#### Speech-to-Text (`Audio To Text.py`)

**What it does:** Transcribes spoken audio into written text.

```bash
python "Audio To Text.py"
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

#### Multimodal Processing (`Multimodal To Text.py`)

**What it does:** Processes both images and text together for comprehensive understanding.

```bash
python "Multimodal To Text.py"
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

## Learning Resources 📚

### Official Documentation

- **Clarifai Docs:** [docs.clarifai.com](https://docs.clarifai.com/)
- **Model Types Guide:** [Portal Guide - Models](https://docs.clarifai.com/portal-guide/model/model-types/)
- **Python SDK:** [Python Client Guide](https://docs.clarifai.com/sdk/python/)
- **API Reference:** [API Documentation](https://docs.clarifai.com/api-guide/)

### Tutorials and Examples

- **Getting Started:** [Quick Start Guide](https://docs.clarifai.com/getting-started/)
- **Model Predictions:** [Making Predictions](https://docs.clarifai.com/portal-guide/predict/)
- **Custom Models:** [Training Custom Models](https://docs.clarifai.com/portal-guide/model/pcustom-model-walkthrough/)

### Community and Support

- **Community Forum:** [community.clarifai.com](https://community.clarifai.com/)
- **GitHub Issues:** Report bugs and request features
- **Discord Server:** Join the Clarifai developer community

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

## Security Note 🔒

**Never commit API keys to version control!**

Always use environment variables or secure config files that are excluded from your repository. Add your API keys to `.gitignore`:

```gitignore
# API keys and secrets
.env
config.py
secrets.json
```

---

**Happy coding with Clarifai! 🎉**

If you build something cool with these examples, we'd love to see it! Share your projects with the community and help others learn AI development.

#!/bin/bash
"""
Clarifai Demo Environment Setup Script

This script automates the setup process for running Clarifai AI model demos.
It creates a conda environment, installs required packages, and guides you
through setting up your API credentials.

Usage:
  chmod +x setup-env.sh
  ./setup-env.sh

Author: Clarifai
Last Updated: 2025
"""

# Configuration
ENV_NAME="clarifai_312"
PYTHON_VERSION="3.12"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Clarifai Demo Environment Setup${NC}"
echo -e "${BLUE}===================================${NC}"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ Conda is not installed or not in PATH${NC}"
    echo -e "${YELLOW}Please install Anaconda or Miniconda first:${NC}"
    echo "   https://docs.anaconda.com/anaconda/install/"
    exit 1
fi

echo -e "${GREEN}✅ Conda found${NC}"

# Create the conda environment if it doesn't exist
echo -e "${BLUE}📦 Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION...${NC}"
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to create conda environment${NC}"
    exit 1
fi

# Activate the environment
echo -e "${BLUE}🔄 Activating environment: $ENV_NAME...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to activate conda environment${NC}"
    exit 1
fi

# Install required Python packages
echo -e "${BLUE}📚 Installing Python packages from requirements.txt...${NC}"
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to install Python packages${NC}"
    echo -e "${YELLOW}Try running manually: pip install -r requirements.txt${NC}"
    exit 1
fi

# Success message
echo -e "${GREEN}✅ Environment '$ENV_NAME' is ready!${NC}"
echo ""

# Guide user to set up API key
echo -e "${YELLOW}🔑 IMPORTANT: Set up your Clarifai API key${NC}"
echo -e "${YELLOW}============================================${NC}"
echo "1. Get your Personal Access Token (PAT) from:"
echo "   https://clarifai.com/settings/security"
echo ""
echo "2. Set your environment variable:"
echo -e "   ${BLUE}export CLARIFAI_PAT='your_actual_api_key_here'${NC}"
echo ""
echo "3. To make it permanent, add to your ~/.bashrc or ~/.zshrc:"
echo -e "   ${BLUE}echo 'export CLARIFAI_PAT=\"your_key_here\"' >> ~/.bashrc${NC}"
echo -e "   ${BLUE}source ~/.bashrc${NC}"
echo ""

# Activation instructions
echo -e "${GREEN}🎯 Next Steps:${NC}"
echo -e "${GREEN}==============${NC}"
echo "1. Activate the environment:"
echo -e "   ${BLUE}conda activate $ENV_NAME${NC}"
echo ""
echo "2. Set your API key (see instructions above)"
echo ""
echo "3. Test with a simple example:"
echo -e "   ${BLUE}python sample-text-to-text-stream.py${NC}"
echo ""
echo "4. Explore other examples in the directory"
echo ""

# Reactivate environment for current session
echo -e "${BLUE}🔄 Activating environment for current session...${NC}"
conda deactivate
conda activate "$ENV_NAME"

echo -e "${GREEN}✅ Setup complete! Environment '$ENV_NAME' is now active.${NC}"
echo -e "${YELLOW}Don't forget to set your CLARIFAI_PAT environment variable!${NC}"

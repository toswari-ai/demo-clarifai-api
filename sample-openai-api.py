# Import necessary libraries
import os  # For accessing environment variables
from openai import OpenAI  # OpenAI library for making API calls

# Security: Get API key from environment variable instead of hardcoding it
# This keeps your secret API key safe and out of your source code
api_key = os.getenv('CLARIFAI_PAT')  # Get the API key from environment variable
if not api_key:  # Check if the API key was found
    # If no API key is found, stop the program with an error message
    raise ValueError("Please set the CLARIFAI_PAT environment variable")

# Create an OpenAI client that will connect to Clarifai's API
# Clarifai provides an OpenAI-compatible endpoint for their models
client = OpenAI(
    base_url="https://api.clarifai.com/v2/ext/openai/v1",  # Clarifai's OpenAI-compatible endpoint
    api_key=api_key,  # Use our securely retrieved API key
)
# Make a request to the AI model to generate a response
response = client.chat.completions.create(
    # Specify which AI model to use (in this case, Google's Gemma model hosted on Clarifai)
    model="https://clarifai.com/gcp/generate/models/gemma-3-12b-it",
    
    # Create a conversation with the AI using a list of messages
    messages=[
        # System message: gives the AI instructions on how to behave
        {"role": "system", "content": "Talk like a pirate."},
        
        # User message: the actual question we want the AI to answer
        {
            "role": "user",
            "content": "How do I check if a Python object is an instance of a class?",
        },
    ],
    
    # AI generation parameters:
    temperature=0.7,  # Controls randomness (0.0 = very predictable, 1.0 = very creative)
    stream=False,     # Get complete response at once instead of streaming
)

# Pretty print the response in a user-friendly format
print("🤖 AI Response:")
print("=" * 60)

# Extract and display the AI's message content with nice formatting
ai_message = response.choices[0].message.content
print(ai_message)

print("=" * 60)
print("✅ Response complete!")
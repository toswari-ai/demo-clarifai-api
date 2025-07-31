# Clarifai Model Client Script
# Set the environment variables `CLARIFAI_PAT` to run this script.
# Example usage:
import os
from clarifai.client import Model

# Get PAT from environment variable for security
pat = os.getenv('CLARIFAI_PAT')
if not pat:
    raise ValueError("Please set the CLARIFAI_PAT environment variable")

model = Model(
  url="https://clarifai.com/gcp/generate/models/gemma-3-12b-it",
  pat=pat
)

# Example model prediction from different model methods: 

response = model.predict(
    prompt = "What is the future of AI?", 
    system_prompt = "You are a helpful assistant.", 
    max_tokens = 1024, 
    temperature = 0.7, 
    top_p = 0.9
)
print(response)

#response = model.generate(prompt="What is the future of AI?", images=[], audios=[], videos=[], chat_history=[], audio=None, video=None, image=None, tools=None, tool_choice=None, system_prompt="What is the future of AI?", max_tokens=1024, temperature=0.7, top_p=0.9)
#for res in response:
#    print(res)


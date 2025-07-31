from clarifai.client.model import Model

prompt = "A cozy cabin in the woods surrounded by colorful autumn leaves"


inference_params = dict(quality="standard", size= '1024x1024')

# Model Predict
model_prediction = Model("https://clarifai.com/openai/dall-e/models/dall-e-3").predict_by_bytes(prompt.encode(), input_type="text", inference_params=inference_params)

output_base64 = model_prediction.outputs[0].data.image.base64

with open('dall-e3-output.png', 'wb') as f:
    f.write(output_base64)
    print
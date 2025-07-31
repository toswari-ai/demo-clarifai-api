from clarifai.client.model import Model

image_url = "https://s3.amazonaws.com/samples.clarifai.com/featured-models/image-captioning-statue-of-liberty.jpeg"

model_url = ("https://clarifai.com/clarifai/main/models/general-english-image-caption-clip")
model_prediction = Model(url=model_url, pat="YOUR_PAT").predict_by_url(image_url)

print(model_prediction.outputs[0].data.text.raw)
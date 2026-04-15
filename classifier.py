from transformers import pipeline
from PIL import Image

classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

image = Image.open("test_art.jpg")

labels = ["watercolor painting","gouache","fresco", "oil painting","acrylic painting", "watercolor", "pencil sketch", "digital art", "charcoal drawing"]

result = classifier(image, candidate_labels=labels)

print(result)
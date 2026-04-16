from transformers import pipeline
from PIL import Image

classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

image = Image.open("test_art.jpg")

medium_labels = ["watercolor painting","gouache","fresco", "oil painting","acrylic painting", "watercolor", "pencil sketch", "digital art", "charcoal drawing"]
style_labels = ["cubism", "neo_mannerism", "impressionism", "expressionism"]
theme_labels =["religion", "still life", "landscape", "historical"]

medium_result = classifier(image, candidate_labels=medium_labels)
style_result = classifier(image, candidate_labels=style_labels)
theme_result = classifier(image, candidate_labels=theme_labels)



print("Medium:",medium_result[0]['label'], str(round(medium_result[0]['score'] * 100)) + "%")
print("Style:",style_result[0]['label'], str(round(style_result[0]['score'] * 100)) + "%")
print("Theme:",theme_result[0]['label'], str(round(theme_result[0]['score'] * 100)) + "%")
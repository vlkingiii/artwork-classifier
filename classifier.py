from transformers import pipeline
from PIL import Image
import sqlite3

conn = sqlite3.connect("artworks.db")
cursor = conn.cursor()

classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

title = input("Enter artwork title: ")
artist = input("Enter artist name: ")

image_path = input("Enter image path: ")
image = Image.open(image_path)

medium_labels = ["watercolor painting","gouache","fresco", "oil painting","acrylic painting", "watercolor", "pencil sketch", "digital art", "charcoal drawing"]
style_labels = ["cubism", "neo_mannerism", "impressionism", "expressionism"]
theme_labels =["religion", "still life", "landscape", "historical"]

medium_result = classifier(image, candidate_labels=medium_labels)
style_result = classifier(image, candidate_labels=style_labels)
theme_result = classifier(image, candidate_labels=theme_labels)



print("Medium:",medium_result[0]['label'], str(round(medium_result[0]['score'] * 100)) + "%")
print("Style:",style_result[0]['label'], str(round(style_result[0]['score'] * 100)) + "%")
print("Theme:",theme_result[0]['label'], str(round(theme_result[0]['score'] * 100)) + "%")

cursor.execute('''
               
    INSERT INTO artworks (title, artist, medium, style, theme)
    VALUES (?, ?, ?, ?, ?)
''', (title, artist, medium_result[0]['label'], style_result[0]['label'], theme_result[0]['label']))

conn.commit()
conn.close()
print("Saved to database.")
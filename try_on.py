from gradio_client import Client, handle_file
from PIL import Image
import os

def tryOnAPI(input_person, input_garment):
	print("api called")

	client = Client("Kwai-Kolors/Kolors-Virtual-Try-On")
	result = client.predict(
	person_img=handle_file(input_person),
	garment_img=handle_file(input_garment),
	seed=0,
	randomize_seed=True,
	api_name="/tryon"
	)

	image_path = result[0]

	# Open the image
	print(image_path)
	image = Image.open(image_path)

	# Save the image to a specific location
	output_image_path = "static/output_image.png"  # Save to the 'static' folder
	image.save(output_image_path)

	return output_image_path  # Return the path to the saved image

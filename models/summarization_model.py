# models/summarization_model.py
import os
import replicate

replicate_api_key = os.getenv("REPLICATE_API_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = replicate_api_key


def summarize_text(image_path):

    # Use Replicate API to generate a short description of the image
    input_image = {"image": open(image_path, "rb")}
    description = replicate.run(
        "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
        input=input_image
    )

    return description

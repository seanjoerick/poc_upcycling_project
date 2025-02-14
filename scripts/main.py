import torch
from PIL import Image
import ollama
from transformers import CLIPProcessor, CLIPModel

# ** load CLIP model and processor once to avoid redundant loading
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

def get_image_features(image_path):
    """
    Extracts image features from the provided image path using CLIP model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Normalized image features.
    """
    # ** load and process the image
    image = Image.open(image_path)
    image_inputs = processor(images=image, return_tensors="pt", padding=True)

    # ** get image features
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)

    # ** normalize features
    image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features

def get_best_match(image_features, labels):
    """
    Finds the best match from the given labels by comparing similarity with image features.

    Args:
        image_features (torch.Tensor): The normalized image features.
        labels (list): List of labels to compare.

    Returns:
        str: The label that best matches the image.
    """
    similarities = []
    for label in labels:
        text_inputs = processor(
            text=label,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)

        # ** normalize text features
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

        # ** compute similarity (dot product)
        similarity = (image_features @ text_features.T).squeeze(0)
        similarities.append(similarity.item())

    # ** return the label with the highest similarity
    best_match_idx = similarities.index(max(similarities))
    return labels[best_match_idx]

def analyze_image(image_path, categories, item_types):
    """
    Analyzes the image to identify material category and item type.

    Args:
        image_path (str): Path to the image file.
        categories (list): List of material categories.
        item_types (list): List of item types.

    Returns:
        tuple: Identified material category and item type.
    """
    image_features = get_image_features(image_path)

    # ** identify material category and item type
    best_match_category = get_best_match(image_features, categories)
    best_match_item = get_best_match(image_features, item_types)

    return best_match_category, best_match_item

def generate_suggestions(material, item):
    """
    Generates upcycling suggestions for the identified material and item type.

    Args:
        material (str): Identified material category.
        item (str): Identified item type.

    Returns:
        list: Upcycling suggestions with titles, instructions, and relevant links.
    """
    prompt = f"""
    Please provide at least three distinct upcycling suggestions for a {material} {item}. For each suggestion, include the following sections:

    **{material.capitalize()} {item.capitalize()} to [New Product Name]**

    **Materials Needed:**

    **Instructions:**

    **Reference Links:**

    Ensure that each suggestion is unique, with clear and concise instructions suitable for someone with basic crafting skills. Include at least one relevant link per suggestion, such as a YouTube tutorial, blog post, 
    or any other resource that provides further information or examples.
    """

    # ** generate suggestions using Llama 2 via Ollama
    response = ollama.chat(model='llama2', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def main(image_path):
    """
    Main function to analyze the image and generate upcycling suggestions.

    Args:
        image_path (str): Path to the image file.
    """
    # ** define fixed material categories and item types
    categories = ["denim", "polyester", "nylon", "rayon", "fur", "leather", "metal", "gems", "stones"]
    item_types = ["shirt", "pants", "blouse", "skirt", "dress", "shoes", "hat", "scarf", "gloves"]

    # ** analyze the image to determine the material category and item type
    material, item = analyze_image(image_path, categories, item_types)
    print(f"Identified Material: {material}")
    print(f"Identified Item: {item}")

    # ** generate upcycling suggestions
    suggestion = generate_suggestions(material, item)
    print(f"Upcycling Suggestion:\n{suggestion}")

if __name__ == "__main__":
    # ** provide the path to the user image
    user_image_path = "data/raw/shirts.jpg"
    main(user_image_path)

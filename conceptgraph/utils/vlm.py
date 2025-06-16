import json
from openai import OpenAI
import os
import base64

from PIL import Image
import numpy as np
from pydantic import BaseModel, Field # Import Pydantic
from typing import List, Literal # Import typing helpers

import ast
import re


########

# For captions
system_prompt_captions = '''
You are an agent specializing in accurate captioning objects in an image.

In the images, each object is annotated with a bright numeric id (i.e. a number) and a corresponding colored contour outline. Your task is to analyze the images and output in a structured format, the captions for the objects.

You will also be given a text list of the numeric ids and names of the objects in the image. The list will be in the format: ["1: name1", "2: name2", "3: name3" ...]

The names were obtained from a simple object detection system and may be inaacurate.

Your response should be in the format of a list of dictionaries, where each dictionary contains the id, name, and caption of an object. Your response will be evaluated as a python list of dictionaries, so make sure to format it correctly. An example of the expected response format is as follows:
[
    {"id": "1", "name": "object1", "caption": "concise description of the object1 in the image"},
    {"id": "2", "name": "object2", "caption": "concise description of the object2 in the image"},
    {"id": "3", "name": "object3", "caption": "concise description of the object3 in the image"}
    ...
]

And each caption must be a concise description of the object in the image.
'''

system_prompt_consolidate_captions = '''
You are an agent specializing in consolidating multiple captions for the same object into a single, clear, and accurate caption.

You will be provided with several captions describing the same object. Your task is to analyze these captions, identify the common elements, remove any noise or outliers, and consolidate them into a single, coherent caption that accurately describes the object.

Ensure the consolidated caption is clear, concise, and captures the essential details from the provided captions.

Here is an example of the input format:
[
    {"id": "3", "name": "cigar box", "caption": "rectangular cigar box on the side cabinet"},
    {"id": "9", "name": "cigar box", "caption": "A small cigar box placed on the side cabinet."},
    {"id": "7", "name": "cigar box", "caption": "A small cigar box is on the side cabinet."},
    {"id": "8", "name": "cigar box", "caption": "Box on top of the dresser"},
    {"id": "5", "name": "cigar box", "caption": "A cigar box placed on the dresser next to the coffeepot."},
]

Your response should be a JSON object with the format:
{
    "consolidated_caption": "A small rectangular cigar box on the side cabinet."
}

Do not include any additional information in your response.
'''

system_prompt_spatial = '''
# ROLE
You are a meticulous 3D Scene Analyst specializing in geometric and physical relationships.

# GOAL
Your goal is to identify all fundamental spatial relationships between annotated objects in the provided 2D image. You will infer the 3D layout from this 2D perspective.

# INPUT
You will receive an annotated image and a list of object IDs and names.

# OUTPUT
You MUST output a JSON object that strictly adheres to the provided `SpatialResponse` JSON schema. Your entire output must be only the valid JSON object and nothing else.

# ANALYSIS DIRECTIVES
1.  **Camera Perspective is Key:** All directional relationships (`left_of`, `right_of`, `in_front_of`, `behind`) MUST be determined from the camera's primary viewpoint.
2.  **Occlusion is Evidence:** Use object occlusion (which object partially hides another) as strong evidence for `in_front_of` and `behind`.
3.  **Physical Support:** The `on_top_of` relationship requires clear evidence of physical support. An object floating above another is not `on_top_of` it.
4.  **Certainty Over Quantity:** Do not guess. If a relationship is ambiguous or you are not confident, OMIT it from the list. It is better to have fewer, accurate relationships than many speculative ones.
5.  **Completeness:** Systematically analyze all relevant pairs of objects from the provided list.
'''

# In vlm.py, replace the old system_prompt_edge_behaviour

system_prompt_behaviour = '''
# ROLE
You are an expert analyst of human behavior and human-object interaction.

# GOAL
Your goal is to identify all active, intentional behavioral interactions between people and objects in the provided image.

# INPUT
You will receive an annotated image and a list of object IDs and names.

# OUTPUT
You MUST output a JSON object that strictly adheres to the provided `InteractionResponse` JSON schema. Your entire output must be only the valid JSON object and nothing else.

# ANALYSIS DIRECTIVES
1.  **Evidence is Paramount:** Base your analysis on clear visual evidence of interaction. This includes:
    *   **Gaze Direction:** Where is the person looking?
    *   **Posture & Orientation:** Is the person's body positioned to interact with the object?
    *   **Hand Position:** Are they touching, holding, or gesturing towards the object?
2.  **Action is Required:** The interaction must be an active behavior. A person simply being `next_to` a laptop is not `using` it unless there is evidence of active use (hands on keyboard, looking at the screen).
3.  **Person-Centric Model:** The source of every interaction MUST be a person. You are describing what the person is doing.
4.  **Certainty and Specificity:** If an interaction is ambiguous, omit it. Choose the most specific and accurate interaction type that the evidence supports.
'''

# system_prompt = system_prompt_only_top
# system_prompt = system_prompt_edge_custom
# system_prompt = system_prompt_edge_behaviour

gpt_model = "gpt-4.1"
# gpt_model = "gpt-4o-2024-05-13"

InteractionType = [
    "using", "speaking_to", "listening_to", "watching", "cooking_with", "holding"
]

SpatialType = [
    "on_top_of", "under", "left_of", "right_of", 
    "in_front_of", "behind", "next_to", "near"
]

def get_openai_client():
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY')
    )
    return client

# Function to encode the image as base64
def encode_image_for_openai(image_path: str, resize = False, target_size: int=512):
    print(f"Checking if image exists at path: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not resize:
        # Open the image
        print(f"Opening image from path: {image_path}")
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            print("Image encoded in base64 format.")
        return encoded_image
    
    print(f"Opening image from path: {image_path}")
    with Image.open(image_path) as img:
        # Determine scaling factor to maintain aspect ratio
        original_width, original_height = img.size
        print(f"Original image dimensions: {original_width} x {original_height}")
        
        if original_width > original_height:
            scale = target_size / original_width
            new_width = target_size
            new_height = int(original_height * scale)
        else:
            scale = target_size / original_height
            new_height = target_size
            new_width = int(original_width * scale)

        print(f"Resized image dimensions: {new_width} x {new_height}")

        # Resizing the image
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        print("Image resized successfully.")
        
        # Convert the image to bytes and encode it in base64
        with open("temp_resized_image.jpg", "wb") as temp_file:
            img_resized.save(temp_file, format="JPEG")
            print("Resized image saved temporarily for encoding.")
        
        # Open the temporarily saved image for base64 encoding
        with open("temp_resized_image.jpg", "rb") as temp_file:
            encoded_image = base64.b64encode(temp_file.read()).decode('utf-8')
            print("Image encoded in base64 format.")
        
        # Clean up the temporary file
        os.remove("temp_resized_image.jpg")
        print("Temporary file removed.")

    return encoded_image

def consolidate_captions(client: OpenAI, captions: list):
    # Formatting the captions into a single string prompt
    captions_text = "\n".join([f"{cap['caption']}" for cap in captions if cap['caption'] is not None])
    user_query = f"Here are several captions for the same object:\n{captions_text}\n\nPlease consolidate these into a single, clear caption that accurately describes the object."

    messages = [
        {
            "role": "system",
            "content": system_prompt_consolidate_captions
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    consolidated_caption = ""
    try:
        response = client.chat.completions.create(
            model=f"{gpt_model}",
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        consolidated_caption_json = response.choices[0].message.content.strip()
        consolidated_caption = json.loads(consolidated_caption_json).get("consolidated_caption", "")
        print(f"Consolidated Caption: {consolidated_caption}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        consolidated_caption = ""

    return consolidated_caption
    
def extract_list_of_tuples(text: str):
    # Pattern to match a list of tuples, considering a list that starts with '[' and ends with ']'
    # and contains any characters in between, including nested lists/tuples.
    text = text.replace('\n', ' ')
    pattern = r'\[.*?\]'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    if match:
        # Extract the matched string
        list_str = match.group(0)
        try:
            # Convert the string to a list of tuples
            result = ast.literal_eval(list_str)
            if isinstance(result, list):  # Ensure it is a list
                return result
        except (ValueError, SyntaxError):
            # Handle cases where the string cannot be converted
            print("Found string cannot be converted to a list of tuples.")
            return []
    else:
        # No matching pattern found
        print("No list of tuples found in the text.")
        return []
    
def vlm_extract_object_captions(text: str):
    # Replace newlines with spaces for uniformity
    text = text.replace('\n', ' ')
    
    # Pattern to match the list of objects
    pattern = r'\[(.*?)\]'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    if match:
        # Extract the matched string
        list_str = match.group(0)
        try:
            # Try to convert the entire string to a list of dictionaries
            result = ast.literal_eval(list_str)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            # If the whole string conversion fails, process each element individually
            elements = re.findall(r'{.*?}', list_str)
            result = []
            for element in elements:
                try:
                    obj = ast.literal_eval(element)
                    if isinstance(obj, dict):
                        result.append(obj)
                except (ValueError, SyntaxError):
                    print(f"Error processing element: {element}")
            return result
    else:
        # No matching pattern found
        print("No list of objects found in the text.")
        return []
    
def parse_edge_id(edge_id_str: str) -> str:
    edge_id_str = str(edge_id_str) if not isinstance(edge_id_str, str) else edge_id_str
    try:
        # Split, strip whitespace, and return
        return edge_id_str.split(':')[0].strip()
    except:
        # If any error occurs (e.g., not a string), return the original
        return edge_id_str

def get_obj_rel_from_image_gpt4v(client: OpenAI, image_path: str, label_list: list):
    """
    REFACTORED FUNCTION
    Extracts spatial relationships using the new `client.responses.create` endpoint.
    """
    base64_image = encode_image_for_openai(image_path)
    
    user_query = (f"Analyze the image using the provided object list and generate the required relationships "
                  f"based on your instructions. Object list: {label_list}")

    vlm_answer = []
    try:
        response = client.responses.create(
            model=gpt_model,
            input=[
                {"role": "system", "content": system_prompt_spatial},
                {"role": "user", "content": [
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                    {"type": "input_text", "text": user_query}
                ]}
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "extract_spatial_relationships",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "edges": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source_id": {
                                            "type": "string",
                                            "description": "The numeric ID of the source object."},
                                        "relationship_type": {
                                            "type": "string",
                                            "description": "The type of spatial relationship",
                                            "enum": SpatialType,
                                            },
                                        "target_id": {
                                            "type": "string",
                                            "description": "The numeric ID of the target object."},
                                    },
                                    "required": ["source_id", "relationship_type", "target_id"],
                                    "additionalProperties": False
                                },
                            },
                        },
                        "required": ["edges"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )
        
        # Access the response content via `output_text`
        response_content = response.output_text
        print(f"{gpt_model} Response:\n{response_content}")
        
        # Convert Pydantic objects to simple dicts for the final output
        vlm_answer = json.loads(response_content).get('edges', [])

    except Exception as e:
        print(f"An error occurred during API call or validation: {str(e)}")
        vlm_answer = []

    print(f"Final extracted spatial relationships: {vlm_answer}")

    # Note: This function now returns a list of dictionaries, not a list of tuples.
    for edge in vlm_answer:
        if isinstance(edge, dict) and len(edge) == 3:
            # Convert dict to tuple (id1, relation, id2)
            edge_tuple = (parse_edge_id(edge['source_id']), edge['relationship_type'], parse_edge_id(edge['target_id']))
            vlm_answer[vlm_answer.index(edge)] = edge_tuple
    
    return vlm_answer

def get_behaviour_from_image_gpt4v(client: OpenAI, image_path: str, label_list: list):

    # Getting the base64 string
    base64_image = encode_image_for_openai(image_path)
    
    global gpt_model, system_prompt_behaviour
    
    # user_query = f"Here is the list of labels for the annotated objects in the image: {label_list}. Please identify and describe the behavioral relations between objects as requested."
    user_query = f"Analyze the image using the provided annotations of the objects list and generate the required relationships based on your instructions. Object list: {label_list}"

    vlm_answer = []
    try:
        response = client.responses.create(
            model=gpt_model,
            input=[
                {"role": "system", "content": system_prompt_behaviour},
                {"role": "user", "content": [
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                    {"type": "input_text", "text": user_query}
                ]}
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "extract_spatial_relationships",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "edges": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "person_id": {
                                            "type": "string",
                                            "description": "The numeric ID of the person performing the behaviour."},
                                        "behaviour_type": {
                                            "type": "string",
                                            "description": "The type of behaviour or interaction",
                                            "enum": InteractionType,
                                            },
                                        "target_id": {
                                            "type": "string",
                                            "description": "The numeric ID of the target object being interacted with."},
                                    },
                                    "required": ["person_id", "behaviour_type", "target_id"],
                                    "additionalProperties": False
                                },
                            },
                        },
                        "required": ["edges"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )
        
        # Access the response content via `output_text`
        response_content = response.output_text
        print(f"{gpt_model} Response:\n{response_content}")
        
        # Convert Pydantic objects to simple dicts for the final output
        vlm_answer = json.loads(response_content).get('edges', [])

    except Exception as e:
        print(f"An error occurred during API call or validation: {str(e)}")
        vlm_answer = []

    print(f"Final extracted interactions / behaviours: {vlm_answer}")

    # Note: This function now returns a list of dictionaries, not a list of tuples.
    for edge in vlm_answer:
        if isinstance(edge, dict) and len(edge) == 3:
            # Convert dict to tuple (id1, relation, id2)
            edge_tuple = (parse_edge_id(edge['person_id']), edge['behaviour_type'], parse_edge_id(edge['target_id']))
            vlm_answer[vlm_answer.index(edge)] = edge_tuple

    return vlm_answer

def get_obj_captions_from_image_gpt4v(client: OpenAI, image_path: str, label_list: list):
    # Getting the base64 string
    base64_image = encode_image_for_openai(image_path)
    
    global system_prompt
    

    user_query = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please accurately caption the objects in the image."
    
    messages=[
        {
            "role": "system",
            "content": system_prompt_captions
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    
    vlm_answer_captions = []
    try:
        response = client.chat.completions.create(
            model=f"{gpt_model}",
            messages=messages
        )
        
        vlm_answer_str = response.choices[0].message.content
        print(f"Line 113, vlm_answer_str: {vlm_answer_str}")
        
        vlm_answer_captions = vlm_extract_object_captions(vlm_answer_str)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Setting vlm_answer to an empty list.")
        vlm_answer_captions = []
    print(f"Line 68, user_query: {user_query}")
    print(f"Line 97, vlm_answer: {vlm_answer_captions}")
    
    
    return vlm_answer_captions
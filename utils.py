import os
import re
import pandas as pd
from rapidfuzz import process, fuzz

FOLDER_PATH = 'C:\\Users\\Admin\\Documents\\Coding DOCS\\Projects\\Workspace\\SkinCare Match Finder'

# datasets
prods_data = pd.read_csv(os.path.join(FOLDER_PATH, "data", "products_data.csv"))
ingreds_data = pd.read_csv(os.path.join(FOLDER_PATH, "data", "ingredients_data.csv"))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^a-z0-9\s]', '', text.lower().strip())

# Match product name (for image with name/tagline)
def match_product_name(output_text, knowledge_base):
    cleaned_output = clean_text(output_text)
    _, score, idx = process.extractOne(cleaned_output, knowledge_base, scorer=fuzz.token_sort_ratio)
    return prods_data.iloc[idx], score

def match_ingredients(extracted_ingreds, knowledge_base):
    matched_rows = []

    for ing in extracted_ingreds:
        clean_ing = clean_text(ing)
        _, score, idx = process.extractOne(clean_ing, knowledge_base, scorer=fuzz.token_sort_ratio)
        if score > 50:
            matched_rows.append((idx, score))

    return sorted(matched_rows, key=lambda x: x[1], reverse=True)

def profile_similarity(skin_profile, ingredient_profile_str):

    # ingredient_profile_str is already a comma-separated string (or a list in string form)
    return fuzz.token_set_ratio(skin_profile, ingredient_profile_str) / 100  # normalize to 0–1

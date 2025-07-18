import os
import streamlit as st
import pandas as pd
from utils import clean_text, match_product_name, match_ingredients, profile_similarity, preprocess_image
import easyocr
from PIL import Image
import numpy as np
from logger import logging
import tensorflow
from tensorflow import keras
from keras.models import load_model

from dotenv import load_dotenv
load_dotenv()
FOLDER_PATH = os.getenv("FOLDER_PATH")

if not FOLDER_PATH:
    raise ValueError("FOLDER_PATH not set in .env file!")


# ====== CACHED RESOURCES ======

@st.cache_resource
def load_easyocr_reader():
    return easyocr.Reader(['en'])

@st.cache_data
def load_datasets():
    prods = pd.read_csv(os.path.join(FOLDER_PATH, "data", "products_data.csv"))
    ingreds = pd.read_csv(os.path.join(FOLDER_PATH, "data", "ingredients_data.csv"))
    return prods, ingreds

@st.cache_data
def prepare_cleaned_choices(prods_df, ingreds_df):
    prods_choices = [clean_text(p) for p in prods_df['prod_descrp'].tolist()]
    ingreds_choices = [clean_text(i) for i in ingreds_df['ingred_name'].tolist()]
    return prods_choices, ingreds_choices

@st.cache_resource
def load_models():
    concern_model = load_model(os.path.join(FOLDER_PATH, 'models', 'skin_concern_classifier_PT.h5'))
    type_model = load_model(os.path.join(FOLDER_PATH, 'models', 'skin_type_classifier_PT.h5'))
    return concern_model, type_model

logging.info('Caching Completed')

# ====== LOAD EVERYTHING ======

skin_concern_classes = {0: 'acne', 1: 'blackheads', 2: 'redness'}
skin_type_classes = {0: 'oily', 1: 'dry', 2: 'normal', 3: 'sensitive', 4: 'combination'}

reader = load_easyocr_reader()
prods_data, ingreds_data = load_datasets()
prods_choices, ingreds_choices = prepare_cleaned_choices(prods_data, ingreds_data)
skin_concern_model, skin_type_model = load_models()

logging.info('Loading Completed')

def predict_single(img_file, concern_model, type_model):
    concern_input = preprocess_image(img_file, (128, 128))
    type_input = preprocess_image(img_file, (150, 150))

    concern_probs = concern_model.predict(concern_input)
    type_probs = type_model.predict(type_input)

    concern = skin_concern_classes[np.argmax(concern_probs)]
    type_ = skin_type_classes[np.argmax(type_probs)]

    return concern, type_

def get_skin_profile(front, left, right):
    concern_list, type_list = [], []
    for img in [front, left, right]:
        c, t = predict_single(img, skin_concern_model, skin_type_model)
        concern_list.append(c)
        type_list.append(t)
    # Mode of predictions
    final_concern = max(set(concern_list), key=concern_list.count)
    final_type = max(set(type_list), key=type_list.count)
    return final_type, final_concern

# ===== UI =====

st.title("💧 SkinCare Product Suitability Checker")

with st.form(key="input_form"):
    image_file = st.file_uploader("📷 Upload Image of Product or Ingredients", type=["jpg", "jpeg", "png"])
    description_type = st.radio("🧾 What does the image contain?", options=["Product", "Ingredients"])

    st.subheader("📸 Upload Your 3 Face Images")
    front_img = st.file_uploader("Front Face", type=["jpg", "jpeg", "png"], key="front")
    left_img = st.file_uploader("Left Face", type=["jpg", "jpeg", "png"], key="left")
    right_img = st.file_uploader("Right Face", type=["jpg", "jpeg", "png"], key="right")

    submit_button = st.form_submit_button(label="Submit")

    if not (front_img and left_img and right_img):
        st.warning("⚠️ Please upload all 3 face images to generate your skin profile.")
        st.stop()

    st.info("🔬 Predicting your skin profile from uploaded images...")
    predicted_skin_type, predicted_concern = get_skin_profile(front_img, left_img, right_img)

    st.success(f"✅ Predicted Skin Type: **{predicted_skin_type}**")
    st.success(f"✅ Predicted Skin Concern: **{predicted_concern}**")

    skin_profile = {
        "skin_types": [predicted_skin_type],
        "concerns": predicted_concern
    }

# ===== PROCESS IMAGE =====

if submit_button:
    if image_file is not None:
        with st.spinner("🔍 Processing the image and analyzing suitability..."):
            image = Image.open(image_file).convert("RGB")
            image_np = np.array(image)
            results = reader.readtext(image_np)
            output = [detection[1] for detection in results]

            if not output:
                st.warning("⚠️ No text detected in the image. Please try a clearer image.")
                st.stop()

            logging.info(len(output))
            k = 30

            if description_type == 'Product':
                output_text = clean_text(' '.join(output).lower())
                product_match, score = match_product_name(output_text, prods_choices)
                output_ingreds = product_match['list_of_ingreds'].split(', ')
                if len(output_ingreds) > k:
                    output_ingreds = output_ingreds[:k]
                logging.info(len(output_ingreds))
            else:
                output_ingreds = output[:k] if len(output) > k else output
                logging.info(len(output_ingreds))

            logging.info('Matching ingredients...')
            matched_data = match_ingredients(output_ingreds, ingreds_choices)
            match_df = ingreds_data.iloc[[item[0] for item in matched_data]].copy()
            match_df["match_score"] = [item[1] for item in matched_data]
            
            logging.info('Calculating suitability...')
            user_skin_profile = ' '.join(skin_profile["skin_types"]) + ' ' + skin_profile.get("concerns", "")
            match_df["profile_similarity"] = match_df["skin_profile"].apply(
                lambda prof: profile_similarity(user_skin_profile, prof)
            )

            # Drop ingredients with similarity < 0.5
            min_rating = match_df["ratingscore"].min()
            max_rating = match_df["ratingscore"].max()
            match_df["normalized_rating"] = (match_df["ratingscore"] - min_rating) / (max_rating - min_rating)

            match_df["weighted_rating"] = match_df["normalized_rating"] * match_df["profile_similarity"]

            if match_df["profile_similarity"].sum() > 0:
                final_score = match_df["weighted_rating"].sum() / match_df["profile_similarity"].sum()
            else:
                final_score = 0.0

        # ===== OUTPUTS =====

        with st.expander("🔤 Text Detected from Image"):
            st.write(output)

        st.subheader("🧾 Final Suitability Score")
        st.metric("🔍 Weighted Suitability Score", round(final_score, 3))

        if final_score >= 0.75:
            st.success("✅ This product seems **highly suitable** for your skin profile!")
        elif final_score >= 0.5:
            st.info("⚠️ This product is **moderately suitable**. Use with caution.")
        else:
            st.error("❌ This product may **not be suitable** for your skin. Please consult a dermatologist.")

        st.subheader("🧪 Top Matched Ingredients")
        display_df = match_df[["ingred_name", "normalized_rating", "profile_similarity", "match_score", "weighted_rating"]]
        display_df.columns = ["Ingredient", "Rating", "Profile Similarity", "Match Score", "Weighted Score"]
        st.dataframe(display_df.sort_values("Weighted Score", ascending=False).reset_index(drop=True))

    else:
        st.warning("⚠️ Please upload an image before submitting.")

import os
import streamlit as st
import pandas as pd
from utils import clean_text, match_product_name, match_ingredients, profile_similarity
import easyocr
from PIL import Image
import numpy as np
from logger import logging

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

logging.info('Caching Completed')

# ====== LOAD EVERYTHING ======

reader = load_easyocr_reader()
prods_data, ingreds_data = load_datasets()
prods_choices, ingreds_choices = prepare_cleaned_choices(prods_data, ingreds_data)

logging.info('Loading Completed')

# ===== UI =====

st.title("💧 SkinCare Product Suitability Checker")

with st.form(key="input_form"):
    image_file = st.file_uploader("📷 Upload Image of Product or Ingredients", type=["jpg", "jpeg", "png"])
    description_type = st.radio("🧾 What does the image contain?", options=["Product", "Ingredients"])

    st.subheader("🧬 Your Skin Profile")
    skin_types = ["Normal", "Dry", "Combination", "Oily", "Sensitive"]
    selected_skin_types = st.multiselect("Select your skin type(s):", options=skin_types)
    user_concerns = st.text_input("Concerns (e.g., Acne, Redness, Pores):")

    submit_button = st.form_submit_button(label="Submit")

skin_profile = {
    "skin_types": selected_skin_types,
    "concerns": user_concerns
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

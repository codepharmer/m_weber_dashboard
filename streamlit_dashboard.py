import streamlit as st
import pandas as pd
import json
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go # For more control if needed
from io import StringIO
import logging
from typing import List, Dict, Tuple, Any, Optional, Set
import datetime
import numpy as np # For handling potential numeric issues

# Setup basic logging (optional for Streamlit, but good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants and Configuration ---

# Define thematic keywords (lowercase) - adapt based on product type
# Example themes for a general product
# Note: Removed the direct reference to FINANCIAL_KEYWORDS here, copied relevant ones
THEME_KEYWORDS = {
    "Quality/Durability": [
        "broke", "broken", "damaged", "cheap", "flimsy", "quality", "durable", "sturdy",
        "lasted", "fell apart", "material", "well-made", 
        # Added for wreath context:
        "weatherproof", "weather-resistant", "fade", "fade-resistant", "shed", "fraying", 
        "needle drop", "wilt", "rot", "loose", "craftsmanship", "long-lasting"
    ],
    "Shipping/Packaging": [
        "shipping", "package", "packaging", "box", "arrived", "delivery", "late", "crushed", 
        "damaged box", 
        # Added for wreath context:
        "bent", "crammed", "squished", "bubble wrap", "stuffing", "wrinkled", "folded", "protection"
    ],
    "Value/Price": [
        "price", "cost", "value", "money", "expensive", "cheap", "affordable", "overpriced", 
        "deal", "worth", "budget", "charge", "rate", "fee", "discount",
        # Added for wreath context:
        "bargain", "steal", "pricier", "coupon", "marked up", "worth the money"
    ],
    "Functionality/Performance": [
        "work", "works", "performance", "functional", "defective", "issue", "problem", "difficult", 
        "easy", "instructions", "use", "feature",
        # Added for wreath context:
        "hang", "hanging", "hook", "loop", "stays up", "battery", "batteries", "timer", 
        "lit", "lights", "LED", "bulbs", "turn on", "switch", "twinkle", "tangle", "fluff", 
        "reshape", "arrange", "setup", "operate", "operation"
    ],
    "Appearance/Size": [
        "look", "looks", "appearance", "color", "design", "style", "size", "small", "large", 
        "smaller", "bigger", "fit", "cute", "beautiful",
        # Added for wreath context:
        "festive", "christmassy", "holiday", "ornaments", "bow", "pine cones", "berries", "holly", 
        "sparkle", "ribbon", "mistletoe", "wintery", "full", "sparse", "fluffy", "realistic", 
        "fake", "artificial", "greenery", "frosted", "flocked"
    ],
    "Customer Service": [
        "service", "customer", "support", "seller", "contact", "return", "refund", "response",
        # Added for wreath context:
        "replacement", "exchange", "compensation", "return label", "helpful", "communication", 
        "warranty", "escalate"
    ],
}

# Sentiment Thresholds
LOW_SENTIMENT_THRESHOLD = 4 # AI Sentiment <= this is considered low
HIGH_SENTIMENT_THRESHOLD = 7 # AI Sentiment > this is considered high
LOW_USER_RATING_THRESHOLD = 2 # User Rating <= this is considered low

# TOS Violation Severity Mapping (Customize based on importance)
TOS_SEVERITY_MAP = {
    "high": ["incentivized_or_promotional_content", "offensive_or_abusive_language", "personal_information_or_privacy_violation"],
    "medium": ["irrelevant_or_spammy_content"],
    "low": [] # Add any flags considered low severity
}
# Create a reverse map for easier lookup: flag -> severity
FLAG_TO_SEVERITY = {flag: severity for severity, flags in TOS_SEVERITY_MAP.items() for flag in flags}

# --- Helper Functions ---

def _parse_json_string(json_string: Any) -> Optional[Dict]:
    """Safely parses a JSON string, returning None on failure."""
    if not isinstance(json_string, str):
        # Attempt to handle if it's already a dict (e.g., from direct JSON load)
        if isinstance(json_string, dict):
             return json_string
        # Handle potential bytes
        if isinstance(json_string, bytes):
            try:
                json_string = json_string.decode('utf-8')
            except UnicodeDecodeError:
                 logging.warning(f"Could not decode bytes to UTF-8 for JSON parsing.")
                 return None
        else:
             # If not string, dict, or bytes, return None
             # logging.debug(f"Input to _parse_json_string is not a string or dict: {type(json_string)}")
             return None

    # If it was bytes and got decoded, or was originally a string
    if isinstance(json_string, str):
        try:
            # Basic cleaning for common issues like escaped quotes within the string
            # This is a simple fix and might need refinement based on actual data issues
            cleaned_string = json_string.replace('\\"', '"').replace("\\'", "'")
            # Sometimes the outer quotes might be escaped too if nested JSON strings
            if cleaned_string.startswith('"') and cleaned_string.endswith('"'):
                 cleaned_string = cleaned_string[1:-1].replace('\\"', '"')

            return json.loads(cleaned_string)
        except json.JSONDecodeError as e:
            # logging.warning(f"JSON Decode Error: {e} for string: {json_string[:100]}...") # Log snippet
            return None
        except Exception as e:
            # logging.error(f"Unexpected error parsing JSON: {e}")
            return None
    return None # Should not be reached if logic is correct, but as a safeguard


# Function to map keywords to themes
def map_keywords_to_themes(keywords: List[str], theme_map: Dict[str, List[str]]) -> Set[str]:
    """Maps a list of keywords to predefined themes."""
    themes = set()
    if not isinstance(keywords, list):
        return themes
    for keyword in keywords:
        if not isinstance(keyword, str):
            continue
        kw_lower = keyword.lower()
        for theme, theme_kws in theme_map.items():
            if kw_lower in theme_kws:
                themes.add(theme)
    return themes if themes else {"Other"} # Assign to 'Other' if no theme matched

def get_tos_flags_with_severity(x: Any) -> Tuple[Dict[str, bool], Dict[str, str]]:
    """Parses TOS JSON, returns flags found (True) and their assigned severity."""
    data = _parse_json_string(x)
    flags_found = {}
    flag_severities = {}
    if isinstance(data, dict):
        flags_data = data.get('flags', {})
        if isinstance(flags_data, dict):
            for flag, value in flags_data.items():
                # Consider only flags explicitly set to True
                if value is True and flag != 'no_tos_violations_detected':
                    flags_found[flag] = True
                    # Assign severity based on mapping, default to 'unknown' if not mapped
                    flag_severities[flag] = FLAG_TO_SEVERITY.get(flag, 'unknown')
    return flags_found, flag_severities


# --- Data Loading and Preprocessing ---

# Use st.cache_data for data loading and heavy processing
@st.cache_data
def load_and_preprocess_data(uploaded_file=None, default_filename="openai_cache_export.csv"):
    """Loads data, preprocesses JSON, parses dates, extracts themes, and handles new columns."""
    df = None
    file_source = None

    # Use sample data if no file is uploaded and default doesn't exist
    use_sample_data = False
    if uploaded_file is not None:
        try:
            # Handle potential different encodings
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0) # Reset file pointer
                df = pd.read_csv(uploaded_file, encoding='latin1') # Try another common encoding
            file_source = uploaded_file.name
            st.info(f"Loaded data from uploaded file: {file_source}")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return pd.DataFrame(), None
    else:
        try:
            df = pd.read_csv(default_filename)
            file_source = default_filename
            st.info(f"Loaded data from default file: {file_source}")
        except FileNotFoundError:
            st.warning(f"'{default_filename}' not found.")
            # Load Sample Data as fallback
            try:
                sample_data_io = StringIO(SAMPLE_DATA_CSV) # Use the sample data string defined below
                df = pd.read_csv(sample_data_io)
                file_source = "Internal Sample Data"
                use_sample_data = True
                st.info("Loaded internal sample data as no file was provided or found.")
            except Exception as e:
                st.error(f"Error reading internal sample data: {e}")
                return pd.DataFrame(), None
        except Exception as e:
            st.error(f"Error reading default file '{default_filename}': {e}")
            return pd.DataFrame(), None

    if df.empty:
        return pd.DataFrame(), None

    # --- Data Cleaning and Column Validation ---
    # Define columns needed for core functionality + new features
    base_required_cols = ['id', 'detect_tos_violation', 'analyze_rating_consistency', 'extract_keywords', 'analyze_review_sentiment']
    date_col = 'review_date'
    user_rating_col = 'review_score'
    verified_col = 'verified'
    helpful_col = 'helpful_counts'
    content_col = 'review_content' # Needed for displaying helpful reviews

    all_expected_cols = base_required_cols + [date_col, user_rating_col, verified_col, helpful_col, content_col]
    missing_cols = [col for col in all_expected_cols if col not in df.columns]

    # Handle missing date column (allow dashboard to run without trends)
    if date_col in missing_cols:
        st.warning(f"Column '{date_col}' not found. Trend analysis will be disabled.")
        df[date_col] = pd.NaT # Add a dummy NaT column
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Handle missing user rating column (affects consistency comparison)
    if user_rating_col in missing_cols:
        st.warning(f"Column '{user_rating_col}' not found. Detailed low-rating/high-sentiment analysis will be limited.")
        df[user_rating_col] = pd.NA # Add dummy NA column
    else:
        # Ensure numeric, coerce errors
        df[user_rating_col] = pd.to_numeric(df[user_rating_col], errors='coerce')


    # Handle missing verified column (affects verified analysis)
    if verified_col in missing_cols:
        st.warning(f"Column '{verified_col}' not found. Verified review analysis will be disabled.")
        df[verified_col] = pd.NA
    else:
        # Convert to boolean, handle potential string representations like 'True'/'False'
        df[verified_col] = df[verified_col].apply(lambda x: str(x).lower() == 'true' if pd.notna(x) else None).astype('boolean')


    # Handle missing helpful counts column (affects helpfulness analysis)
    if helpful_col in missing_cols:
        st.warning(f"Column '{helpful_col}' not found. Helpful review analysis will be disabled.")
        df[helpful_col] = pd.NA
    else:
        df[helpful_col] = pd.to_numeric(df[helpful_col], errors='coerce').fillna(0).astype(int) # Coerce, fill NaN with 0, convert to int

    # Handle missing review content column
    if content_col in missing_cols:
         st.warning(f"Column '{content_col}' not found. Displaying review text for helpful reviews will be disabled.")
         df[content_col] = "" # Add empty string column


    # Check for essential base columns for AI analysis
    missing_base_cols = [col for col in base_required_cols if col not in df.columns]
    if missing_base_cols:
        st.error(f"Missing essential AI analysis columns: {', '.join(missing_base_cols)}. Dashboard cannot proceed.")
        return pd.DataFrame(), file_source

    # --- JSON Parsing and Feature Extraction ---
    def get_sentiment_rating(x):
        data = _parse_json_string(x)
        return data.get('rating') if isinstance(data, dict) else None

    def get_keywords(x):
        data = _parse_json_string(x)
        # Ensure keywords exist and is a list before returning
        keywords = data.get('keywords') if isinstance(data, dict) else None
        return keywords if isinstance(keywords, list) else []


    def get_consistency_flag(x):
        data = _parse_json_string(x)
        if isinstance(data, dict):
            eval_data = data.get('consistency_evaluation', {})
            # Return the boolean flag, default to None if not found
            return eval_data.get('overall_consistency') if isinstance(eval_data, dict) else None
        return None

    # Apply parsing
    df['ai_sentiment_rating'] = df['analyze_review_sentiment'].apply(get_sentiment_rating)
    df['extracted_keywords_list'] = df['extract_keywords'].apply(get_keywords)
    df['is_consistent'] = df['analyze_rating_consistency'].apply(get_consistency_flag)

    # Parse TOS flags and severity
    tos_results = df['detect_tos_violation'].apply(get_tos_flags_with_severity)
    df['tos_flags_dict'] = tos_results.apply(lambda x: x[0]) # Dict of flags found
    df['tos_severity_dict'] = tos_results.apply(lambda x: x[1]) # Dict of flag -> severity
    df['has_tos_violation'] = df['tos_flags_dict'].apply(lambda x: bool(x))
    # Determine the highest severity level for each review
    df['highest_tos_severity'] = df['tos_severity_dict'].apply(
        lambda severities: max(severities.values(), key=lambda s: ["low", "medium", "high", "unknown"].index(s)) if severities else None
    )


    # Drop rows where essential parsing failed (e.g., sentiment rating)
    initial_rows = len(df)
    df.dropna(subset=['ai_sentiment_rating'], inplace=True) # Drop rows where AI sentiment couldn't be parsed
    rows_after_dropna = len(df)
    if initial_rows > rows_after_dropna:
        st.info(f"Dropped {initial_rows - rows_after_dropna} rows due to missing/invalid AI sentiment rating JSON.")

    # Convert sentiment rating to integer after handling potential NaNs/parsing errors
    # Ensure it's numeric first, then int
    df['ai_sentiment_rating'] = pd.to_numeric(df['ai_sentiment_rating'], errors='coerce')
    df.dropna(subset=['ai_sentiment_rating'], inplace=True) # Drop if conversion failed
    df['ai_sentiment_rating'] = df['ai_sentiment_rating'].astype(int)


    # --- Feature Engineering ---
    # Map keywords to themes
    df['review_themes'] = df['extracted_keywords_list'].apply(lambda kws: map_keywords_to_themes(kws, THEME_KEYWORDS))

    # Flag for low user rating & high AI sentiment (potential mismatch focus)
    if user_rating_col not in missing_cols: # Only calculate if user rating exists
        df['low_user_high_ai'] = (
            (df[user_rating_col] <= LOW_USER_RATING_THRESHOLD) &
            (df['ai_sentiment_rating'] >= HIGH_SENTIMENT_THRESHOLD) # Use >= for high AI sentiment
        )
    else:
        df['low_user_high_ai'] = False # Default if no user rating

    return df, file_source

# --- Analysis Functions (Operating on Preprocessed DF) ---

def get_top_keywords(reviews_subset_df: pd.DataFrame, top_n: int = 15) -> List[Tuple[str, int]]:
    """Aggregates keywords from the 'extracted_keywords_list' column."""
    if reviews_subset_df.empty or 'extracted_keywords_list' not in reviews_subset_df.columns:
        return []
    # Ensure robust handling of potential non-list entries after filtering
    keywords_series = reviews_subset_df['extracted_keywords_list'][reviews_subset_df['extracted_keywords_list'].apply(lambda x: isinstance(x, list))]
    all_keywords = [kw.lower() for sublist in keywords_series for kw in sublist if isinstance(kw, str)]
    return Counter(all_keywords).most_common(top_n)


def analyze_themes(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes frequency and average sentiment per theme."""
    if df.empty or 'review_themes' not in df.columns or 'ai_sentiment_rating' not in df.columns:
        return pd.DataFrame(columns=['Theme', 'Frequency', 'Average Sentiment'])

    theme_data = defaultdict(lambda: {'total_sentiment': 0, 'count': 0})
    for _, row in df.iterrows():
        themes = row['review_themes']
        sentiment = row['ai_sentiment_rating']
        # Ensure themes is a set and sentiment is numeric before processing
        if isinstance(themes, set) and isinstance(sentiment, (int, float)):
            for theme in themes:
                theme_data[theme]['total_sentiment'] += sentiment
                theme_data[theme]['count'] += 1

    theme_list = []
    for theme, data in theme_data.items():
        if data['count'] > 0:
            avg_sentiment = round(data['total_sentiment'] / data['count'], 2)
            theme_list.append({'Theme': theme, 'Frequency': data['count'], 'Average Sentiment': avg_sentiment})

    return pd.DataFrame(theme_list).sort_values(by='Frequency', ascending=False)

def get_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates key performance indicators for the executive summary."""
    kpis = {}
    total_reviews = len(df)
    kpis['total_reviews'] = total_reviews

    # Default values for when df is empty
    default_kpis = {
        'total_reviews': 0, 'avg_sentiment': None, 'pct_negative': None, 'pct_positive': None,
        'pct_low_user_high_ai': None, 'total_tos_violations': 0, 'critical_tos_flags': {},
        'pct_verified': None, 'avg_helpful_count': None
    }

    if total_reviews == 0:
        return default_kpis

    # Basic Sentiment KPIs
    kpis['avg_sentiment'] = round(df['ai_sentiment_rating'].mean(), 2) if 'ai_sentiment_rating' in df.columns else None
    kpis['pct_negative'] = round((df['ai_sentiment_rating'] <= LOW_SENTIMENT_THRESHOLD).sum() / total_reviews * 100, 1) if 'ai_sentiment_rating' in df.columns else None
    kpis['pct_positive'] = round((df['ai_sentiment_rating'] >= HIGH_SENTIMENT_THRESHOLD).sum() / total_reviews * 100, 1) if 'ai_sentiment_rating' in df.columns else None # Use >= for high

    # Mismatch KPI (Low User Rating, High AI Sentiment)
    if 'low_user_high_ai' in df.columns:
         mismatched_count = df['low_user_high_ai'].sum()
         # Calculate percentage based on reviews where user rating is available for a fair comparison
         valid_user_rating_count = df['review_score'].notna().sum()
         kpis['pct_low_user_high_ai'] = round(mismatched_count / valid_user_rating_count * 100, 1) if valid_user_rating_count > 0 else 0
    else:
         kpis['pct_low_user_high_ai'] = None


    # TOS Violation KPIs
    if 'has_tos_violation' in df.columns:
        kpis['total_tos_violations'] = df['has_tos_violation'].sum()
        # Count specific HIGH SEVERITY flags for summary
        high_severity_flags_counter = Counter()
        # Iterate over rows with violations and check severity dict
        for flags_dict, severity_dict in zip(df.loc[df['has_tos_violation'], 'tos_flags_dict'], df.loc[df['has_tos_violation'], 'tos_severity_dict']):
             if isinstance(flags_dict, dict) and isinstance(severity_dict, dict):
                  for flag, severity in severity_dict.items():
                       if severity == 'high':
                            high_severity_flags_counter[flag] += 1
        kpis['critical_tos_flags'] = dict(high_severity_flags_counter)
    else:
        kpis['total_tos_violations'] = 0
        kpis['critical_tos_flags'] = {}

    # Verified Review KPI
    if 'verified' in df.columns and df['verified'].notna().any():
        verified_count = df['verified'].sum()
        valid_verified_count = df['verified'].notna().sum()
        kpis['pct_verified'] = round(verified_count / valid_verified_count * 100, 1) if valid_verified_count > 0 else 0
    else:
        kpis['pct_verified'] = None

    # Helpfulness KPI
    if 'helpful_counts' in df.columns and df['helpful_counts'].notna().any():
         # Use mean of non-zero helpful counts, or overall mean? Let's use overall mean.
         kpis['avg_helpful_count'] = round(df['helpful_counts'].mean(), 2)
    else:
         kpis['avg_helpful_count'] = None

    return kpis

def analyze_verified_reviews(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyzes verified vs. unverified reviews."""
    results = {'summary': {}, 'sentiment_comparison': pd.DataFrame()}
    if df.empty or 'verified' not in df.columns or df['verified'].isna().all():
        return results # Return empty structure if no verified data

    verified_col = 'verified'
    sentiment_col = 'ai_sentiment_rating'

    # Summary counts and percentages
    total_reviews = len(df)
    verified_counts = df[verified_col].value_counts(dropna=False) # Include counts of NA if any
    results['summary']['total'] = total_reviews
    results['summary']['verified'] = verified_counts.get(True, 0)
    results['summary']['unverified'] = verified_counts.get(False, 0)
    results['summary']['unknown'] = verified_counts.get(None, 0) # Count those marked NA/None
    if total_reviews > 0:
        results['summary']['pct_verified'] = round(results['summary']['verified'] / total_reviews * 100, 1)
        results['summary']['pct_unverified'] = round(results['summary']['unverified'] / total_reviews * 100, 1)
        results['summary']['pct_unknown'] = round(results['summary']['unknown'] / total_reviews * 100, 1)


    # Sentiment Comparison (only if sentiment column exists)
    if sentiment_col in df.columns:
        # Group by verified status and calculate mean sentiment, count
        # Handle potential NA in verified column by filling temporarily for grouping
        sentiment_comp = df.fillna({verified_col: 'Unknown'}).groupby(verified_col)[sentiment_col].agg(['mean', 'count']).reset_index()
        sentiment_comp.rename(columns={'mean': 'Average AI Sentiment', 'count': 'Review Count', verified_col: 'Verified Status'}, inplace=True)
        # Map True/False back for clarity if needed, keep 'Unknown'
        sentiment_comp['Verified Status'] = sentiment_comp['Verified Status']
        results['sentiment_comparison'] = sentiment_comp

    return results


def analyze_helpful_reviews(df: pd.DataFrame, top_n_reviews=5) -> Dict[str, Any]:
    """Analyzes helpful review counts."""
    results = {'summary': {}, 'distribution': pd.DataFrame(), 'top_reviews': pd.DataFrame()}
    if df.empty or 'helpful_counts' not in df.columns or df['helpful_counts'].isna().all():
        return results # Return empty structure if no helpful data

    helpful_col = 'helpful_counts'
    sentiment_col = 'ai_sentiment_rating'
    content_col = 'review_content'
    id_col = 'id'

    total_reviews = len(df)
    total_helpful_votes = df[helpful_col].sum()
    avg_helpful_votes = df[helpful_col].mean()
    max_helpful_votes = df[helpful_col].max()
    reviews_with_votes = (df[helpful_col] > 0).sum()

    results['summary']['total_reviews'] = total_reviews
    results['summary']['total_helpful_votes'] = total_helpful_votes
    results['summary']['avg_helpful_votes'] = round(avg_helpful_votes, 2) if pd.notna(avg_helpful_votes) else 0
    results['summary']['max_helpful_votes'] = max_helpful_votes if pd.notna(max_helpful_votes) else 0
    results['summary']['reviews_with_votes'] = reviews_with_votes
    results['summary']['pct_reviews_with_votes'] = round(reviews_with_votes / total_reviews * 100, 1) if total_reviews > 0 else 0

    # Distribution (e.g., histogram bins)
    # Create bins for the histogram (e.g., 0, 1-5, 6-10, 11-20, 21+)
    bins = [-np.inf, 0, 5, 10, 20, np.inf]
    labels = ['0 Votes', '1-5 Votes', '6-10 Votes', '11-20 Votes', '21+ Votes']
    # Check if helpful_col exists and is numeric before cutting
    if pd.api.types.is_numeric_dtype(df[helpful_col]):
         # Use dropna=False if you want to see how many had NaN counts originally, though we filled them
         df['helpful_bin'] = pd.cut(df[helpful_col], bins=bins, labels=labels, right=True)
         distribution_counts = df['helpful_bin'].value_counts().reset_index()
         distribution_counts.columns = ['Helpful Votes Range', 'Number of Reviews']
         # Ensure correct categorical order for plotting
         distribution_counts['Helpful Votes Range'] = pd.Categorical(distribution_counts['Helpful Votes Range'], categories=labels, ordered=True)
         results['distribution'] = distribution_counts.sort_values('Helpful Votes Range')
    else:
         st.warning(f"Cannot generate helpfulness distribution: '{helpful_col}' is not numeric.")


    # Top N Helpful Reviews (if content column exists)
    if content_col in df.columns and id_col in df.columns:
        top_reviews_df = df.nlargest(top_n_reviews, helpful_col)
        cols_to_show = [id_col, helpful_col, content_col, sentiment_col, 'review_score', 'verified']
        # Filter cols_to_show to only include columns that actually exist in top_reviews_df
        existing_cols = [col for col in cols_to_show if col in top_reviews_df.columns]
        results['top_reviews'] = top_reviews_df[existing_cols]

    return results


# --- Sample Data (as string) ---
SAMPLE_DATA_CSV = """id,review_date,ASIN,helpful_counts,product_title,review_content,review_score,review_title,reviewer,verified,detect_tos_violation,analyze_rating_consistency,extract_keywords,analyze_review_sentiment
7a275b153c2286bfad6bd1dd10207d309cc1738649a6657105a76da3e6723b8a,2025-01-06 00:00:00,B0856SMBJC,0,"Gift Boutique 6 Christmas Front Door with Red Bow 13"" Winter Decoration Wall Decor Hanging Wreaths Kitchen Decorations Artificial Home Decor Holiday Indoor Window",Nice quality held up very nice out doors,1.0,Good Quality,Shannon. Baluta,True,"{""ratings"": {""clarity_and_relevance_one_to_five"": 3, ""compliance_with_community_guidelines_one_to_five"": 5, ""authenticity_and_credibility_one_to_five"": 4}, ""flags"": {""incentivized_or_promotional_content"": false, ""offensive_or_abusive_language"": false, ""personal_information_or_privacy_violation"": false, ""irrelevant_or_spammy_content"": false, ""no_tos_violations_detected"": true}, ""summary"": ""The review is compliant with Amazon's TOS. It is relevant and does not contain any violations, though it is somewhat vague.""}","{""consistency_evaluation"": {""overall_consistency"": false, ""consistency_score"": 1, ""notes"": ""The review text is positive, mentioning 'nice quality' and 'held up very nice outdoors', which contradicts the low 1-star rating.""}}","{""keywords"": [""quality"", ""held up"", ""outdoors""], ""explanation"": ""The keywords 'quality' and 'held up' are chosen as they describe the durability and performance aspect of the item. 'Outdoors' is selected as it specifies the context or environment where the item is used.""}","{""rating"": 8, ""explanation"": ""The review uses positive language such as 'nice quality' and 'held up very nice,' indicating satisfaction with the product's performance outdoors. The repetition of 'nice' suggests a strong positive sentiment, though not overwhelmingly enthusiastic.""}"
6ce56f9dfc77a2bdb830d36025b9af12b0c5b62631045e228710a44343497f0a,2025-01-03 00:00:00,B0856SMBJC,10,"Gift Boutique 6 Christmas Front Door with Red Bow 13"" Winter Decoration Wall Decor Hanging Wreaths Kitchen Decorations Artificial Home Decor Holiday Indoor Window",Nice touch to my small windows.,4.0,These were good,BOOK MAVEN,True,"{""ratings"": {""clarity_and_relevance_one_to_five"": 3, ""compliance_with_community_guidelines_one_to_five"": 5, ""authenticity_and_credibility_one_to_five"": 4}, ""flags"": {""incentivized_or_promotional_content"": false, ""offensive_or_abusive_language"": false, ""personal_information_or_privacy_violation"": false, ""irrelevant_or_spammy_content"": false, ""no_tos_violations_detected"": true}, ""summary"": ""The review is compliant with Amazon's TOS, though it is brief and somewhat vague. It does not contain any violations.""}","{""consistency_evaluation"": {""overall_consistency"": true, ""consistency_score"": 4, ""notes"": ""The review text is positive, indicating satisfaction, which aligns with a 4-star rating.""}}","{""keywords"": [""nice touch"", ""small windows""], ""explanation"": ""The keywords 'nice touch' and 'small windows' capture the essence of the sentence, focusing on the compliment ('nice touch') and the object of the compliment ('small windows').""}","{""rating"": 8, ""explanation"": ""The review expresses a positive sentiment, appreciating the addition to the windows. The use of 'nice touch' indicates satisfaction and approval, though it is not overly enthusiastic, hence an 8.""}"
34a9509fc13f76cf6d79f8ede04d50304aecfd3643ab183ad3266cd4fb400447,2025-01-02 00:00:00,B0856SMBJC,25,"Gift Boutique 6 Christmas Front Door with Red Bow 13"" Winter Decoration Wall Decor Hanging Wreaths Kitchen Decorations Artificial Home Decor Holiday Indoor Window",I bought these hoping for a cheaper alternative to a wreath look for the holidays. These are very cheaply made and don't even look like wreaths in person. The material almost feels like plastic strands that are spray painted.,1.0,Low quality doesnt even look like a wreath,Jacqueline S,True,"{""ratings"": {""clarity_and_relevance_one_to_five"": 4, ""compliance_with_community_guidelines_one_to_five"": 5, ""authenticity_and_credibility_one_to_five"": 5}, ""flags"": {""incentivized_or_promotional_content"": false, ""offensive_or_abusive_language"": false, ""personal_information_or_privacy_violation"": false, ""irrelevant_or_spammy_content"": false, ""no_tos_violations_detected"": true}, ""summary"": ""The review is clear, relevant, and compliant with Amazon's community guidelines. No TOS violations detected.""}","{""consistency_evaluation"": {""overall_consistency"": true, ""consistency_score"": 5, ""notes"": ""The review text is negative, highlighting disappointment with the product's quality and appearance, which aligns with the low 1-star rating.""}}","{""keywords"": [""cheaper alternative"", ""wreath"", ""holidays"", ""cheap"", ""plastic strands"", ""spray painted""], ""explanation"": ""The keywords were chosen based on the main themes and concepts in the text: the search for a budget-friendly option ('cheaper alternative'), the intended use during a specific time ('holidays'), the product being discussed ('wreath'), and the quality and material description ('cheap', 'plastic strands', 'spray painted').""}","{""rating"": 3, ""explanation"": ""The review expresses disappointment with the product's quality and appearance, indicating a negative sentiment, but not the most extreme dissatisfaction.""}"
0cf600c4f58ec8c15aede3e923ccd3f051ef6348fd34a3ecc07826c06490bcee,2025-01-01 00:00:00,B0856SMBJC,2,"Gift Boutique 6 Christmas Front Door with Red Bow 13"" Winter Decoration Wall Decor Hanging Wreaths Kitchen Decorations Artificial Home Decor Holiday Indoor Window",Loved these wreaths for my windows!!,5.0,"So soft, well-made, easy to hang!",Lissette,False,"{""ratings"": {""clarity_and_relevance_one_to_five"": 4, ""compliance_with_community_guidelines_one_to_five"": 5, ""authenticity_and_credibility_one_to_five"": 5}, ""flags"": {""incentivized_or_promotional_content"": false, ""offensive_or_abusive_language"": false, ""personal_information_or_privacy_violation"": false, ""irrelevant_or_spammy_content"": false, ""no_tos_violations_detected"": true}, ""summary"": ""The review is concise, relevant, and complies with Amazon's TOS. No violations detected.""}","{""consistency_evaluation"": {""overall_consistency"": true, ""consistency_score"": 5, ""notes"": ""The review text is highly positive and aligns well with the 5-star rating.""}}","{""keywords"": [""wreaths"", ""windows""], ""explanation"": ""The keywords 'wreaths' and 'windows' are chosen as they represent the main subjects of the text, indicating the items being referred to and their context of use.""}","{""rating"": 10, ""explanation"": ""The use of the word 'Loved' indicates a highly positive sentiment towards the wreaths, suggesting complete satisfaction and delight.""}"
# Added another sample row for TOS testing
a1b2c3d4e5f6,2025-01-07 00:00:00,B0856SMBJC,5,"Gift Boutique 6 Christmas Front Door with Red Bow 13"" Winter Decoration Wall Decor Hanging Wreaths Kitchen Decorations Artificial Home Decor Holiday Indoor Window",Got this for free for my review. It looks okay, kinda flimsy though. Arrived fast.",3.0,Okay for free item,ReviewerX,True,"{""ratings"": {""clarity_and_relevance_one_to_five"": 4, ""compliance_with_community_guidelines_one_to_five"": 2, ""authenticity_and_credibility_one_to_five"": 2}, ""flags"": {""incentivized_or_promotional_content"": true, ""offensive_or_abusive_language"": false, ""personal_information_or_privacy_violation"": false, ""irrelevant_or_spammy_content"": false, ""no_tos_violations_detected"": false}, ""summary"": ""Review indicates it was incentivized ('Got this for free for my review'), violating TOS against undisclosed incentives.""}","{""consistency_evaluation"": {""overall_consistency"": true, ""consistency_score"": 4, ""notes"": ""The review text expresses mixed feelings ('okay', 'kinda flimsy'), which is reasonably consistent with a 3-star rating.""}}","{""keywords"": [""free"", ""review"", ""okay"", ""flimsy"", ""fast""], ""explanation"": ""Keywords capture the incentivized nature, the mixed quality assessment, and shipping speed.""}","{""rating"": 6, ""explanation"": ""The review has mixed sentiment. 'Okay' and 'Arrived fast' are neutral to positive, while 'kinda flimsy' is negative. The mention of getting it 'free' doesn't directly impact product sentiment but is noted. Overall slightly positive leaning.""}"
"""


# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="AI Review Analysis Dashboard")
st.title("📊 FBA Product Review Analysis Dashboard") # Changed title emoji

# --- Sidebar for Controls ---
st.sidebar.header("⚙️ Controls")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Review Data CSV", type=["csv"])

# Load and process data (cached)
# Include sample data CSV string in the function call context if needed, though it's global now.
df_processed, source_name = load_and_preprocess_data(uploaded_file=uploaded_file) # Uses default/sample if no upload

if df_processed.empty:
    # Added more specific instructions if sample data also failed
    st.error("No data loaded. Please upload a CSV file or ensure 'openai_cache_export.csv' is present. If using sample data, check for errors above.")
    st.stop() # Stop execution if no data

# Date Range Filter (only if 'review_date' was parsed successfully and has multiple dates)
min_date = df_processed['review_date'].min()
max_date = df_processed['review_date'].max()

# Ensure min/max are valid Timestamps before proceeding
if pd.notna(min_date) and pd.notna(max_date) and min_date != max_date:
    st.sidebar.subheader("📅 Filter by Review Date")
    try:
        date_range = st.sidebar.date_input(
            "Select date range:",
            value=(min_date.date(), max_date.date()), # Use .date() for widget compatibility
            min_value=min_date.date(),
            max_value=max_date.date(),
            key='date_filter'
        )
        # Ensure date_range has two values before unpacking
        if len(date_range) == 2:
            start_date, end_date = date_range
            # Convert back to datetime for filtering (inclusive of the end date)
            # Handle NaT in start/end date just in case widget returns weird value
            if pd.notna(start_date) and pd.notna(end_date):
                start_datetime = pd.to_datetime(start_date)
                end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) # Include end date
                # Filter the DataFrame based on the selected date range
                df_filtered = df_processed[(df_processed['review_date'] >= start_datetime) & (df_processed['review_date'] < end_datetime)].copy()
                st.sidebar.info(f"Filtering data between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
            else:
                 st.sidebar.warning("Invalid date selection. Using all data.")
                 df_filtered = df_processed.copy()
        else:
            # Handle case where widget might not return 2 dates initially or on error
            st.sidebar.warning("Please select a valid start and end date.")
            df_filtered = df_processed.copy() # Use all data if date range is invalid

    except Exception as e:
         st.sidebar.error(f"Date filter error: {e}. Using all data.")
         df_filtered = df_processed.copy()

elif pd.notna(min_date): # Handle case with only one unique date or parsing issues yielding one date
     st.sidebar.info("Only one unique review date found or date range invalid. No date range filter applied.")
     df_filtered = df_processed.copy()
else:
    st.sidebar.info("Review date information not available or usable for filtering.")
    df_filtered = df_processed.copy() # Use all data if no dates


# Check if filtering resulted in an empty dataframe
if df_filtered.empty and not df_processed.empty : # Check original df isn't empty
     st.warning("No reviews found within the selected date range.")
     # Optionally stop, or just let sections show "no data" messages
     # st.stop()


st.sidebar.markdown("---")
st.sidebar.info(f"Data Source: {source_name}\n\n"
                f"Total Reviews in Source: {len(df_processed)}\n\n"
                f"Reviews in Filtered Range: {len(df_filtered)}")


# --- Main Dashboard Area ---

# 1. Executive Summary / KPI Snapshot
st.header("⭐ Executive Summary")
if not df_filtered.empty:
    kpis = get_kpis(df_filtered)
    # Adjusted columns based on added/removed KPIs
    cols_kpi_row1 = st.columns(4)
    cols_kpi_row2 = st.columns(4)

    cols_kpi_row1[0].metric("Total Reviews Analyzed", kpis['total_reviews'])
    cols_kpi_row1[1].metric("Avg. AI Sentiment (1-10)", f"{kpis['avg_sentiment']:.2f}" if kpis['avg_sentiment'] is not None else "N/A")
    cols_kpi_row1[2].metric(f"% Negative (AI ≤ {LOW_SENTIMENT_THRESHOLD})", f"{kpis['pct_negative']}%" if kpis['pct_negative'] is not None else "N/A", delta_color="inverse")
    cols_kpi_row1[3].metric(f"% Positive (AI ≥ {HIGH_SENTIMENT_THRESHOLD})", f"{kpis['pct_positive']}%" if kpis['pct_positive'] is not None else "N/A") # Changed threshold display

    # Use pct_low_user_high_ai instead of general mismatch
    cols_kpi_row2[0].metric("% Low Rating / High AI Sent.", f"{kpis['pct_low_user_high_ai']}%" if kpis['pct_low_user_high_ai'] is not None else "N/A")
    cols_kpi_row2[1].metric("Total TOS Violations", kpis['total_tos_violations'])
    cols_kpi_row2[2].metric("% Verified Reviews", f"{kpis['pct_verified']}%" if kpis['pct_verified'] is not None else "N/A")
    cols_kpi_row2[3].metric("Avg. Helpful Votes", f"{kpis['avg_helpful_count']:.2f}" if kpis['avg_helpful_count'] is not None else "N/A")


    # Display HIGH SEVERITY TOS flags if any
    if kpis['critical_tos_flags']:
        st.subheader("High Severity TOS Flags Found:")
        tos_flag_summary = ", ".join([f"{flag.replace('_', ' ').title()}: {count}" for flag, count in kpis['critical_tos_flags'].items()])
        st.warning(f"🚩 {tos_flag_summary}")
    # Only show success message if TOS check was possible
    elif 'has_tos_violation' in df_filtered.columns:
         st.success("✅ No high severity TOS flags detected in the selected period.")

else:
    st.info("No data available in the selected date range for summary.")


# 2. Trend Analysis (Requires Date Data)
st.header("📈 Trend Analysis")
date_col = 'review_date'
# Check if date column exists and has more than one unique, non-NaT date
if date_col in df_filtered.columns and df_filtered[date_col].nunique(dropna=True) > 1:
    # Ensure index is datetime before resampling
    df_filtered_dated = df_filtered.dropna(subset=[date_col]).set_index(date_col)

    if not df_filtered_dated.empty:
        # Choose frequency based on data span
        time_span_days = (df_filtered_dated.index.max() - df_filtered_dated.index.min()).days
        resample_freq = 'W-MON' # Default to weekly, starting Monday
        if time_span_days <= 30: # Use Daily if 1 month or less
             resample_freq = 'D'
        elif time_span_days > 365 * 1.5: # Monthly if > 1.5 years
            resample_freq = 'ME' # Month End frequency

        st.info(f"Resampling trend data by: {'Daily' if resample_freq=='D' else ('Weekly (Mon)' if resample_freq=='W-MON' else 'Monthly')}")

        try:
            # Define aggregations, checking if columns exist
            agg_dict = {
                'average_sentiment': pd.NamedAgg(column='ai_sentiment_rating', aggfunc='mean'),
                'review_count': pd.NamedAgg(column='id', aggfunc='count'),
                'negative_review_count': pd.NamedAgg(column='ai_sentiment_rating', aggfunc=lambda x: (x <= LOW_SENTIMENT_THRESHOLD).sum()),
                'positive_review_count': pd.NamedAgg(column='ai_sentiment_rating', aggfunc=lambda x: (x >= HIGH_SENTIMENT_THRESHOLD).sum()), # Use >=
            }
            # Add optional aggregations if columns exist
            if 'low_user_high_ai' in df_filtered_dated.columns:
                 agg_dict['mismatched_luh_count'] = pd.NamedAgg(column='low_user_high_ai', aggfunc='sum')
            if 'has_tos_violation' in df_filtered_dated.columns:
                 agg_dict['tos_violation_count'] = pd.NamedAgg(column='has_tos_violation', aggfunc='sum')
            if 'helpful_counts' in df_filtered_dated.columns:
                 agg_dict['average_helpful_count'] = pd.NamedAgg(column='helpful_counts', aggfunc='mean')


            df_trends = df_filtered_dated.resample(resample_freq).agg(**agg_dict).reset_index()

            # Remove potential NaN sentiment for plotting if a period has 0 reviews before calculating counts
            # df_trends.dropna(subset=['average_sentiment'], inplace=True) # Keep rows even if sentiment is NaN, maybe count is useful
            df_trends = df_trends[df_trends['review_count'] > 0].copy() # Filter out periods with zero reviews

            if not df_trends.empty:
                col_trend1, col_trend2 = st.columns(2)
                with col_trend1:
                    fig_trend_sentiment = px.line(df_trends, x=date_col, y='average_sentiment', title='Average AI Sentiment Over Time', markers=True, labels={date_col: "Date"})
                    fig_trend_sentiment.update_layout(yaxis_title="Avg. AI Sentiment Rating")
                    st.plotly_chart(fig_trend_sentiment, use_container_width=True)

                    # Plot Low User Rating / High AI Sentiment Trend (if available)
                    if 'mismatched_luh_count' in df_trends.columns:
                         fig_trend_mismatch = px.line(df_trends, x=date_col, y='mismatched_luh_count', title='Low Rating/High AI Reviews Over Time', markers=True, labels={date_col: "Date"})
                         fig_trend_mismatch.update_layout(yaxis_title="Count of Low Rating / High AI Sent.")
                         st.plotly_chart(fig_trend_mismatch, use_container_width=True)
                    else:
                         st.info("Low Rating/High AI trend requires 'review_score' column.")


                with col_trend2:
                    # Combined Volume Plot
                    fig_trend_volume = px.line(df_trends, x=date_col, y=['review_count', 'negative_review_count', 'positive_review_count'],
                                                title='Review Volume Over Time', markers=True, labels={date_col: "Date"})
                    fig_trend_volume.update_layout(yaxis_title="Number of Reviews", legend_title_text='Review Type')
                    st.plotly_chart(fig_trend_volume, use_container_width=True)

                    # Plot TOS Violation Trend (if available)
                    if 'tos_violation_count' in df_trends.columns:
                        fig_trend_tos = px.line(df_trends, x=date_col, y='tos_violation_count', title='TOS Violations Over Time', markers=True, labels={date_col: "Date"})
                        fig_trend_tos.update_layout(yaxis_title="Count of TOS Violations")
                        st.plotly_chart(fig_trend_tos, use_container_width=True)
                    else:
                        st.info("TOS trend requires 'detect_tos_violation' column processing.")

                    # # Optional: Plot Helpfulness Trend (if available)
                    # if 'average_helpful_count' in df_trends.columns:
                    #     fig_trend_helpful = px.line(df_trends, x=date_col, y='average_helpful_count', title='Average Helpfulness Over Time', markers=True)
                    #     fig_trend_helpful.update_layout(yaxis_title="Avg. Helpful Votes")
                    #     st.plotly_chart(fig_trend_helpful, use_container_width=True)

            else:
                st.info("Not enough data points in the selected range for meaningful trend analysis after resampling.")
        except Exception as e:
             st.error(f"Could not generate trend data. Error during resampling/aggregation: {e}")


    else:
         st.info("No valid date data found in the filtered range for trend analysis.")


else:
    st.info("Trend analysis requires a 'review_date' column with multiple unique dates in the selected range.")


# 3. Sentiment Analysis
st.header("😊 Sentiment Distribution & Keywords")

if not df_filtered.empty and 'ai_sentiment_rating' in df_filtered.columns:
    col_sent1, col_sent2 = st.columns([1, 2]) # Adjust column ratios if needed
    with col_sent1:
        # Sentiment Distribution Chart
        sentiment_counts = df_filtered['ai_sentiment_rating'].value_counts().sort_index()
        fig_sentiment_dist = px.bar(
            sentiment_counts,
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            labels={'x': 'AI Sentiment Rating (1-10)', 'y': 'Number of Reviews'},
            title="Distribution of AI Sentiment Ratings"
        )
        fig_sentiment_dist.update_layout(xaxis={'type': 'category'}) # Treat ratings as categories
        st.plotly_chart(fig_sentiment_dist, use_container_width=True)

    with col_sent2:
        st.subheader("Keywords by Sentiment Segment")
        # Filter data based on thresholds
        negative_reviews_df = df_filtered[df_filtered['ai_sentiment_rating'] <= LOW_SENTIMENT_THRESHOLD]
        positive_reviews_df = df_filtered[df_filtered['ai_sentiment_rating'] >= HIGH_SENTIMENT_THRESHOLD] # Use >= for high

        # Calculate top keywords
        top_neg_keywords = get_top_keywords(negative_reviews_df, top_n=10)
        top_pos_keywords = get_top_keywords(positive_reviews_df, top_n=10)

        # Display Keywords side-by-side
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            st.markdown(f"**Top Keywords in Low Sentiment Reviews (AI ≤ {LOW_SENTIMENT_THRESHOLD})**")
            if top_neg_keywords:
                neg_kw_df = pd.DataFrame(top_neg_keywords, columns=['Keyword', 'Frequency'])
                fig_neg = px.bar(neg_kw_df.sort_values('Frequency', ascending=True), x='Frequency', y='Keyword', orientation='h')
                fig_neg.update_layout(yaxis_title=None, height=300) # Compact layout
                st.plotly_chart(fig_neg, use_container_width=True)
            else:
                st.write("No low sentiment reviews or keywords found.")

        with subcol2:
            st.markdown(f"**Top Keywords in High Sentiment Reviews (AI ≥ {HIGH_SENTIMENT_THRESHOLD})**") # Use >=
            if top_pos_keywords:
                pos_kw_df = pd.DataFrame(top_pos_keywords, columns=['Keyword', 'Frequency'])
                fig_pos = px.bar(pos_kw_df.sort_values('Frequency', ascending=True), x='Frequency', y='Keyword', orientation='h',
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_pos.update_layout(yaxis_title=None, height=300) # Compact layout
                st.plotly_chart(fig_pos, use_container_width=True)
            else:
                st.write("No high sentiment reviews or keywords found.")
else:
    st.info("Sentiment data not available for distribution/keyword plot.")

# 4. Financial & Value Perception Analysis - REMOVED as per request


# 5. Thematic Analysis of Feedback (Renumbered from 5 to 4)
st.header("📊 Thematic Analysis")
if not df_filtered.empty and 'review_themes' in df_filtered.columns:
    theme_analysis_df = analyze_themes(df_filtered)

    if not theme_analysis_df.empty:
        st.subheader("Review Themes: Frequency and Sentiment")

        # Allow sorting
        sort_by = st.selectbox("Sort themes by:", ['Frequency', 'Average Sentiment'], index=0, key="theme_sort")
        theme_analysis_df_sorted = theme_analysis_df.sort_values(by=sort_by, ascending=(sort_by == 'Average Sentiment')) # Ascending for sentiment, Descending for freq

        # Plot using Plotly bar chart
        fig_themes = px.bar(theme_analysis_df_sorted,
                            x=sort_by,
                            y='Theme',
                            color='Average Sentiment',
                            orientation='h',
                            color_continuous_scale=px.colors.sequential.RdBu, # Red-Blue scale
                            range_color=[1, 10], # Fix color scale range
                            title=f"Review Themes Sorted by {sort_by}",
                            labels={'Average Sentiment': 'Avg. Sentiment (1-10)'},
                            hover_data=['Frequency', 'Average Sentiment'])
        fig_themes.update_layout(yaxis_title=None, height=max(400, len(theme_analysis_df_sorted)*30)) # Dynamic height
        st.plotly_chart(fig_themes, use_container_width=True)

        # Optional: Display theme data table
        if st.checkbox("Show Theme Data Table", value=False, key="theme_table_cb"):
            st.dataframe(theme_analysis_df_sorted)
    else:
        st.info("Could not extract themes from the reviews.")
else:
    st.info("Theme analysis requires theme mapping during preprocessing.")


# 6. Review Integrity & Risk Analysis (Renumbered from 6 to 5)
st.header("🛡️ Review Integrity & Risk Analysis")

col_int1, col_int2 = st.columns(2)

with col_int1:
    st.subheader("Low User Rating / High AI Sentiment")
    # Requires both review_score and ai_sentiment_rating
    if 'review_score' in df_filtered.columns and 'ai_sentiment_rating' in df_filtered.columns and 'low_user_high_ai' in df_filtered.columns:
        mismatched_focus_df = df_filtered[df_filtered['low_user_high_ai'] == True]
        mismatch_focus_count = len(mismatched_focus_df)

        # Calculate percentage based on reviews where user rating is available
        valid_user_rating_count = df_filtered['review_score'].notna().sum()
        mismatch_focus_percentage = round(mismatch_focus_count / valid_user_rating_count * 100, 1) if valid_user_rating_count > 0 else 0

        st.metric(f"Reviews with Low Rating (≤{LOW_USER_RATING_THRESHOLD}) & High AI Sentiment (≥{HIGH_SENTIMENT_THRESHOLD})",
                  mismatch_focus_count, f"{mismatch_focus_percentage}% of rated reviews") # Use >=

        if mismatch_focus_count > 0:
            # Show top keywords in these specific mismatched reviews
            st.markdown("**Common Keywords in these Reviews:**")
            top_mismatch_keywords = get_top_keywords(mismatched_focus_df, top_n=5)
            if top_mismatch_keywords:
                st.markdown(f"`{', '.join([kw[0] for kw in top_mismatch_keywords])}`")
            else:
                st.write("_No recurring keywords found._")

            # Scatter plot of User Rating vs AI Sentiment for these mismatches
            fig_mismatch_scatter = px.scatter(mismatched_focus_df,
                                               x='review_score',
                                               y='ai_sentiment_rating',
                                               title=f'User Rating vs AI Sentiment (Low Rating ≤{LOW_USER_RATING_THRESHOLD} / High AI ≥{HIGH_SENTIMENT_THRESHOLD})',
                                               labels={'review_score': 'User Rating', 'ai_sentiment_rating': 'AI Sentiment'},
                                               hover_data=['id']) # Add hover data if needed
            fig_mismatch_scatter.update_layout(xaxis = dict(tickmode = 'linear', tick0 = 1, dtick = 1), # Force integer ticks for rating
                                               yaxis = dict(range=[HIGH_SENTIMENT_THRESHOLD-0.5, 10.5])) # Focus y-axis
            st.plotly_chart(fig_mismatch_scatter, use_container_width=True)


            if st.checkbox("Show Low Rating / High AI Sentiment Review Details", value=False, key="mismatch_details_cb"):
                cols_to_show = ['id', 'review_score', 'ai_sentiment_rating', 'is_consistent', 'review_content', 'extracted_keywords_list', 'analyze_rating_consistency', 'analyze_review_sentiment']
                cols_exist = [col for col in cols_to_show if col in mismatched_focus_df.columns]
                st.dataframe(mismatched_focus_df[cols_exist])
        else:
             st.info("No reviews found matching the low user rating / high AI sentiment criteria.")
    else:
        st.info("Low Rating / High AI Sentiment analysis requires 'review_score' and 'ai_sentiment_rating' columns.")


with col_int2:
    st.subheader("🚨 Terms of Service (TOS) Violations")
    if 'has_tos_violation' in df_filtered.columns and 'highest_tos_severity' in df_filtered.columns:
        tos_flagged_df = df_filtered[df_filtered['has_tos_violation'] == True].copy() # Work on a copy
        st.metric("Total Reviews with Potential TOS Flags", len(tos_flagged_df))

        if not tos_flagged_df.empty:
            # Summarize flag types and their counts / severities
            all_flags_counter = Counter()
            flag_details = [] # Store flag, severity, count
            for _, row in tos_flagged_df.iterrows():
                flags = row['tos_flags_dict']
                severities = row['tos_severity_dict']
                if isinstance(flags, dict):
                     all_flags_counter.update(flags.keys())
                     for flag in flags.keys():
                          flag_details.append({
                               'Flag Type': flag,
                               'Severity': severities.get(flag, 'unknown')
                          })

            flags_summary_df = pd.DataFrame(flag_details).groupby(['Flag Type', 'Severity']).size().reset_index(name='Count')
            flags_summary_df['Flag Type'] = flags_summary_df['Flag Type'].str.replace('_', ' ').str.title() # Make readable
            flags_summary_df = flags_summary_df.sort_values(by=['Severity', 'Count'], ascending=[False, False]) # Sort by severity then count

            # --- Severity Filter ---
            available_severities = sorted(flags_summary_df['Severity'].unique(), key=lambda s: ["low", "medium", "high", "unknown"].index(s))
            # Default to high and medium if present, otherwise all available
            default_selection = [s for s in available_severities if s in ['high', 'medium']]
            if not default_selection: default_selection = available_severities # Select all if no high/medium found


            selected_severities = st.multiselect(
                 "Filter by TOS Violation Severity:",
                 options=available_severities,
                 default=default_selection,
                 format_func=lambda x: x.title() # Capitalize severity for display
            )

            # Filter the summary table and the original flagged df
            filtered_summary_df = flags_summary_df[flags_summary_df['Severity'].isin(selected_severities)]
            filtered_tos_reviews_df = tos_flagged_df[tos_flagged_df['highest_tos_severity'].isin(selected_severities)]

            st.markdown("**TOS Flag Types Found (Filtered):**")
            if not filtered_summary_df.empty:
                 # Display filtered summary table
                 st.dataframe(filtered_summary_df)

                 # Plot distribution of filtered flags
                 fig_tos_dist = px.bar(filtered_summary_df, x='Count', y='Flag Type', color='Severity',
                                        title='Distribution of Filtered TOS Flags by Severity',
                                        orientation='h', category_orders={"Severity": ["unknown", "low", "medium", "high"]}, # Ensure order
                                        color_discrete_map={'high': 'red', 'medium': 'orange', 'low': 'yellow', 'unknown': 'grey'})
                 fig_tos_dist.update_layout(yaxis_title=None, height=max(300, len(filtered_summary_df)*25))
                 st.plotly_chart(fig_tos_dist, use_container_width=True)

            else:
                 st.info("No TOS flags found matching the selected severity levels.")


            # Allow viewing details of the filtered reviews
            if st.checkbox("Show Filtered TOS-Flagged Reviews Details", value=False, key="tos_details_cb"):
                if not filtered_tos_reviews_df.empty:
                    # Show specific flags clearly in the table
                    def format_flags(row):
                         flags = row['tos_flags_dict']
                         severities = row['tos_severity_dict']
                         if isinstance(flags, dict):
                             return ", ".join([f"{f.replace('_', ' ').title()} ({severities.get(f, '?').title()})" for f in flags.keys()])
                         return ""

                    filtered_tos_reviews_df['Detected Flags (Severity)'] = filtered_tos_reviews_df.apply(format_flags, axis=1)
                    cols_to_show_tos = ['id', 'Detected Flags (Severity)', 'review_content', 'detect_tos_violation'] # Show formatted flags & content
                    cols_exist_tos = [col for col in cols_to_show_tos if col in filtered_tos_reviews_df.columns]
                    st.dataframe(filtered_tos_reviews_df[cols_exist_tos])
                else:
                    st.write("No reviews match the selected TOS severity filter.")
        else:
            st.info("No TOS violations detected in the filtered dataset.")
    else:
        st.info("TOS violation analysis requires 'detect_tos_violation' column processing with severity.")


# --- NEW SECTION: Verified Review Analysis --- (Renumbered to 6)
st.header("✅ Verified Purchase Analysis")
if 'verified' in df_filtered.columns and not df_filtered['verified'].isna().all():
    verified_analysis = analyze_verified_reviews(df_filtered)

    col_ver1, col_ver2 = st.columns(2)

    with col_ver1:
        st.subheader("Verification Status Distribution")
        if verified_analysis['summary']:
            summary = verified_analysis['summary']
            # Create data for pie chart
            labels = ['Verified', 'Unverified', 'Unknown']
            values = [summary.get('verified', 0), summary.get('unverified', 0), summary.get('unknown', 0)]
            # Filter out zero values to avoid cluttering pie chart
            labels = [l for l, v in zip(labels, values) if v > 0]
            values = [v for v in values if v > 0]

            if values: # Only plot if there are values
                fig_pie_verified = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3,
                                                           marker_colors=px.colors.qualitative.Pastel)]) # Use Plotly GO for pie
                fig_pie_verified.update_layout(title_text='Review Verification Status')
                st.plotly_chart(fig_pie_verified, use_container_width=True)
            else:
                 st.info("No verification data to display.")

        else:
             st.info("Could not calculate verification summary.")


    with col_ver2:
        st.subheader("Sentiment by Verification Status")
        sentiment_comp_df = verified_analysis.get('sentiment_comparison')
        if sentiment_comp_df is not None and not sentiment_comp_df.empty:
            # Ensure 'Verified Status' column exists before plotting
            if 'Verified Status' in sentiment_comp_df.columns:
                fig_bar_verified_sent = px.bar(sentiment_comp_df,
                                               x='Verified Status', y='Average AI Sentiment',
                                               color='Verified Status', text='Average AI Sentiment',
                                               title="Average AI Sentiment: Verified vs. Unverified",
                                               labels={'Average AI Sentiment': 'Avg. AI Sentiment (1-10)'})
                fig_bar_verified_sent.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig_bar_verified_sent.update_layout(yaxis=dict(range=[0, 10.5])) # Set y-axis range 0-10
                st.plotly_chart(fig_bar_verified_sent, use_container_width=True)
            else:
                 st.warning("Column 'Verified Status' not found in sentiment comparison data.")

            if st.checkbox("Show Verified Sentiment Data Table", value=False, key="verified_table_cb"):
                 st.dataframe(sentiment_comp_df)
        else:
            st.info("Could not compare sentiment by verification status. Requires 'verified' and 'ai_sentiment_rating' columns.")
else:
    st.info("Verified review analysis requires a 'verified' column in the data.")

# --- NEW SECTION: Helpful Review Analysis --- (Renumbered to 7)
st.header("👍 Helpful Votes Analysis")
if 'helpful_counts' in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered['helpful_counts']):
    helpful_analysis = analyze_helpful_reviews(df_filtered)

    col_help1, col_help2 = st.columns(2)

    with col_help1:
        st.subheader("Helpfulness Summary")
        if helpful_analysis['summary']:
            summary = helpful_analysis['summary']
            st.metric("Reviews with > 0 Helpful Votes", f"{summary.get('reviews_with_votes', 0)} ({summary.get('pct_reviews_with_votes', 0):.1f}%)")
            st.metric("Average Helpful Votes (per review)", f"{summary.get('avg_helpful_votes', 0):.2f}")
            st.metric("Max Helpful Votes on a Single Review", f"{summary.get('max_helpful_votes', 0)}")
            st.metric("Total Helpful Votes Received", f"{summary.get('total_helpful_votes', 0)}")
        else:
             st.info("Could not calculate helpfulness summary.")

        st.subheader("Helpfulness Distribution")
        dist_df = helpful_analysis.get('distribution')
        if dist_df is not None and not dist_df.empty:
            fig_hist_helpful = px.bar(dist_df, x='Helpful Votes Range', y='Number of Reviews',
                                      title="Distribution of Helpful Votes per Review")
            st.plotly_chart(fig_hist_helpful, use_container_width=True)
        else:
            st.info("Could not generate helpfulness distribution chart.")


    with col_help2:
        st.subheader(f"Top {len(helpful_analysis.get('top_reviews', []))} Most Helpful Reviews")
        top_reviews_df = helpful_analysis.get('top_reviews')
        if top_reviews_df is not None and not top_reviews_df.empty:
             # Select and rename columns for display
             display_cols = {
                  'id': 'ID',
                  'helpful_counts': 'Helpful Votes',
                  'review_content': 'Review Text',
                  'review_score': 'User Rating',
                  'ai_sentiment_rating': 'AI Sentiment',
                  'verified': 'Verified'
             }
             # Filter display_cols to only include columns that actually exist
             cols_to_display = {k: v for k, v in display_cols.items() if k in top_reviews_df.columns}
             display_df = top_reviews_df[list(cols_to_display.keys())].rename(columns=cols_to_display)

             # Show limited text initially
             display_df['Review Text'] = display_df['Review Text'].str[:150] + '...' # Truncate for display
             st.dataframe(display_df)
             # Optionally show full text if checkbox is ticked
             if st.checkbox("Expand Review Text for Top Helpful Reviews", value=False, key="helpful_expand_cb"):
                  full_text_cols = {k: v for k, v in display_cols.items() if k in top_reviews_df.columns} # Use original names for lookup
                  full_text_df = top_reviews_df[list(full_text_cols.keys())].rename(columns=full_text_cols)
                  st.dataframe(full_text_df) # Show non-truncated version
        else:
            st.info("Could not retrieve top helpful reviews. Requires 'helpful_counts' and 'review_content' columns.")

else:
    st.info("Helpful review analysis requires a numeric 'helpful_counts' column in the data.")


# --- Optional: Raw Data View --- (Renumbered to 8)
st.header("📄 Raw Data Viewer")
if st.checkbox("Show Filtered Processed Data Sample", value=False, key="raw_data_cb"):
    st.info("Displaying the first 100 rows of the filtered and processed data.")
    # Show potentially relevant columns, check existence first
    all_possible_cols = [
        'id', 'review_date', 'review_score', 'ai_sentiment_rating', 'verified', 'helpful_counts',
        'is_consistent', 'low_user_high_ai', 'has_tos_violation', 'highest_tos_severity',
        'review_themes', 'extracted_keywords_list', 'review_content'
        ]
    available_cols = [col for col in all_possible_cols if col in df_filtered.columns]
    st.dataframe(df_filtered[available_cols].head(100))

st.markdown("---")
st.caption("Dashboard updated to include verified/helpful analysis, refined consistency & TOS sections.") # Updated caption
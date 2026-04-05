
import streamlit as st
import pandas as pd
import numpy as np
import ast
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from rapidfuzz import process
from datetime import datetime

# =========================================================
# ADVANCED MOVIE RECOMMENDER SYSTEM
# Includes:
# - Content-Based Recommendation
# - Collaborative Filtering
# - Hybrid Recommendation
# - Explore Mode
# - New User Recommendation
# - Mood / Personality Filtering
# - Hidden Gems
# - Explainable Recommendations
# - Fuzzy Search
# - Model Deep-Dive
# - Evaluation Metrics (Precision / Recall / F1 / MSE / RMSE)
# - User Satisfaction Questionnaire
# =========================================================

st.set_page_config(page_title="Advanced Movie Expert System", layout="wide")

# -----------------------------
# 1. DATA LOADING & PREPROCESSING
# -----------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies_metadata.csv", low_memory=False)
    ratings = pd.read_csv("ratings_small.csv")
    links = pd.read_csv("links_small.csv")

    required_movie_cols = ['id', 'title', 'overview', 'genres', 'release_date', 'vote_average', 'vote_count']
    movies = movies[required_movie_cols].copy()
    movies = movies.dropna(subset=['id', 'title', 'overview'])

    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    movies = movies.dropna(subset=['id'])
    movies['id'] = movies['id'].astype(int)

    def extract_genres(genre_str):
        try:
            genre_list = ast.literal_eval(genre_str)
            return " ".join([g['name'] for g in genre_list])
        except:
            return ""

    movies['genres_clean'] = movies['genres'].apply(extract_genres)

    links = links[['movieId', 'tmdbId']].copy()
    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')
    links = links.dropna(subset=['tmdbId'])
    links['tmdbId'] = links['tmdbId'].astype(int)

    merged = pd.merge(links, movies, left_on='tmdbId', right_on='id', how='inner')
    merged = merged[['movieId', 'title', 'overview', 'genres_clean', 'release_date', 'vote_average', 'vote_count']].copy()
    merged = merged.drop_duplicates(subset='movieId')
    merged = merged.drop_duplicates(subset='title').reset_index(drop=True)

    merged['combined_features'] = (
        merged['overview'].fillna('') + " " +
        (merged['genres_clean'].fillna('') + " ") * 3
    )

    merged['vote_average'] = pd.to_numeric(merged['vote_average'], errors='coerce').fillna(0)
    merged['vote_count'] = pd.to_numeric(merged['vote_count'], errors='coerce').fillna(0).astype(int)
    merged['release_date'] = merged['release_date'].fillna('Unknown')

    return merged, ratings


@st.cache_resource
def compute_models(movies_merged, ratings):
    # Content model
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_merged['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Collaborative model
    ratings_movies = pd.merge(ratings, movies_merged[['movieId', 'title']], on='movieId', how='inner')
    user_movie_matrix = ratings_movies.pivot_table(index='userId', columns='title', values='rating')
    user_movie_matrix_filled = user_movie_matrix.fillna(0)

    movie_similarity = cosine_similarity(user_movie_matrix_filled.T)
    movie_similarity_df = pd.DataFrame(
        movie_similarity,
        index=user_movie_matrix_filled.columns,
        columns=user_movie_matrix_filled.columns
    )

    indices = pd.Series(movies_merged.index, index=movies_merged['title']).drop_duplicates()

    return tfidf_matrix, cosine_sim, user_movie_matrix, user_movie_matrix_filled, movie_similarity_df, indices


movies_merged, ratings_data = load_data()
tfidf_matrix, cosine_sim, user_movie_matrix, user_movie_matrix_filled, movie_similarity_df, indices = compute_models(
    movies_merged, ratings_data
)

# -----------------------------
# 2. CORE RECOMMENDATION ENGINES
# -----------------------------
def recommend_content(title, top_n=10):
    if title not in indices:
        return pd.DataFrame()

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1: top_n + 30]

    recs = pd.DataFrame({
        'title': [movies_merged.iloc[i[0]]['title'] for i in sim_scores],
        'model_score': [i[1] for i in sim_scores]
    })

    recs = pd.merge(
        recs,
        movies_merged[['title', 'overview', 'genres_clean', 'release_date', 'vote_average', 'vote_count']],
        on='title',
        how='left'
    )

    recs = recs[recs['vote_count'] > 50]
    recs = recs.drop_duplicates(subset='title')
    return recs.reset_index(drop=True)


def recommend_collaborative(title, top_n=10):
    if title not in movie_similarity_df.columns:
        return pd.DataFrame()

    sim_scores = movie_similarity_df[title].sort_values(ascending=False)
    sim_scores = sim_scores.iloc[1: top_n + 30]

    recs = pd.DataFrame({
        'title': sim_scores.index,
        'model_score': sim_scores.values
    })

    recs = pd.merge(
        recs,
        movies_merged[['title', 'overview', 'genres_clean', 'release_date', 'vote_average', 'vote_count']],
        on='title',
        how='left'
    )

    recs = recs[recs['vote_count'] > 50]
    recs = recs.drop_duplicates(subset='title')
    return recs.reset_index(drop=True)


def hybrid_recommend(movie_title, top_n=10, alpha=0.5):
    if movie_title not in indices or movie_title not in movie_similarity_df.columns:
        return pd.DataFrame()

    content_df = recommend_content(movie_title, top_n=100).rename(columns={'model_score': 'content_score'})
    collab_df = recommend_collaborative(movie_title, top_n=100).rename(columns={'model_score': 'collab_score'})

    hybrid_df = pd.merge(
        content_df[['title', 'content_score']],
        collab_df[['title', 'collab_score']],
        on='title',
        how='outer'
    )

    hybrid_df['content_score'] = hybrid_df['content_score'].fillna(0)
    hybrid_df['collab_score'] = hybrid_df['collab_score'].fillna(0)

    if hybrid_df['content_score'].max() != 0:
        hybrid_df['content_score'] = hybrid_df['content_score'] / hybrid_df['content_score'].max()

    if hybrid_df['collab_score'].max() != 0:
        hybrid_df['collab_score'] = hybrid_df['collab_score'] / hybrid_df['collab_score'].max()

    hybrid_df['model_score'] = (
        alpha * hybrid_df['content_score'] +
        (1 - alpha) * hybrid_df['collab_score']
    )

    hybrid_df = pd.merge(
        hybrid_df,
        movies_merged[['title', 'overview', 'genres_clean', 'release_date', 'vote_average', 'vote_count']],
        on='title',
        how='left'
    )

    hybrid_df = hybrid_df[hybrid_df['vote_count'] > 50]
    hybrid_df = hybrid_df.drop_duplicates(subset='title')
    hybrid_df = hybrid_df.sort_values(by='model_score', ascending=False)
    return hybrid_df.reset_index(drop=True)


def get_explore_mode(movie_title, top_n=5, alpha=0.5):
    results = hybrid_recommend(movie_title, top_n=20, alpha=alpha)
    if results.empty:
        return results

    explore_results = results.iloc[5:5 + top_n].copy()
    if explore_results.empty:
        explore_results = results.head(top_n).copy()

    explore_results['explanation'] = (
        "Recommended to help you explore something less obvious but still relevant."
    )
    return explore_results.reset_index(drop=True)


def get_new_user_recommendations(personality="None", mood="None", top_n=10):
    results = movies_merged.copy()

    results = apply_filters(results, mood, personality, hidden_gems=False)
    results = results.sort_values(by=['vote_average', 'vote_count'], ascending=[False, False])
    results = results[results['vote_count'] > 100]
    results = results.drop_duplicates(subset='title').head(top_n).copy()

    results['model_score'] = results['vote_average']
    results['explanation'] = (
        "Recommended for new users based on mood, personality, popularity, and movie quality."
    )
    return results.reset_index(drop=True)


# -----------------------------
# 3. FILTERS & EXPLAINABILITY
# -----------------------------
def mood_to_genres():
    return {
        "Happy": ["Comedy", "Family", "Animation"],
        "Sad": ["Drama"],
        "Romantic": ["Romance"],
        "Excited": ["Action", "Adventure", "Thriller", "Science Fiction"],
        "Curious": ["Mystery", "Documentary", "History"],
        "Scared": ["Horror", "Thriller"],
        "Relaxed": ["Music", "Family", "Comedy"]
    }


def personality_to_genres():
    return {
        "Adventurer": ["Adventure", "Action"],
        "Romantic": ["Romance", "Drama"],
        "Thinker": ["Science Fiction", "Documentary", "Mystery"],
        "Fun Lover": ["Comedy", "Animation", "Family"],
        "Dreamer": ["Fantasy", "Science Fiction", "Animation"],
        "Bold Explorer": ["Action", "Thriller", "Adventure"]
    }


def apply_filters(df, mood="None", personality="None", hidden_gems=False):
    filtered = df.copy()

    if mood != "None":
        genres = mood_to_genres().get(mood, [])
        if genres:
            pattern = "|".join(genres)
            filtered = filtered[filtered['genres_clean'].str.contains(pattern, case=False, na=False)]

    if personality != "None":
        genres = personality_to_genres().get(personality, [])
        if genres:
            pattern = "|".join(genres)
            filtered = filtered[filtered['genres_clean'].str.contains(pattern, case=False, na=False)]

    if hidden_gems:
        filtered = filtered[(filtered['vote_average'] >= 6.5) & (filtered['vote_count'] < 500)]

    return filtered.reset_index(drop=True)


def explain_recommendation(row, mode):
    genres = row['genres_clean'] if pd.notna(row.get('genres_clean', None)) else "related genres"

    if mode == "Content-Based":
        return f"Recommended because it has similar storyline and genres: {genres}."
    elif mode == "Collaborative":
        return "Recommended because users who liked the selected movie also liked this movie."
    elif mode == "Hybrid":
        return f"Recommended using both content similarity and user rating behaviour, with related genres: {genres}."
    elif mode == "Explore":
        return "Recommended to help you discover something different from the usual top picks."
    elif mode == "New User":
        return "Recommended based on your selected vibe, personality, and highly rated movies."
    return "Recommended based on system analysis."


def get_recommendations(movie_title=None, mode="Hybrid", top_n=10, mood="None",
                        hidden_gems=False, personality="None", alpha=0.5):
    if mode == "New User":
        results = get_new_user_recommendations(personality=personality, mood=mood, top_n=max(top_n, 30))

    elif mode == "Explore":
        results = get_explore_mode(movie_title, top_n=max(top_n, 10), alpha=alpha)

    elif mode == "Content-Based":
        results = recommend_content(movie_title, top_n=max(top_n, 30))

    elif mode == "Collaborative":
        results = recommend_collaborative(movie_title, top_n=max(top_n, 30))

    else:
        results = hybrid_recommend(movie_title, top_n=max(top_n, 30), alpha=alpha)

    if results.empty:
        return results

    if mode != "New User":
        results = apply_filters(results, mood, personality, hidden_gems)

    results = results.head(top_n).copy()

    if 'explanation' not in results.columns:
        results['explanation'] = results.apply(
            lambda row: explain_recommendation(row, mode),
            axis=1
        )

    return results.reset_index(drop=True)


# -----------------------------
# 4. SEARCH
# -----------------------------
def fuzzy_search_movies(query, limit=20, score_cutoff=60):
    all_titles = movies_merged['title'].dropna().tolist()
    fuzzy_matches = process.extract(query, all_titles, limit=limit, score_cutoff=score_cutoff)
    found_titles = [m[0] for m in fuzzy_matches]
    search_results = movies_merged[movies_merged['title'].isin(found_titles)].copy()
    return search_results


# -----------------------------
# 5. EVALUATION METRICS
# -----------------------------
def predict_item_based_rating(user_id, target_title, top_k=10):
    if target_title not in user_movie_matrix_filled.columns:
        return np.nan
    if user_id not in user_movie_matrix_filled.index:
        return np.nan

    user_ratings = user_movie_matrix_filled.loc[user_id]
    rated_titles = user_ratings[user_ratings > 0].index.tolist()

    if len(rated_titles) == 0:
        return np.nan

    sims = []
    for title in rated_titles:
        if title != target_title and title in movie_similarity_df.index:
            sim = movie_similarity_df.loc[target_title, title]
            rating = user_ratings[title]
            sims.append((title, sim, rating))

    if not sims:
        return np.nan

    sims = sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]

    numerator = 0
    denominator = 0
    for _, sim, rating in sims:
        if sim > 0:
            numerator += sim * rating
            denominator += abs(sim)

    if denominator == 0:
        return np.nan

    return numerator / denominator


@st.cache_data
def evaluate_rating_prediction(sample_size=300):
    ratings_movies = pd.merge(
        ratings_data,
        movies_merged[['movieId', 'title']],
        on='movieId',
        how='inner'
    ).dropna(subset=['title'])

    eval_df = ratings_movies[['userId', 'title', 'rating']].copy()

    if len(eval_df) > sample_size:
        eval_df = eval_df.sample(sample_size, random_state=42)

    actuals = []
    preds = []

    for _, row in eval_df.iterrows():
        pred = predict_item_based_rating(row['userId'], row['title'], top_k=10)
        if not np.isnan(pred):
            actuals.append(row['rating'])
            preds.append(pred)

    if len(actuals) == 0:
        return None

    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)

    return {
        "count": len(actuals),
        "mse": float(mse),
        "rmse": float(rmse),
        "actuals": actuals,
        "preds": preds
    }


def precision_recall_f1_at_k(k=5, positive_threshold=4.0, user_limit=80):
    ratings_movies = pd.merge(
        ratings_data,
        movies_merged[['movieId', 'title']],
        on='movieId',
        how='inner'
    )

    users = ratings_movies['userId'].drop_duplicates().tolist()[:user_limit]

    y_true_all = []
    y_pred_all = []

    for user_id in users:
        user_rows = ratings_movies[ratings_movies['userId'] == user_id].copy()
        user_rows = user_rows.sort_values('rating', ascending=False)

        liked = user_rows[user_rows['rating'] >= positive_threshold]
        disliked = user_rows[user_rows['rating'] < positive_threshold]

        if liked.empty:
            continue

        seed_title = liked.iloc[0]['title']
        recs = hybrid_recommend(seed_title, top_n=k, alpha=0.5)

        if recs.empty:
            continue

        relevant_titles = set(liked['title'].tolist())
        recommended_titles = set(recs.head(k)['title'].tolist())

        # Build binary labels over union
        candidate_titles = list(recommended_titles.union(relevant_titles))
        if not candidate_titles:
            continue

        for title in candidate_titles:
            y_true_all.append(1 if title in relevant_titles else 0)
            y_pred_all.append(1 if title in recommended_titles else 0)

    if len(y_true_all) == 0:
        return None

    precision = precision_score(y_true_all, y_pred_all, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "instances": len(y_true_all)
    }


# -----------------------------
# 6. QUESTIONNAIRE / USER SATISFACTION
# -----------------------------
SURVEY_FILE = "user_satisfaction_responses.csv"

def save_questionnaire_response(data):
    row = pd.DataFrame([data])
    try:
        old = pd.read_csv(SURVEY_FILE)
        combined = pd.concat([old, row], ignore_index=True)
    except FileNotFoundError:
        combined = row
    combined.to_csv(SURVEY_FILE, index=False)


def load_questionnaire_data():
    try:
        return pd.read_csv(SURVEY_FILE)
    except FileNotFoundError:
        return pd.DataFrame()


# -----------------------------
# 7. PAGE LAYOUT
# -----------------------------
st.sidebar.title("🎬 Discovery Menu")
page = st.sidebar.radio(
    "Select Interface",
    [
        "Smart Recommendations",
        "Advanced Search",
        "Model Deep-Dive",
        "Evaluation Metrics",
        "User Satisfaction Survey",
        "System Features"
    ]
)

# -----------------------------
# PAGE: SMART RECOMMENDATIONS
# -----------------------------
if page == "Smart Recommendations":
    st.header("🎯 Personalized Movie Matches")

    mode = st.selectbox(
        "Recommender Engine",
        ["Hybrid", "Content-Based", "Collaborative", "Explore", "New User"]
    )

    movie_title = None
    if mode != "New User":
        movie_title = st.selectbox("Pick a movie you love:", movies_merged['title'].sort_values().unique())

    c1, c2, c3 = st.columns(3)

    mood = c1.selectbox(
        "Your Current Vibe",
        ["None", "Happy", "Sad", "Romantic", "Excited", "Curious", "Scared", "Relaxed"]
    )

    pers = c2.selectbox(
        "Your Personality",
        ["None", "Adventurer", "Romantic", "Thinker", "Fun Lover", "Dreamer", "Bold Explorer"]
    )

    top_n = c3.slider("Results per page", 1, 15, 5)

    hidden_gems = st.checkbox("Focus on Hidden Gems (Underrated movies)")
    alpha = 0.5

    if mode in ["Hybrid", "Explore"]:
        alpha = st.slider("Hybrid Alpha (content vs collaborative)", 0.0, 1.0, 0.5, 0.1)

    if st.button("Generate Experience"):
        results = get_recommendations(
            movie_title=movie_title,
            mode=mode,
            top_n=top_n,
            mood=mood,
            hidden_gems=hidden_gems,
            personality=pers,
            alpha=alpha
        )

        if not results.empty:
            st.success(f"Algorithm applied: {mode}")

            st.dataframe(
                results[['title', 'genres_clean', 'vote_average', 'vote_count', 'release_date']].head(top_n),
                use_container_width=True
            )

            for _, row in results.iterrows():
                with st.expander(f"🎥 {row['title']} (Rating: {row['vote_average']})"):
                    st.write(f"**Genres:** {row['genres_clean']}")
                    st.write(f"**Release Date:** {row['release_date']}")
                    st.write(f"**Votes:** {row['vote_count']}")
                    if 'model_score' in row:
                        try:
                            st.write(f"**Recommendation Score:** {float(row['model_score']):.4f}")
                        except:
                            st.write(f"**Recommendation Score:** {row['model_score']}")
                    st.write(f"**Overview:** {row['overview']}")
                    st.info(f"Why recommended: {row['explanation']}")
        else:
            st.error("No recommendations found. Try changing the movie, mood, or personality filter.")

# -----------------------------
# PAGE: ADVANCED SEARCH
# -----------------------------
elif page == "Advanced Search":
    st.header("🔍 Semantic Search & Discovery")
    query = st.text_input("Search titles or themes (Typos allowed!)")

    if query:
        search_results = fuzzy_search_movies(query, limit=20, score_cutoff=60)

        if not search_results.empty:
            st.write(f"Found {len(search_results)} relevant matches.")
            col_left, col_right = st.columns([2, 1])

            with col_left:
                st.dataframe(
                    search_results[['title', 'genres_clean', 'vote_average', 'release_date']],
                    use_container_width=True
                )

            with col_right:
                st.write("**Rating Spread**")
                st.bar_chart(search_results.set_index('title')['vote_average'])
        else:
            st.warning("No matches found. Try another keyword.")

# -----------------------------
# PAGE: MODEL DEEP-DIVE
# -----------------------------
elif page == "Model Deep-Dive":
    st.header("📊 Deep-Dive Model Comparison")

    comp_movie = st.selectbox("Select Movie to Analyze", movies_merged['title'].sort_values().unique())

    if st.button("Run Model Comparison"):
        c_recs = recommend_content(comp_movie, 10).assign(Model='Content')
        cl_recs = recommend_collaborative(comp_movie, 10).assign(Model='Collaborative')
        h_recs = hybrid_recommend(comp_movie, 10).assign(Model='Hybrid')

        if c_recs.empty and cl_recs.empty and h_recs.empty:
            st.error("No comparison results available for this movie.")
        else:
            all_results = pd.concat([c_recs, cl_recs, h_recs], ignore_index=True)

            fig = px.box(
                all_results,
                x="Model",
                y="vote_average",
                color="Model",
                title="Movie Ratings Distribution by Model Selection"
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**1. Content-Based**")
                st.table(c_recs[['title']].head(5) if not c_recs.empty else pd.DataFrame({"title": ["No results"]}))
            with col2:
                st.write("**2. Collaborative**")
                st.table(cl_recs[['title']].head(5) if not cl_recs.empty else pd.DataFrame({"title": ["No results"]}))
            with col3:
                st.write("**3. Hybrid (AI Choice)**")
                st.table(h_recs[['title']].head(5) if not h_recs.empty else pd.DataFrame({"title": ["No results"]}))

            overlap = set(c_recs['title']) & set(cl_recs['title'])
            if overlap:
                st.success(f"Models agree on these high-confidence matches: {', '.join(list(overlap)[:10])}")
            else:
                st.info("Models are diverse. Hybrid produces different recommendations from single methods.")

# -----------------------------
# PAGE: EVALUATION METRICS
# -----------------------------
elif page == "Evaluation Metrics":
    st.subheader("Rating Prediction Metrics")

    sample_size = st.slider("Sample Size for MSE / RMSE", 100, 500, 300, 50)

    if st.button("Run MSE / RMSE Evaluation"):
        rating_metrics = evaluate_rating_prediction(sample_size=sample_size)

        if rating_metrics is None:
            st.error("Unable to compute MSE / RMSE.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Test Cases", f"{rating_metrics['count']}")
            c2.metric("MSE", f"{rating_metrics['mse']:.4f}")
            c3.metric("RMSE", f"{rating_metrics['rmse']:.4f}")

            plot_df = pd.DataFrame({
                "Actual Rating": rating_metrics['actuals'],
                "Predicted Rating": rating_metrics['preds']
            })

            fig = px.scatter(
                plot_df,
                x="Actual Rating",
                y="Predicted Rating",
                title="Actual vs Predicted Ratings"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Note: These metrics are approximate offline evaluations. "
        "They help assess recommendation accuracy and prediction quality for the academic project."
    )

# -----------------------------
# PAGE: USER SATISFACTION SURVEY
# -----------------------------
elif page == "User Satisfaction Survey":
    st.header("📝 User Satisfaction Questionnaire")
    st.write("Please answer the questionnaire after trying the recommender system.")

    with st.form("survey_form"):
        user_name = st.text_input("Name / Tester ID")
        used_mode = st.selectbox(
            "Which recommender mode did you test most?",
            ["Hybrid", "Content-Based", "Collaborative", "Explore", "New User"]
        )

        ease_of_use = st.slider("1. The system is easy to use", 1, 5, 4)
        recommendation_quality = st.slider("2. The recommendations match my interests", 1, 5, 4)
        explanation_clarity = st.slider("3. The recommendation explanations are understandable", 1, 5, 4)
        diversity_score = st.slider("4. The system provides diverse movie choices", 1, 5, 4)
        overall_satisfaction = st.slider("5. Overall, I am satisfied with the system", 1, 5, 4)

        comment = st.text_area("Additional Comments")

        submitted = st.form_submit_button("Submit Questionnaire")

    if submitted:
        response = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_name": user_name if user_name.strip() else "Anonymous",
            "used_mode": used_mode,
            "ease_of_use": ease_of_use,
            "recommendation_quality": recommendation_quality,
            "explanation_clarity": explanation_clarity,
            "diversity_score": diversity_score,
            "overall_satisfaction": overall_satisfaction,
            "comment": comment
        }
        save_questionnaire_response(response)
        st.success("Thank you. Your questionnaire response has been saved.")

    st.subheader("Survey Summary")
    survey_df = load_questionnaire_data()

    if not survey_df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Responses", len(survey_df))
        c2.metric("Avg Satisfaction", f"{survey_df['overall_satisfaction'].mean():.2f}/5")
        c3.metric("Avg Quality", f"{survey_df['recommendation_quality'].mean():.2f}/5")

        avg_scores = pd.DataFrame({
            "Question": [
                "Ease of Use",
                "Recommendation Quality",
                "Explanation Clarity",
                "Diversity",
                "Overall Satisfaction"
            ],
            "Average Score": [
                survey_df['ease_of_use'].mean(),
                survey_df['recommendation_quality'].mean(),
                survey_df['explanation_clarity'].mean(),
                survey_df['diversity_score'].mean(),
                survey_df['overall_satisfaction'].mean()
            ]
        })

        fig = px.bar(avg_scores, x="Question", y="Average Score", title="Average User Satisfaction Scores")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(survey_df, use_container_width=True)
    else:
        st.info("No questionnaire responses yet.")

# -----------------------------
# PAGE: SYSTEM FEATURES
# -----------------------------
elif page == "System Features":
    st.header("✨ Comprehensive System Features")

    features = {
        "Content-Based Recommendation": "Uses TF-IDF and cosine similarity based on overview and genres.",
        "Collaborative Filtering": "Uses user-movie rating behaviour to find similar movies.",
        "Hybrid Recommendation": "Combines content and collaborative scores using alpha weighting.",
        "Explore Mode": "Suggests less obvious but still relevant recommendations.",
        "New User Recommendation": "Provides results even when no movie history is available.",
        "Explainable Recommendation": "Shows why a movie was suggested to the user.",
        "Advanced Search": "Uses fuzzy search to handle typos and partial movie titles.",
        "Mood / Personality Filtering": "Provides context-aware filtering based on user preferences.",
        "Hidden Gems Logic": "Highlights good movies that are less popular.",
        "Model Deep-Dive": "Compares model outputs with visual analysis.",
        "Evaluation Metrics": "Calculates Precision, Recall, F1, MSE, and RMSE for academic assessment.",
        "User Satisfaction Questionnaire": "Collects user feedback and summarizes satisfaction scores.",
        "Caching System": "Uses Streamlit caching to improve performance."
    }

    for feat, desc in features.items():
        st.write(f"🔹 **{feat}**: {desc}")

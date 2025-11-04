import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="BioBeat Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1DB954;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .song-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1DB954;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# LOAD MODELS AND DATA
# ============================================================
@st.cache_resource
def load_models_and_data():
    """Load all models and processed data"""
    try:
        # Load models
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('pca.pkl', 'rb') as f:
            pca = pickle.load(f)
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)
        with open('dbscan_model.pkl', 'rb') as f:
            dbscan_model = pickle.load(f)
        # with open('hierarchical_model.pkl', 'rb') as f:
        #     hierarchical_model = pickle.load(f)
        
        # Load data
        df = pd.read_csv('processed_music_data.csv')
        X_pca = np.load('X_pca.npy')
        
        return scaler, pca, kmeans_model, dbscan_model, df, X_pca
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        st.error("Please run the training script first to generate the required model files!")
        return None, None, None, None, None, None, None

# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================
def get_recommendations(song_name, model_type, df, X_pca, n_recommendations=10):
    """
    Get song recommendations based on DNA pattern similarity
    """
    # Find the selected song
    song_idx = df[df['track_name'] == song_name].index
    
    if len(song_idx) == 0:
        return None, "Song not found in database"
    
    song_idx = song_idx[0]
    
    # Get cluster label based on model type
    if model_type == 'K-Means':
        cluster_col = 'kmeans_cluster'
    elif model_type == 'DBSCAN':
        cluster_col = 'dbscan_cluster'
    else:  # Hierarchical
        cluster_col = 'hierarchical_cluster'
    
    song_cluster = df.iloc[song_idx][cluster_col]
    
    # Get songs from the same cluster
    cluster_songs = df[df[cluster_col] == song_cluster].copy()
    cluster_indices = cluster_songs.index.tolist()
    
    # Calculate cosine similarity within the cluster
    song_features = X_pca[song_idx].reshape(1, -1)
    cluster_features = X_pca[cluster_indices]
    
    similarities = cosine_similarity(song_features, cluster_features)[0]
    
    # Add similarity scores to cluster_songs
    cluster_songs['similarity'] = similarities
    
    # Remove the input song itself
    cluster_songs = cluster_songs[cluster_songs.index != song_idx]
    
    # Sort by similarity and get top N
    recommendations = cluster_songs.nlargest(n_recommendations, 'similarity')
    
    return recommendations, None

# ============================================================
# MAIN APP
# ============================================================
def main():
    # Header
    st.markdown('<p class="main-header">🎵 BioBeat Music Recommender</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">DNA Pattern Analysis for Personalized Music Discovery</p>', unsafe_allow_html=True)
    # st.markdown('<p class="sub-header">Please Use In Light Theme for Good UI</p>', unsafe_allow_html=True)
    
    # Load models and data
    scaler, pca, kmeans_model, dbscan_model, df, X_pca = load_models_and_data()
    
    if df is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/music.png", width=100)
        st.title("🎼 Controls")
        st.markdown("---")
        
        # Model selection
        st.subheader("Select Clustering Model")
        model_type = st.selectbox(
            "Choose Model:",
            ['K-Means', 'DBSCAN', 'Hierarchical'],
            help="Different clustering algorithms for finding similar songs"
        )
        
        # Number of recommendations
        n_recommendations = st.slider(
            "Number of Recommendations:",
            min_value=5,
            max_value=20,
            value=10,
            help="How many similar songs to recommend"
        )
        
        st.markdown("---")
        
        # Model info
        st.subheader("📊 Model Info")
        if model_type == 'K-Means':
            st.info("**K-Means**: Groups songs into distinct clusters based on audio features")
        elif model_type == 'DBSCAN':
            st.info("**DBSCAN**: Finds dense regions of similar songs, robust to outliers")
        else:
            st.info("**Hierarchical**: Creates a tree-like structure of song similarities")
        
        st.markdown("---")
        st.subheader("📈 Dataset Stats")
        st.metric("Total Songs", len(df))
        st.metric("Audio Features", "15 DNA patterns")
        
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Search for a Song")
        
        # Song search with autocomplete
        search_term = st.text_input(
            "Type to search:",
            placeholder="Enter song name...",
            help="Start typing to find a song"
        )
        
        # Filter songs based on search
        if search_term:
            filtered_songs = df[df['track_name'].str.contains(search_term, case=False, na=False)]
            song_options = filtered_songs['track_name'].tolist()
        else:
            song_options = df['track_name'].tolist()
        
        selected_song = st.selectbox(
            "Select a song:",
            song_options,
            help="Choose the song you want recommendations for"
        )
    
    with col2:
        st.subheader("🎤 Song Details")
        if selected_song:
            song_info = df[df['track_name'] == selected_song].iloc[0]
            st.markdown(f"""
            <div class="song-card">
                <strong>🎵 Track:</strong> {song_info['track_name']}<br>
                <strong>👤 Artist:</strong> {song_info['artist_name']}<br>
                <strong>📅 Year:</strong> {song_info['year']}<br>
                <strong>⭐ Popularity:</strong> {song_info['popularity']}
            </div>
            """, unsafe_allow_html=True)
    
    # Get recommendations button
    st.markdown("---")
    
    if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Analyzing DNA patterns and finding similar songs..."):
            recommendations, error = get_recommendations(
                selected_song, 
                model_type, 
                df, 
                X_pca, 
                n_recommendations
            )
            
            if error:
                st.error(error)
            else:
                st.success(f"Found {len(recommendations)} similar songs using {model_type}!")
                
                # Display recommendations
                st.subheader(f"🎼 Top {n_recommendations} Recommendations")
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["📋 List View", "📊 Chart View", "🧬 DNA Pattern"])
                
                with tab1:
                    # Display as cards
                    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                        col_a, col_b, col_c = st.columns([1, 3, 1])
                        
                        with col_a:
                            st.markdown(f"### #{idx}")
                        
                        with col_b:
                            st.markdown(f"""
                            <div class="song-card">
                                <strong>🎵 {row['track_name']}</strong><br>
                                <strong>👤 Artist:</strong> {row['artist_name']}<br>
                                <strong>📀 Album:</strong> {row['album_name']}<br>
                                <strong>📅 Year:</strong> {row['year']} | 
                                <strong>⭐ Popularity:</strong> {row['popularity']}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_c:
                            similarity_pct = row['similarity'] * 100
                            st.metric("Match", f"{similarity_pct:.1f}%")
                
                with tab2:
                    # Similarity chart
                    fig = px.bar(
                        recommendations.head(n_recommendations),
                        x='similarity',
                        y='track_name',
                        orientation='h',
                        title=f'Top {n_recommendations} Songs by Similarity Score',
                        labels={'similarity': 'Similarity Score', 'track_name': 'Song'},
                        color='similarity',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Audio features comparison
                    audio_cols = ['acousticness', 'danceability', 'energy', 'valence', 'speechiness']
                    
                    selected_song_features = df[df['track_name'] == selected_song][audio_cols].values[0]
                    top_rec_features = recommendations.iloc[0][audio_cols].values
                    
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatterpolar(
                        r=selected_song_features,
                        theta=audio_cols,
                        fill='toself',
                        name='Selected Song'
                    ))
                    fig2.add_trace(go.Scatterpolar(
                        r=top_rec_features,
                        theta=audio_cols,
                        fill='toself',
                        name='Top Recommendation'
                    ))
                    fig2.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=True,
                        title="Audio Feature Comparison"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                with tab3:
                    # DNA Pattern visualization
                    st.markdown("### 🧬 Audio DNA Pattern Analysis")
                    
                    dna_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                                    'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
                    
                    # Create a heatmap of top 5 recommendations
                    top_5 = recommendations.head(5)
                    heatmap_data = top_5[dna_features].values
                    
                    fig3 = px.imshow(
                        heatmap_data,
                        labels=dict(x="Audio Features", y="Recommended Songs", color="Value"),
                        x=dna_features,
                        y=top_5['track_name'].tolist(),
                        aspect="auto",
                        color_continuous_scale='RdYlGn'
                    )
                    fig3.update_layout(title="DNA Pattern Heatmap - Top 5 Recommendations")
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    st.info("**Interpretation**: Similar colors indicate similar audio characteristics. Songs with similar DNA patterns sound alike!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p> Use Light Theme for better View</p>
        <p>🎵 Powered by Machine Learning & Audio DNA Analysis | BioBeat Recommender System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
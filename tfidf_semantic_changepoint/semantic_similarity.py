"""
Semantic similarity analysis using embeddings
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import cos_sim


def tokens_to_text(x):
    """
    Convert token list to string for encoding.
    
    Args:
        x: Token list or other value
        
    Returns:
        String representation
    """
    if isinstance(x, list):
        return " ".join(x)
    elif pd.isna(x):
        return ""
    else:
        return str(x)


def compute_embeddings(df, text_column="tokenized_abstract", model_name="sentence-transformers/all-mpnet-base-v2"):
    """
    Compute embeddings for text data using sentence transformers.
    
    Args:
        df: DataFrame with text data
        text_column: Name of column containing tokenized text
        model_name: Name of sentence transformer model
        
    Returns:
        DataFrame with added 'embedding' column
    """
    model = SentenceTransformer(model_name)
    
    df = df.copy()
    df["string_tokenized_abstract"] = df[text_column].apply(tokens_to_text)
    
    embeddings = model.encode(
        df["string_tokenized_abstract"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True
    )
    
    df["embedding"] = list(embeddings)
    return df


def division_year_centroid(df, division):
    """
    Compute mean embedding centroid for each year within a division.
    
    Args:
        df: DataFrame with embeddings
        division: Division name to filter
        
    Returns:
        Series with year as index and centroid embeddings as values
    """
    sub = df[df["division_name"] == division]
    return sub.groupby("year")["embedding"].apply(lambda x: np.stack(x).mean(axis=0))


def compute_similarity_matrix(centroids):
    """
    Compute pairwise cosine similarity matrix for year centroids.
    
    Args:
        centroids: Dictionary mapping year to embedding vector
        
    Returns:
        Similarity matrix as numpy array
    """
    years = sorted(centroids.keys())
    M = np.zeros((len(years), len(years)))
    
    for i, y1 in enumerate(years):
        for j, y2 in enumerate(years):
            M[i, j] = cos_sim(centroids[y1], centroids[y2]).item()
    
    return M, years


def plot_similarity_heatmap(df, division, save_path=None):
    """
    Plot semantic similarity heatmap for a division across years.
    
    Args:
        df: DataFrame with embeddings
        division: Division name
        save_path: Optional path to save plot
    """
    div_df = df[df["division_name"] == division]
    
    centroids = (
        div_df.groupby("year")["embedding"]
              .apply(lambda x: np.mean(np.stack(x.values), axis=0))
              .to_dict()
    )
    
    years = sorted(centroids.keys())
    if len(years) < 2:
        print(f"Skipping {division}: only {len(years)} year(s)")
        return
    
    M, years = compute_similarity_matrix(centroids)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(M, xticklabels=years, yticklabels=years, annot=True, fmt='.3f')
    plt.title(f"Semantic similarity across years – {division}")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def compute_semantic_drift(df, baseline_years=None, target_year=2025):
    """
    Compute semantic drift relative to baseline years.
    
    Args:
        df: DataFrame with embeddings
        baseline_years: Years to use as baseline (default: all years before target_year)
        target_year: Year to compare against baseline
        
    Returns:
        DataFrame with year and drift values
    """
    df = df.copy()
    df["embedding_array"] = df["embedding"].apply(lambda x: np.array(x))
    
    year_means = (
        df.groupby("year")["embedding_array"]
          .apply(lambda xs: np.vstack(xs).mean(axis=0))
    )
    
    if baseline_years is None:
        pre_mask = year_means.index < target_year
        baseline_vec = np.vstack(year_means[pre_mask].values).mean(axis=0)
    else:
        baseline_vec = np.vstack(year_means[baseline_years].values).mean(axis=0)
    
    def cosine_dist(a, b):
        return cosine(a, b)
    
    year_drift = year_means.apply(lambda v: cosine_dist(v, baseline_vec))
    
    drift_df = pd.DataFrame({
        "year": year_drift.index.values,
        "drift": year_drift.values,
    }).sort_values("year")
    
    return drift_df


def get_eligible_divisions(df, min_years=2, min_embeddings=20):
    """
    Get divisions that meet minimum requirements for analysis.
    
    Args:
        df: DataFrame with embeddings
        min_years: Minimum number of distinct years required
        min_embeddings: Minimum number of embeddings required
        
    Returns:
        List of eligible division names
    """
    years_per_div = df.groupby("division_name")["year"].nunique()
    embeddings_per_div = df.groupby("division_name")["embedding"].size()
    
    mask = (years_per_div >= min_years) & (embeddings_per_div >= min_embeddings)
    eligible_divisions = years_per_div.index[mask].tolist()
    
    return eligible_divisions


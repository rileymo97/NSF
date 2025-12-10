"""
TF-IDF analysis functions for DEI-related keywords
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


def get_dei_keywords():
    """
    Returns list of DEI-related keywords for TF-IDF analysis.
    
    Returns:
        List of keyword strings
    """
    return [
        "activism", "activists", "advocacy", "advocate",
        "barrier", "barriers", "biased", "bias", "bipox",
        "black", "latino", "latinx", "community",
        "diversity", "equity", "cultural", "disabilities",
        "discrimination", "discriminatory", "backgrounds",
        "diversified", "diversify", "equal", "equality",
        "equitable", "ethnicity", "excluded", "female",
        "fostering", "gender", "hate", "hispanic",
        "historically", "implicit bias", "inclusion",
        "inclusive", "indigenous", "inequalities",
        "inequities", "institutional", "lgbtq",
        "marginalize", "minorities", "multicultural",
        "polarization", "political", "prejudice",
        "privileges", "promoting", "race", "racial",
        "justice", "sociocultural", "socioeconomic",
        "status", "stereotypes", "systemic", "trauma",
        "underappreciated", "underrepresented",
        "underserved", "victim", "women",
    ]


def compute_tfidf_by_year(df, text_column="clean_text", keywords=None):
    """
    Compute TF-IDF scores for keywords by year.
    
    Args:
        df: DataFrame with text data and year column
        text_column: Name of column containing text
        keywords: List of keywords to analyze (defaults to DEI keywords)
        
    Returns:
        DataFrame with yearly mean TF-IDF scores for each keyword
    """
    if keywords is None:
        keywords = get_dei_keywords()
    
    vectorizer = TfidfVectorizer(
        vocabulary=keywords,
        lowercase=True,
        ngram_range=(1, 3),
    )
    
    tfidf_matrix = vectorizer.fit_transform(df[text_column])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    
    tfidf_df["year"] = df["year"].values
    
    yearly_tfidf = (
        tfidf_df.groupby("year")[vectorizer.get_feature_names_out()].mean()
    )
    
    return yearly_tfidf


def compute_dei_index(yearly_tfidf):
    """
    Compute DEI index as mean TF-IDF across all keywords.
    
    Args:
        yearly_tfidf: DataFrame with yearly TF-IDF scores
        
    Returns:
        Series with DEI index by year
    """
    yearly_tfidf["dei_index"] = yearly_tfidf.mean(axis=1)
    return yearly_tfidf["dei_index"]


def plot_tfidf_trends(yearly_tfidf, reference_year=2025, save_path=None):
    """
    Plot TF-IDF trends for all keywords over time.
    
    Args:
        yearly_tfidf: DataFrame with yearly TF-IDF scores
        reference_year: Year to use for ordering keywords
        save_path: Optional path to save plot
    """
    word_cols = [w for w in yearly_tfidf.columns if w != "dei_index"]
    
    ordered_words = sorted(
        word_cols,
        key=lambda w: yearly_tfidf.loc[reference_year, w],
        reverse=True
    )
    
    plt.figure(figsize=(10, 10))
    years = yearly_tfidf.index
    
    for word in ordered_words:
        plt.plot(years, yearly_tfidf[word], label=word)
    
    plt.title("TF-IDF Prevalence of Selected Words by Year")
    plt.xlabel("Year")
    plt.ylabel("Mean TF-IDF")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def plot_dei_index(yearly_tfidf, save_path=None):
    """
    Plot DEI index over time.
    
    Args:
        yearly_tfidf: DataFrame with yearly TF-IDF scores (must include dei_index)
        save_path: Optional path to save plot
    """
    if "dei_index" not in yearly_tfidf.columns:
        yearly_tfidf = yearly_tfidf.copy()
        yearly_tfidf["dei_index"] = compute_dei_index(yearly_tfidf)
    
    plt.figure(figsize=(10, 5))
    plt.plot(yearly_tfidf.index, yearly_tfidf["dei_index"], marker="o")
    plt.title("DEI Index (Mean TF-IDF of DEI Terms by Year)")
    plt.xlabel("Year")
    plt.ylabel("Mean TF-IDF")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


"""
Main pipeline for TF-IDF, semantic similarity, and changepoint analysis
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tfidf_semantic_changepoint import tfidf_analysis
from tfidf_semantic_changepoint import semantic_similarity
from tfidf_semantic_changepoint import changepoint_detection


def load_data(data_path):
    """
    Load grant data from JSON file.
    
    Args:
        data_path: Path to grants_df.json file
        
    Returns:
        DataFrame with grant data
    """
    with open(data_path, "r") as f:
        grants_raw = json.load(f)
    df = pd.DataFrame(grants_raw)
    return df


def main(data_path="data/grants_df.json", output_dir="data/tfidf_semantic_analysis"):
    """
    Main pipeline for TF-IDF, semantic similarity, and changepoint analysis.
    
    Args:
        data_path: Path to grants_df.json file
        output_dir: Directory to save outputs
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df = load_data(data_path)
    
    # Check if embeddings exist, compute if not
    if "embedding" not in df.columns or df["embedding"].isna().any():
        print("Computing embeddings...")
        df = semantic_similarity.compute_embeddings(df)
        df.to_json(data_path)
    
    # Convert embeddings to arrays if needed
    if isinstance(df["embedding"].iloc[0], list):
        df["embedding"] = df["embedding"].apply(lambda x: np.array(x))
    
    print("\n=== TF-IDF Analysis ===")
    yearly_tfidf = tfidf_analysis.compute_tfidf_by_year(df)
    yearly_tfidf["dei_index"] = tfidf_analysis.compute_dei_index(yearly_tfidf)
    
    tfidf_analysis.plot_tfidf_trends(
        yearly_tfidf,
        save_path=str(output_path / "tfidf_trends.png")
    )
    tfidf_analysis.plot_dei_index(
        yearly_tfidf,
        save_path=str(output_path / "dei_index.png")
    )
    
    yearly_tfidf.to_csv(output_path / "yearly_tfidf.csv")
    print(f"TF-IDF analysis saved to {output_path}")
    
    print("\n=== Semantic Similarity Analysis ===")
    eligible_divisions = semantic_similarity.get_eligible_divisions(df)
    print(f"Found {len(eligible_divisions)} eligible divisions")
    
    for division in eligible_divisions[:5]:  # Limit to first 5 for demo
        semantic_similarity.plot_similarity_heatmap(
            df,
            division,
            save_path=str(output_path / f"similarity_{division.replace(' ', '_')}.png")
        )
    
    print("\n=== Changepoint Detection ===")
    drift_df = semantic_similarity.compute_semantic_drift(df)
    cp_trace = changepoint_detection.fit_changepoint_model(drift_df)
    
    changepoint_detection.plot_changepoint_posterior(
        cp_trace,
        save_path=str(output_path / "changepoint_posterior.png")
    )
    
    drift_df.to_csv(output_path / "semantic_drift.csv", index=False)
    print(f"Changepoint analysis saved to {output_path}")
    
    # Analyze specific divisions
    target_divisions = [
        "EPSCoR",
        "Physics",
        "Social and Economic Sciences",
        "Chemical, Bioengineering, Environmental, and Transport Systems",
        "Polar Programs",
    ]
    
    print("\n=== Division-Specific Changepoint Analysis ===")
    drift_df_multi, cp_trace_multi = changepoint_detection.analyze_multiple_divisions(
        df,
        target_divisions,
        save_dir=str(output_path / "multi_division_changepoint")
    )
    
    drift_df_polar, cp_trace_polar = changepoint_detection.analyze_division_changepoint(
        df,
        "Polar Programs",
        save_path=str(output_path / "polar_programs_changepoint.png")
    )
    
    print(f"\nComplete! Results saved to {output_path}")
    
    return df, yearly_tfidf, drift_df, cp_trace


if __name__ == "__main__":
    main()


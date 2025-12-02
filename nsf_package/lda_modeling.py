"""
LDA modeling functions
"""

import pandas as pd
import gensim.corpora as corpora
from gensim.models import LdaModel


def train_division_models(grants_df, num_topics=10, alpha=0.1, eta=0.05):
    """
    Train separate LDA models for each division (Cell 8)
    """
    # Dictionary to store models and metadata for each division
    division_models = {}
    
    for division in grants_df['division_name'].unique():
        print(f"\nProcessing: {division}")
        
        # Filter dataframe for this division
        div_df = grants_df[grants_df['division_name'] == division].copy()
        
        # Filter out abstracts (or lack thereof) with no keyphrases
        div_df = div_df.dropna(subset=['keyphrases'])
        
        print(f"  Number of grants: {len(div_df)}")
        
        # Skip divisions with less than 50 grants
        if len(div_df) < 50:
            print(f"  Skipping {division}: only has {len(div_df)} total grants")
            continue
        
        # Get token lists for this division
        token_lists = div_df["keyphrases"].tolist()
        
        # Create dictionary for this division
        id2word = corpora.Dictionary(token_lists)
        
        # Filter extremes: remove words that appear in <2 documents or >50% of documents
        id2word.filter_extremes(no_below=2, no_above=0.5)
        print(f"  Corpus size: {len(id2word)}")
        
        if len(id2word) < 10:
            print(f"  Skipping {division}: insufficient vocabulary ({len(id2word)} < 10)")
            continue
        
        # Create corpus for this division
        corpus = [id2word.doc2bow(text) for text in token_lists]
        
        # Train LDA model
        print(f"  Training LDA model...")
        lda_model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=num_topics,
            alpha=alpha,
            eta=eta,
            random_state=100,
            passes=50,
            iterations=400,
            per_word_topics=True,
            eval_every=10
        )
        
        # Store model and metadata
        division_models[division] = {
            'model': lda_model,
            'id2word': id2word,
            'corpus': corpus,
            'dataframe': div_df,
            'num_grants': len(div_df),
            'vocab_size': len(id2word)
        }
        
        # Print topics for this division
        print(f"\n  Topics for {division}:")
        for i, topic in lda_model.print_topics(-1, num_words=10):
            print(f"    Topic {i+1}: {topic}")
    
    print(f"\n\nCompleted! Trained models for {len(division_models)} divisions.")
    
    return division_models


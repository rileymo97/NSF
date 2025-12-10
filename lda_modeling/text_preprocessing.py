"""
Text preprocessing functions
"""
import pandas as pd
import gensim
import os 

from pathlib import Path
import os
PROJECT_ROOT = Path(__file__).parent.parent
STOPWORDS_PATH = PROJECT_ROOT / os.getenv("STOPWORDS_PATH")

def load_stopwords(stopwords_file=STOPWORDS_PATH):
    """
    Load stopwords from stored file 
    """
    with open(stopwords_file, 'r') as f:
        words = []
        for line in f:
            words.append(line.strip())
    stop_words = set(words)
    return stop_words


def tokenize_abstracts(grants_df: pd.DataFrame) -> tuple[pd.DataFrame, 
                                                         gensim.models.phrases.Phraser, 
                                                         gensim.models.phrases.Phraser]:
    """
    Build bigram and trigram models and tokenize abstracts 
    Returns: (grants_df, bigram_mod, trigram_mod)
    """
    # Load stopwords
    stop_words = load_stopwords(STOPWORDS_PATH)
    
    # Loop through each row in the df, tokenize the abstract and store in the df
    grants_df["tokenized_abstract"] = ""
    data = []
    for i, row in grants_df.iterrows():
        tokens = [word for word in gensim.utils.simple_preprocess(str(row["abstract"]), deacc=True, min_len=3) 
                 if word not in stop_words]
        grants_df.at[i, 'tokenized_abstract'] = tokens
        
        # Also add to data list for model training below
        data.append(tokens)
    
    # Train models on tokenized data
    bigram = gensim.models.Phrases(data, min_count=20, threshold=100)
    trigram = gensim.models.Phrases(bigram[data], threshold=100)
    
    # Create phrasers
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    return grants_df, bigram_mod, trigram_mod


def process_words(row, bigram_mod, trigram_mod, nlp, allowed_tags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """
    Convert a document into a list of lowercase tokens, build bigrams-trigrams, implement lemmatization 
    """
    # Load stopwords
    stop_words = load_stopwords(STOPWORDS_PATH)
    
    tokens = row["tokenized_abstract"]
    
    # Apply bigram and trigram models to create phrases
    tokens = bigram_mod[tokens]
    tokens = trigram_mod[bigram_mod[tokens]]
    
    # Separate phrases (with underscores) from single words
    phrases = []
    single_words = []
    for token in tokens:
        if '_' in token:
            phrases.append(token)
        else:
            single_words.append(token)
    
    # Lemmatize single words only (to not remove underscores)
    if single_words:
        doc = nlp(" ".join(single_words))
        lemmatized = [token.lemma_ for token in doc if token.pos_ in allowed_tags]
    else:
        lemmatized = []
    
    # Combine phrases and lemmatized words, filter stopwords and short tokens
    result = phrases + lemmatized
    return [word for word in result if word not in stop_words and len(word) >= 3]


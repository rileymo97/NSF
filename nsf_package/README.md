# NSF Data Processing Package

This package provides modules for processing NSF grant data, including data loading, cleaning, text preprocessing, and LDA topic modeling.

## Package Structure

```
nsf_data/
├── __init__.py          # Package initialization and exports
├── constants.py         # Division names and constants
├── data_loading.py      # Load grants data and terminated awards
├── data_cleaning.py     # Clean data and add columns
├── text_preprocessing.py # Tokenization, bigrams, keyphrases
├── lda_modeling.py      # Train LDA models per division
├── export.py            # Export topics to Excel
└── models.py            # Model initialization (spaCy)
```

## Usage

```python
import nsf_data

# Load spaCy model
nlp = nsf_data.load_spacy_model()

# Load and clean data
grants_df = nsf_data.load_grants_data("data")
terminated_awards = nsf_data.load_terminated_awards("data")
grants_df = nsf_data.add_termination_flag(grants_df, terminated_awards)
grants_df = nsf_data.add_abstract_column(grants_df)
grants_df = nsf_data.add_division_names(grants_df)

# Preprocess text
stop_words = nsf_data.load_stopwords()
grants_df, data, bigram_mod, trigram_mod = nsf_data.tokenize_abstracts(grants_df, stop_words)
grants_df = nsf_data.add_keyphrases(grants_df, bigram_mod, trigram_mod, nlp, stop_words)

# Train LDA models
division_models = nsf_data.train_division_models(grants_df, num_topics=10, alpha=0.1, eta=0.05)

# Export to Excel
nsf_data.export_topics_to_excel(division_models, 'lda_topics_by_division.xlsx')
```

See `example_usage.py` in the parent directory for a complete example.


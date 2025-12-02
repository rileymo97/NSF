# NSF Grant Data Processing Package

A Python package for processing NSF grant data, performing text preprocessing, and training LDA topic models to discover research themes across NSF divisions.

## Installation

```bash
pip install -e .
python -m spacy download en_core_web_sm
```

Ensure data structure: `data/YYYY/*.json` and `data/NSF-Terminated-Awards.csv`

## Quick Start

### Complete Pipeline

```python
from nsf_package.main import main

grants_df, division_models, excel_file = main()
```

### Individual Modules

```python
from nsf_package import models, data_loading, data_cleaning, text_preprocessing, lda_modeling, export

# Load models and data
nlp = models.load_spacy_model('models/en_core_web_sm/en_core_web_sm-3.8.0')
grants_df = data_loading.load_grants_data("data")
terminated_awards = data_loading.load_terminated_awards("data")

# Clean data
grants_df = data_cleaning.add_termination_flag(grants_df, terminated_awards)
grants_df = data_cleaning.add_abstract_column(grants_df)
grants_df = data_cleaning.add_division_names(grants_df)

# Preprocess text
stop_words = text_preprocessing.load_stopwords('models/stopwords/english')
grants_df, data, bigram_mod, trigram_mod = text_preprocessing.tokenize_abstracts(grants_df, stop_words)
grants_df = text_preprocessing.add_keyphrases(grants_df, bigram_mod, trigram_mod, nlp, stop_words)

# Train LDA models
division_models = lda_modeling.train_division_models(grants_df, num_topics=10, alpha=0.1, eta=0.05)

# Export
export.export_topics_to_excel(division_models, 'lda_topics_by_division.xlsx')
```

## Modules

- **`models.py`**: Load spaCy models
- **`data_loading.py`**: Load grants data and terminated awards
- **`data_cleaning.py`**: Add termination flags, abstract columns, division names
- **`text_preprocessing.py`**: Tokenization, bigrams/trigrams, keyphrase extraction
- **`lda_modeling.py`**: Train division-specific LDA models (filters divisions with < 50 grants)
- **`export.py`**: Export topics to Excel

## Output

- Processed grants DataFrame with tokenized abstracts and keyphrases
- Dictionary of LDA models per division with metadata
- Excel file (`lda_topics_by_division.xlsx`) with top 20 words per topic

## Default Parameters

- LDA: `num_topics=10`, `alpha=0.1`, `eta=0.05`, `passes=50`, `iterations=400`
- Paths: spaCy model at `models/en_core_web_sm/en_core_web_sm-3.8.0`, stopwords at `models/stopwords/english`

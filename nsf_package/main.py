"""
Main workflow - replicates the notebook cells
"""

import pandas as pd
import spacy
from . import constants
from . import data_loading
from . import data_cleaning
from . import text_preprocessing
from . import lda_modeling
from . import export
from . import models


def main():
    """
    Main workflow function that replicates the notebook
    """
    # Cell 0: Load spaCy model
    nlp = models.load_spacy_model('models/en_core_web_sm/en_core_web_sm-3.8.0')
    
    # Cell 1: Load grants data
    data_path = "data"
    grants_df = data_loading.load_grants_data(data_path)
    
    # Cell 2: Load terminated awards
    terminated_awards = data_loading.load_terminated_awards(data_path)
    
    # Cell 3: Add termination flag
    grants_df = data_cleaning.add_termination_flag(grants_df, terminated_awards)
    
    # Cell 4: Add abstract column
    grants_df = data_cleaning.add_abstract_column(grants_df)
    
    # Cell 5: Add division names
    grants_df = data_cleaning.add_division_names(grants_df)
    
    # Cell 6: Tokenize abstracts and build phrase models
    stop_words = text_preprocessing.load_stopwords('models/stopwords/english')
    grants_df, data, bigram_mod, trigram_mod = text_preprocessing.tokenize_abstracts(grants_df, stop_words)
    
    # Cell 7: Add keyphrases
    grants_df = text_preprocessing.add_keyphrases(grants_df, bigram_mod, trigram_mod, nlp, stop_words)
    
    # Cell 8: Train LDA models
    num_topics = 10
    alpha = 0.1
    eta = 0.05
    division_models = lda_modeling.train_division_models(grants_df, num_topics=num_topics, alpha=alpha, eta=eta)
    
    # Cell 9: Export to Excel
    excel_file = export.export_topics_to_excel(division_models, 'lda_topics_by_division.xlsx')
    
    print(f"\nComplete! Results saved to: {excel_file}")
    
    return grants_df, division_models, excel_file


if __name__ == "__main__":
    grants_df, division_models, excel_file = main()


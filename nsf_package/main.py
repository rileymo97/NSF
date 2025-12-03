"""
Main workflow - replicates the notebook cells
"""
import spacy
import os
import sys
from pathlib import Path

# Handle imports for both package and direct execution
try:
    from . import data_loading, data_cleaning, text_preprocessing, lda_modeling, export
except ImportError:
    # If running directly, add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    import data_loading, data_cleaning, text_preprocessing, lda_modeling, export 

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
NLP_PATH = PROJECT_ROOT / os.getenv("NLP_PATH", "models/en_core_web_sm")

def main():
    # Load data
    print("Loading data...")
    grants_df = data_loading.load_grants_data()
    terminated_awards = data_loading.load_terminated_awards()
    print(f"Loaded {len(grants_df)} grants and {len(terminated_awards)} terminated awards")
    
    # Add flag describing whether grant was terminated or not
    print("Adding termination flags...")
    grants_df = data_cleaning.add_termination_flag(grants_df, terminated_awards)
    
    # Add abstract column
    print("Creating abstract column...")
    grants_df['abstract'] = grants_df.apply(data_cleaning.add_abstract_column, axis=1)
    
    # Add division names
    print("Adding division names...")
    grants_df = data_cleaning.add_division_names(grants_df)
    
    # Tokenize abstracts and build phrase models
    print("Tokenizing abstracts and building phrase models...")
    grants_df, bigram_mod, trigram_mod = text_preprocessing.tokenize_abstracts(grants_df)
    
    # Add keyphrases
    print("Extracting keyphrases...")
    nlp = spacy.load(NLP_PATH, disable=['parser', 'ner']) # don't need parser or ner because we're only using lemmatization
    grants_df["keyphrases"] = grants_df.apply(lambda row: text_preprocessing.process_words(row, bigram_mod, trigram_mod, nlp), axis=1)
    
    # Save grants df to json file
    print("Saving grants dataframe to JSON...")
    grants_df.to_json(data_loading.DATA_PATH / 'grants_df.json')
    
    # Train LDA models
    print("Training LDA models for each division...")
    num_topics = 10
    alpha = 0.1
    eta = 0.05
    division_models = lda_modeling.train_division_lda_models(grants_df, num_topics=num_topics, alpha=alpha, eta=eta)
    
    # Export topics to Excel
    print("Exporting topics to Excel...")
    excel_file = export.export_topics_to_excel(division_models, str(data_loading.DATA_PATH / 'lda_topics_by_division.xlsx'))
    
    # Export document-topic assignments to CSV
    print("Exporting document-topic assignments to CSV and JSON...")
    doc_assignments_df = export.export_document_topic_assignments(division_models, str(data_loading.DATA_PATH))
    
    print(f"\nComplete! Results saved to:")
    
    return grants_df, division_models, excel_file


if __name__ == "__main__":
    grants_df, division_models, excel_file = main()


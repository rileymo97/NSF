"""
Export functions
"""

import pandas as pd
from pathlib import Path


def export_topics_to_excel(division_models, excel_file='lda_topics_by_division.xlsx'):
    """
    Export topics to Excel file with one sheet per division
    """
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        
        for division, model_data in division_models.items():
            lda_model = model_data['model']
            
            # Extract topics and their words
            topics_data = []
            
            # Get all topics for this division
            for topic_id in range(lda_model.num_topics):
                
                # Get topic words 
                topic_words = lda_model.show_topic(topic_id, topn=20)  # Get top 20 words
                
                # Create a row for this topic
                topic_row = {'Topic': f'Topic {topic_id + 1}'}
                
                # Add topics 
                for rank, (token, _) in enumerate(topic_words):
                    topic_row[f'Word {rank+1}'] = token
                
                topics_data.append(topic_row)
            
            # Create DataFrame for this division and write to Excel
            df_topics = pd.DataFrame(topics_data)
            df_topics.to_excel(writer, sheet_name=division[:31], index=False)
    
    print(f"\nExcel file created: {excel_file}")
    return excel_file


def export_document_topic_assignments(division_models, output_dir='output') -> pd.DataFrame:
    """
    Export document-topic assignments showing which documents belong to which topics
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    abstracts = []
    
    for division, model_data in division_models.items():
        lda_model = model_data['model']
        corpus = model_data['corpus']
        div_df = model_data['dataframe']
        
        
        for i in range(len(corpus)):
            # Get topic distribution for this abstract
            abstract_topic_dist = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
            
            # Convert to dictionary for easier access
            topic_probs = {topic_id: prob for topic_id, prob in abstract_topic_dist}
            
            # Create row with abstract 
            abstract = {
                'division': division,
                'year': div_df.iloc[i]['year'],
                'awd_id': div_df.iloc[i]['awd_id'],
            }
            
            # Add percentages for all topics (convert probability to percentage)
            for topic_id in range(lda_model.num_topics):
                prob = topic_probs.get(topic_id, 0.0)
                abstract[f'Topic_{topic_id + 1}_pct'] = round(prob * 100, 2)
            
            abstracts.append(abstract)
        
    # Create DataFrame and write to JSON and CSV
    df_topics = pd.DataFrame(abstracts)
    df_topics.to_json(output_path / 'topic_assignments.json')
    df_topics.to_csv(output_path / 'topic_assignments.csv')

    return df_topics

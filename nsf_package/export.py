"""
Export functions
"""

import pandas as pd


def export_topics_to_excel(division_models, excel_file='lda_topics_by_division.xlsx'):
    """
    Export topics to Excel file with one sheet per division (Cell 9)
    """
    # Create Excel writer
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        
        for division, model_data in division_models.items():
            lda_model = model_data['model']
            
            # Extract topics and their words
            topics_data = []
            
            # Get all topics for this division
            for topic_id in range(lda_model.num_topics):
                
                # Get topic words with probabilities of each token
                topic_words = lda_model.show_topic(topic_id, topn=20)  # Get top 20 words
                
                # Create a row for this topic
                topic_row = {'Topic': f'Topic {topic_id + 1}'}
                
                # Add words and probabilities
                for rank, (token, _) in enumerate(topic_words):
                    topic_row[f'Word {rank+1}'] = token
                
                topics_data.append(topic_row)
            
            # Create DataFrame for this division and write to Excel
            df_topics = pd.DataFrame(topics_data)
            df_topics.to_excel(writer, sheet_name=division[:31], index=False)
            
            print(f"Exported topics for: {division}")
    
    print(f"\nExcel file created: {excel_file}")
    return excel_file


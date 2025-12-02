"""
Model initialization
"""

import spacy


def load_spacy_model(model_path='models/en_core_web_sm/en_core_web_sm-3.8.0'):
    """
    Load spaCy model (Cell 0)
    """
    nlp = spacy.load(model_path, disable=['parser', 'ner'])
    return nlp


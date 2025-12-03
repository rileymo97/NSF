"""
Data cleaning functions
"""

import pandas as pd

# Handle imports for both package and direct execution
try:
    from .constants import NSF_MISSION_STATEMENT, DIVISION_NAMES
except ImportError:
    from constants import NSF_MISSION_STATEMENT, DIVISION_NAMES


def add_termination_flag(grants_df: pd.DataFrame, terminated_awards: pd.DataFrame) -> pd.DataFrame:
    """
    Identify whether grants are terminated or not 
    """
    grants_df['terminated'] = grants_df['awd_id'].isin(terminated_awards['awd_id']).astype(int)
    return grants_df


def add_abstract_column(row: pd.Series) -> str:
    """
    Combine both abstract cols into one value to extract keywords 
    """
    abstract = []
    if pd.notna(row["abst_narr_txt"]):
        abstract.append(row["abst_narr_txt"])
    if pd.notna(row["awd_abstract_narration"]):
        abstract.append(row["awd_abstract_narration"])
    
    abstract = "; ".join(abstract) if abstract else ""
    
    # Remove the NSF mission statement from the abstract
    abstract = abstract.replace(NSF_MISSION_STATEMENT, "")
    
    return abstract


def add_division_names(grants_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply division name mapping to grants datagrame
    """
    grants_df['division_name'] = grants_df['div_abbr'].map(DIVISION_NAMES)
    return grants_df
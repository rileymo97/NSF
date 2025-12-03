"""
Data loading functions
"""

import pandas as pd
import glob
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / os.getenv("DATA_PATH", "data")
TERMINATED_AWARDS_PATH = PROJECT_ROOT / os.getenv("TERMINATED_AWARDS_PATH", "data/NSF-Terminated-Awards.csv")

def load_grants_data():
    """Load grants data from JSON files"""
    rows = []
    
    # Loop through each year, 
    for year in range(2019, 2025):
        pattern = str(DATA_PATH / str(year) / "*.json")
        json_files = glob.glob(pattern)
        
        # Loop through each file
        for json_file in json_files:
            # Open the file and append rows to master list
            with open(json_file, "r") as f:
                data = json.load(f)
                data["year"] = year
                rows.append(data)
    
    # Convert rows to df
    grants_df = pd.DataFrame(rows)
    grants_df["awd_id"] = grants_df["awd_id"].astype(str)
    return grants_df


def load_terminated_awards():
    """Load list of NSF-terminated grants"""
    df = pd.read_csv(TERMINATED_AWARDS_PATH, encoding='latin1')
    df = df.rename(columns={"Award ID": "awd_id"})[["awd_id"]]
    df["awd_id"] = df["awd_id"].astype(str)
    return df


import re
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Optional but recommended
import matplotlib.pyplot as plt
import seaborn as sns

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # Remove URLs
    text = re.sub(r"@\w+", "", text)                 # Remove mentions
    text = re.sub(r"[^a-zA-Z0-9\s!?]", "", text)    # Keep ! and ? 
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    data_dir = r"C:\FYPPP2\toxicbaddie\data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("Loading Aggressive data...")
    df_agg = pd.read_csv(os.path.join(data_dir, "Aggressive_All.csv"))
    df_agg['label'] = 1
    
    print("Loading Non-Aggressive data...")
    df_non = pd.read_csv(os.path.join(data_dir, "Non_Aggressive_All.csv"))
    df_non['label'] = 0
    
    # Combine
    df = pd.concat([df_agg, df_non], ignore_index=True)
    
    print(f"\nTotal rows before preprocessing: {len(df):,}")
    print("Label distribution:")
    print(df['label'].value_counts())
    
    # ====================== PREPROCESSING STEPS ======================
    
    # 1. Handle missing values
    print("\nHandling missing values...")
    df = df.dropna(subset=['Message'])          # Remove rows with no message
    
    # 2. Remove duplicates
    print("Removing duplicates...")
    df = df.drop_duplicates(subset=['Message'])
    
    # 3. Text Cleaning
    print("Cleaning text...")
    tqdm.pandas()                               # Progress bar
    df['clean_text'] = df['Message'].progress_apply(clean_text)
    
    # 4. Add useful features (Text Statistics)
    print("Adding text statistics...")
    df['text_length'] = df['clean_text'].apply(len)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    
    # 5. Remove very short messages (optional)
    df = df[df['word_count'] >= 3]               # Remove messages with less than 3 words
    
    # ====================== EDA (Exploratory Data Analysis) ======================
    print("\nGenerating basic statistics...")
    print(df.groupby('label')[['text_length', 'word_count']].mean())
    
    # Optional: Save visualization
    plt.figure(figsize=(10,5))
    sns.histplot(data=df, x='text_length', hue='label', bins=50, alpha=0.7)
    plt.title('Message Length Distribution by Class')
    plt.savefig(os.path.join(data_dir, 'text_length_distribution.png'))
    plt.close()
    
    # ====================== Final Output ======================
    df = df[['Message', 'clean_text', 'text_length', 'word_count', 'label']]
    
    output_path = os.path.join(data_dir, "combined_cleaned_data.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ PREPROCESSING COMPLETED!")
    print(f"Final rows: {len(df):,}")
    print(f"File saved at: {output_path}")
    print("\nReady for Feature Engineering & Model Training!")
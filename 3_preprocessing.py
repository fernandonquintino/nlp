import pandas as pd
import os
import sys
sys.path.append(os.path.abspath('src'))
import re
from collections import Counter
from nltk.corpus import stopwords
from symspellpy.symspellpy import SymSpell
from src.llm_preprocessing import generate_whitelist
from src.text_preprocessing import clean_text, correct_typos, spacy_lemmatize, remove_stopwords, light_clean_text_for_bert, apply_whitelist
from resources import blocklist, whitelist

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = 'data/dictionaries/frequency_dictionary_en_82_765.txt'  # included in the repo
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def generate_corrections_and_whitelist(df_texts, flag_api=False):
    correction_counter = Counter()
    blocklist = {'the': 'the', '!': '!', 's': 's', 'n': 'n', 'wa': 'was'}
    whitelist = []

    if not flag_api:
        # Use hardcoded defaults and save immediately
        whitelist = ['branzino', 'carozza', 'confit', 'congee', 'cozy', 'dal', 'facie', 'favorite', 'fixe', 'flavoring',
                     'foie', 'frites', 'gras', 'grazie', 'gulab', 'jamun', 'kha', 'maitre', 'medicore', 'mizu', 'mom',
                     'msg', 'neighborhood', 'noir', 'nyc', 'overpack', 'porcini', 'prima', 'prixe', 'russe', 'shabu',
                     'svc', 'tartare', 'theater', 'tristate', 'uncourteous', 'volare', 'wks']
    else:
        # First correction pass with empty whitelist (using only blocklist)
        df_texts[['corrected_text', 'corrections']] = df_texts['text_v1'].apply(
            lambda x: pd.Series(correct_typos(x, blocklist, whitelist))
        )
        # Generate whitelist using the corrections and the LLM
        df_corrections = df_texts.loc[df_texts['corrections'].notna(), ["corrected_text", "corrections"]]
        results = generate_whitelist(df_corrections, 'corrected_text', model_name='gpt4')
        df_corrections['result'] = results
        whitelist = apply_whitelist(df_corrections)

    # Apply second correction pass with updated whitelist
    df_texts['text_v2'] = df_texts["text_v1"].apply(lambda x: pd.Series(correct_typos(x, blocklist, whitelist)))[0]

    # Clean up temp columns
    df_texts.drop(columns=['corrected_text', 'corrections'], inplace=True, errors='ignore')

    # Custom stopwords (optional, you might want to export this as well)
    stop_words = set(stopwords.words("english"))
    stop_words.update({'I', 'i', 'n', 'wa', 's'})

    # Save the blocklist and whitelist to `resources.py`
    write_resources_py(blocklist, whitelist)

    return df_texts, blocklist, whitelist, stop_words

def write_resources_py(blocklist, whitelist):
    content = f"""# This file is auto-generated by 3_preprocessing.py
blocklist = {repr(blocklist)}
whitelist = {repr(whitelist)}
"""
    with open('src/resources.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("src/resources.py updated successfully.")


def full_preprocess(text, blocklist=blocklist, whitelist=whitelist):
    text = clean_text(text)
    text = correct_typos(text, blocklist, whitelist)[0]
    text = spacy_lemmatize(text)
    text = remove_stopwords(text)
    return text


df_pretreatment = pd.read_csv('data/processed/final_label.csv')
df = df_pretreatment.copy()

df['text_v1'] = df['text'].apply(clean_text)
df, blocklist, whitelist, stop_words = generate_corrections_and_whitelist(df, flag_api=False)

df = df_pretreatment.copy()
df['has_number'] = df['text'].str.contains(r"\d")
df['has_rating_number'] = df["text"].str.contains(
    r"\b(?:give it a|rated?|score|rate|i give|i give this|its a|it's a|i rate|rating|^)(?:\s+)?(?:10|[0-9])\b",
    flags=re.IGNORECASE
)

df['final_text'] = df['text'].apply(full_preprocess)
df['bert_text'] = df['text'].apply(light_clean_text_for_bert)

df.to_csv('data/processed/preprocessed_train.csv', index=False)
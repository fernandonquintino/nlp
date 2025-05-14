import re
import spacy
import emoji
from nltk.corpus import stopwords
from symspellpy import SymSpell, Verbosity
from collections import Counter
from resources import blocklist, whitelist
from resources_lemmatization import trigram_group_mapping, bigram_group_mapping, trigram_list, bigram_list

# Load resources once
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

# Symspell setup (you should load your dictionary separately if needed)
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
# Example load dictionary (ensure you load your correct file here)
# sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

# Counter for corrections
correction_counter = Counter()

def contains_emoji(text):
    return any(char in emoji.EMOJI_DATA for char in text)

def clean_text(text):
    text = text.lower()
    text = emoji.demojize(text)  # Convert emojis to text
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s!]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def correct_typos(text, blocklist=blocklist, whitelist=whitelist):
    corrected_words = []
    corrections_in_row = []
    
    for word in text.split():
        if word in blocklist or word in whitelist or len(word) <= 2:
            corrected_words.append(word)
            continue
        
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        
        if suggestions and suggestions[0].term != word:
            correction_counter[(word, suggestions[0].term)] += 1
            corrected_words.append(suggestions[0].term)
            corrections_in_row.append(word)
        else:
            corrected_words.append(word)
    
    return " ".join(corrected_words), "; ".join(corrections_in_row) if corrections_in_row else None

def spacy_lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def remove_stopwords(text):
    tokens = text.split()
    return " ".join([w for w in tokens if w not in stop_words])

def light_clean_text_for_bert(text):
    text = emoji.demojize(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def enrich_text_full(text, trigram_group_mapping=trigram_group_mapping, 
                     bigram_group_mapping=bigram_group_mapping, 
                     trigram_list=trigram_list, bigram_list=bigram_list):
    # 1. Replace trigram group mappings
    for trigram, unified in trigram_group_mapping.items():
        text = text.replace(trigram, unified)
    
    # 2. Replace bigram group mappings
    for bigram, unified in bigram_group_mapping.items():
        text = text.replace(bigram, unified)
    
    # 3. Replace remaining specific trigrams as unigrams
    for trigram in trigram_list:
        text = text.replace(trigram, trigram.replace(' ', '_'))
    
    # 4. Replace remaining specific bigrams as unigrams
    for bigram in bigram_list:
        text = text.replace(bigram, bigram.replace(' ', '_'))
    
    return text

def apply_whitelist(df):
    whitelist_set = set()

    for idx, row in df.iterrows():
        words = [w.strip() for w in row['corrections'].split(';') if w.strip()]
        results = row['result']

        for word, flag in zip(words, results):
            if flag == 1:
                whitelist_set.add(word)

    return sorted(whitelist_set)
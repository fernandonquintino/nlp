from pathlib import Path

def ensure_resources_files():
    resources_file = Path("src/resources.py")
    resources_lemmatization_file = Path("src/resources_lemmatization.py")

    if not resources_file.exists():
        print(" Creating empty src/resources.py...")
        with open(resources_file, 'w', encoding='utf-8') as f:
            f.write("# Auto-created placeholder\n\nblocklist = {}\nwhitelist = []\n")

    if not resources_lemmatization_file.exists():
        print(" Creating empty src/resources_lemmatization.py...")
        with open(resources_lemmatization_file, 'w', encoding='utf-8') as f:
            f.write("# Auto-created placeholder\n\ntrigram_group_mapping = {}\nbigram_group_mapping = {}\ntrigram_list = []\nbigram_list = []\n")

    print(" Resource files ensured.")
import spacy
import pandas as pd

df = pd.read_json("without_assessment.jsonl", lines=True)
articles = df["Text"].to_list()
titles = df["Title"].to_list()
data = df[["Title", "Text"]].to_dict(orient="records")

nlp = spacy.load("en_core_web_sm")

def pos_tagging(text):
    if isinstance(text, str):
        doc = nlp(text)
        pos_counts = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0, 'PRON': 0}
        total_tokens = 0

        for token in doc:
            if token.is_alpha:
                total_tokens += 1
                if token.pos_ in pos_counts:
                    pos_counts[token.pos_] += 1

      
        if total_tokens == 0:
            return []

   
        noun_pct = round(pos_counts['NOUN'] / total_tokens, 3)
        verb_pct = round(pos_counts['VERB'] / total_tokens, 3)
        adj_pct  = round(pos_counts['ADJ']  / total_tokens, 3)
        adv_pct  = round(pos_counts['ADV']  / total_tokens, 3)
        pron_pct = round(pos_counts['PRON'] / total_tokens, 3)

      
        adj_noun_ratio = round(pos_counts['ADJ'] / pos_counts['NOUN'], 3) if pos_counts['NOUN'] != 0 else 0
        adv_to_verb = round(pos_counts['ADV']   / pos_counts['VERB'], 3) if pos_counts['VERB'] else 0
        return [noun_pct, verb_pct, adj_pct, adv_pct, pron_pct, adj_noun_ratio, adv_to_verb]
    return []

# print(pos_tagging(data[10]["Text"]))
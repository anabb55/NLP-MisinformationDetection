import spacy
import pandas as pd
from nrclex import NRCLex
from preprocessing import lowercase_text_fields, tokenize_regex, remove_stopwords, remove_punctuation, split_sentence
from spellchecker import SpellChecker
import language_tool_python 
from collections import defaultdict

df = pd.read_json("without_assessment_updated.jsonl", lines=True)
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


## I found out that before applying emotional analysis stop words should be removed
def emotion_analysis(text):
    if not isinstance(text, str):
        return [0.0] * 10
    
    emotion = NRCLex(text)

    emotion_list = ['anger', 'fear', 'disgust', 'sadness', 'joy', 'surprise', 'trust', 'anticipation', 'positive', 'negative']
    total_words =  len(emotion.words)
    raw_emotions = emotion.raw_emotion_scores

    features = []
    for e in emotion_list:
        count = raw_emotions.get(e, 0)
        score = count / total_words if total_words > 0 else 0.0
        features.append(round(score, 3))

    return features


def named_entity_recognition(text):
    nlp = spacy.load("en_core_web_sm")
    if not isinstance(text, str) or not text.strip():
        return {}
    
    doc = nlp(text)
    entities = defaultdict(list)
    
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)

    return dict(entities)


def remove_named_entities(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    
    doc = nlp(text)
    cleaned_text = text

    for ent in reversed(doc.ents):
        start = ent.start_char
        end = ent.end_char
        cleaned_text = cleaned_text[:start] + " " * (end - start) + cleaned_text[end:]

       
    cleaned_text = ' '.join(cleaned_text.split()) 
    return cleaned_text
    


def analyze_text_errors(text):
    """
    Calculates spelling error rate and grammar error rate for a given text

    """

    if not isinstance(text, str):
        return 0.0, 0.0
    
    spell = SpellChecker()
    words = text.lower().split()
    misspelled = spell.unknown(words)
    spelling_error_rate = len(misspelled) / len(words) if words else 0.0

    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    num_sentences = max(text.count('.'), 1)
    grammar_error_rate = len(matches) / num_sentences

    #Test for identifying grammar errors
    # print("\n--- Grammar Issues ---")
    # for match in matches:
    #     start = match.offset
    #     end = match.offset + match.errorLength
    #     error_snippet = text[start:end]
    #     print(f"• Issue at position [{start}:{end}] → '{error_snippet}'")
    #     print(f"  ↳ Message: {match.message}")
    #     print(f"  ↳ Suggested Correction(s): {match.replacements}\n")

    return round(spelling_error_rate, 3), round(grammar_error_rate, 3)




##TEST - this should be in another file, but this is just test
# original_text = data[100]["Text"]
# raw_scores = emotion_analysis(original_text)

# data_clean = lowercase_text_fields(data)
# tokens = tokenize_regex(data_clean[100]["Text"])
# tokens_without_sw = remove_stopwords(tokens)
# data_clean_text = " ".join(tokens_without_sw)

# print("Raw", raw_scores)
# print(f"Preprocessed: {emotion_analysis(data_clean_text)}")

# for article in data:
#    print(analyze_text_errors(article["Text"]))
   
#print(analyze_text_errors(remove_named_entities(data[1]["Text"])))


###COMENTS: 
# 1. Applying lowercasing before emotion_analysis doesn't change the output
# 2. After tokenization text will look like 'word' 'word'... but emotion_analysis requires string, so it is necessary to join tokens
# 3. Function tokenize_regex also removes punctuation
# 4. Function remove_stopwords has list of tokens as input
# 5. After all this preprocessing steps (lowercasing, tokenizing and removing stopwords) the result is better(got higher values)
# 6. It's really important to remove names of organizations, perosnal names ... before using analyze_text_error because they cause a lot of grammatical errors
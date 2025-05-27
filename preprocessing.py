import pandas as pd
import re
import html
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("punkt")
import string


df = pd.read_json("without_assessment.jsonl", lines=True)
articles = df["Text"].to_list()
titles = df["Title"].to_list()
data = df[["Title", "Text"]].to_dict(orient="records")


def lowercase_text_fields(data):
    """
    Lowercase the "Title" and "Text" fields of each dictionary in the list"

    """
    lowercased_data = []
    for i in data:
        def clean(text):
            if isinstance(text, str):
                text = text.lower()
                text = html.unescape(text)
                text = re.sub(r"[\r\n]+", " ", text)  
                text = re.sub(r"\s+", " ", text)  
                return text.strip()
            
            return ""
    
        lowered_item = {
            "Title":  clean(i.get("Title")),
            "Text": clean(i.get("Text"))
        }

        lowercased_data.append(lowered_item)

    return lowercased_data




def tokenize(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in string.punctuation]
        return tokens
    return []


##maybe we also can include hyphenated words (self-esteem)??
def tokenize_regex(text):
    if isinstance(text, str):
        return re.findall(r"\b\w[\w']*\b", text.lower())
    return []

def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    return [t for t in tokens if t not in stop_words]

def split_sentence(text):
    if isinstance(text, str):
        return sent_tokenize(text)
    return []

def remove_punctuation(text):
    if isinstance(text, str):
        punctuation_to_remove = string.punctuation.replace("'", "")
        return text.translate(str.maketrans("", "", punctuation_to_remove))
    return []

# cl_data = lowercase_text_fields(data)
# tokens = tokenize_regex(cl_data[1]["Text"])
# print(remove_punctuation(data[1]["Text"]))
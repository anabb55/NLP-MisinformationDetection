from text_analysis import lowercase_text_fields, remove_named_entities, pos_tagging, tokenize_regex, remove_stopwords, emotion_analysis, remove_punctuation, analyze_text_errors
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


df = pd.read_json("without_assessment_updated.jsonl", lines=True)
articles = df["Text"].to_list()
titles = df["Title"].to_list()
data = df[["Title", "Text"]].to_dict(orient="records")

features = []
data_lowercase = lowercase_text_fields(data)

for article in data_lowercase:
    # print(article["Title"])
    title = article["Title"]
    text = article["Text"]

    # POS tagging
    pos_feats = pos_tagging(text)

    # Emotion analysis (on preprocessed text)
    text_wo_punct = remove_punctuation(text)
    tokens = tokenize_regex(text_wo_punct)
    tokens_wo_sw = remove_stopwords(tokens)
    clean_text = " ".join(tokens_wo_sw)
    emotion_feats = emotion_analysis(clean_text)

    # Text errors (really slow)
    #ner_cleaned_text = remove_named_entities(text)
    #spell_err, grammar_err = analyze_text_errors(ner_cleaned_text)

    row = {
        "Title": title,
        "noun_pct": pos_feats[0] if pos_feats else 0.0,
        "verb_pct": pos_feats[1] if pos_feats else 0.0,
        "adj_pct": pos_feats[2] if pos_feats else 0.0,
        "adv_pct": pos_feats[3] if pos_feats else 0.0,
        "pron_pct": pos_feats[4] if pos_feats else 0.0,
        "adj_noun_ratio": pos_feats[5] if pos_feats else 0.0,
        "adv_verb_ratio": pos_feats[6] if pos_feats else 0.0,
        #"spelling_error_rate": spell_err,
        #"grammar_error_rate": grammar_err
    }

    emotions = ['anger', 'fear', 'disgust', 'sadness', 'joy', 'surprise', 'trust', 'anticipation', 'positive', 'negative']
    for i, emotion in enumerate(emotions):
        row[f"emotion_{emotion}"] = emotion_feats[i]

    features.append(row)

features_df = pd.DataFrame(features)
labels_df = pd.read_csv("group31_stage1.csv", sep=";")
labels_df["label"] = labels_df["real_news"].replace({"yes": 1, "no": 0})
adjusted_index = labels_df["index"] - 1
features_df.loc[adjusted_index, "Label"] = labels_df["label"].values
# print(features_df.head)

X = features_df.drop(['Label', 'Title'], axis=1)
y = features_df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluation(model, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = model.predict(X_train)
        y_true = y_train
        dataset_type = "Training"
    else:
        pred = model.predict(X_test)
        y_true = y_test
        dataset_type = "Test"

    # Metrics
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred)
    rec = recall_score(y_true, pred)
    f1 = f1_score(y_true, pred)

    print(f"{dataset_type} Result:\n" + "="*50)
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision      : {prec:.4f}")
    print(f"Recall         : {rec:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"{dataset_type} Confusion Matrix")
    plt.grid(False)
    plt.show()

# Define parameter grid
param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': [None, 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'bootstrap': [True, False]
}


random_forest_model = RandomForestClassifier(random_state=0)

# RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=random_forest_model,
                                   param_distributions=param_distributions,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   random_state=42,
                                   n_jobs=-1)

random_search.fit(X_train, y_train)

# Best parameters
best_params = random_search.best_params_
print("Best Parameters found: ", best_params)

best_model = random_search.best_estimator_
evaluation(best_model, X_train, y_train, X_test, y_test, train=True)
evaluation(best_model, X_train, y_train, X_test, y_test, train=False)

# Result on the whole set
evaluation(best_model, X_train, y_train, X, y, train=False)
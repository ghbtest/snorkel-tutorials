# 2022-03-16 20:07:35
# for: organize codes from the examples

#%% /Users/robinlu/Library/CloudStorage/OneDrive-UNIPHORESOFTWARESYSTEMSPVTLTD/Data_Programming/snorkel-tutorials/spam/01_spam_tutorial.py
# /Users/robinlu/NLP/data_programming/snorkel-tutorials/spam/01_spam_tutorial.py
# line 471
# 1. write labelling functions
# a) Keyword LFs
from snorkel.labeling import LabelingFunction
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import labeling_function
from snorkel.labeling import LFAnalysis
import matplotlib.pyplot as plt


ABSTAIN = -1
HAM = 0
SPAM = 1
fp="/Users/robinlu/NLP/data_programming/snorkel-tutorials/"

import pickle
# pickle.dump([df_train, df_test],open(fp+"/spam/spamdata.pickle","wb"))

[df_train,df_test] = pickle.load(open(fp+"spam/spamdata.pickle", "rb"))

# We pull out the label vectors for ease of use later
print (df_train[:3])
Y_test = df_test.label.values

#%%

def keyword_lookup(x, keywords, label):
    if any(word in x.text.lower() for word in keywords):
        return label
    return ABSTAIN


def make_keyword_lf(keywords, label=SPAM):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )


"""Spam comments talk about 'my channel', 'my video', etc."""
keyword_my = make_keyword_lf(keywords=["my"])

"""Spam comments ask users to subscribe to their channels."""
keyword_subscribe = make_keyword_lf(keywords=["subscribe"])

"""Spam comments post links to other channels."""
keyword_link = make_keyword_lf(keywords=["http"])

"""Spam comments make requests rather than commenting."""
keyword_please = make_keyword_lf(keywords=["please", "plz"])

"""Ham comments actually talk about the video's content."""
keyword_song = make_keyword_lf(keywords=["song"], label=HAM)


# b) Pattern-matching LFs (regular expressions)
import re
@labeling_function()
def regex_check_out(x):
    return SPAM if re.search(r"check.*out", x.text, flags=re.I) else ABSTAIN

# c)  Heuristic LFs
@labeling_function()
def short_comment(x):
    """Ham comments are often short, such as 'cool video!'"""
    return HAM if len(x.text.split()) < 5 else ABSTAIN

# d) LFs with Complex Preprocessors
from snorkel.labeling.lf.nlp import nlp_labeling_function

@nlp_labeling_function()
def has_person_nlp(x):
    """Ham comments mention specific people and are short."""
    if len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents]):
        return HAM
    else:
        return ABSTAIN

# e) Third-party Model LFs
from snorkel.preprocess import preprocessor
from textblob import TextBlob

@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    x.subjectivity = scores.sentiment.subjectivity
    return x

@labeling_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    return HAM if x.polarity > 0.9 else ABSTAIN

@labeling_function(pre=[textblob_sentiment])
def textblob_subjectivity(x):
    return HAM if x.subjectivity >= 0.5 else ABSTAIN



#%% 2. Combining Labeling Function Outputs with the Label Model
lfs = [
    keyword_my,
    keyword_subscribe,
    keyword_link,
    keyword_please,
    keyword_song,
    regex_check_out,
    short_comment,
    has_person_nlp,
    textblob_polarity,
    textblob_subjectivity,
]



#%% from raw data to label matrix
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
type(L_train)
L_test = applier.apply(df=df_test)
L_test
LFAnalysis(L=L_train, lfs=lfs).lf_summary()

#%% check histogram: no of labels vs. coverage

# %matplotlib inline
def plot_label_frequency(L):
    plt.hist((L != ABSTAIN).sum(axis=1), density=True, bins=range(L.shape[1]))
    plt.xlabel("Number of labels")
    plt.ylabel("Fraction of dataset")
    plt.show()

plot_label_frequency(L_train)
L_train.shape[1] #cols
#%% generate one label based on the label matrix
from snorkel.labeling.model.baselines import MajorityLabelVoter

majority_model = MajorityLabelVoter()
preds_train = majority_model.predict(L=L_train)
preds_train

import importlib
importlib.reload(snorkel)
import snorkel
from snorkel.labeling.model.label_model import LabelModel

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)
type(label_model) #<class 'snorkel.labeling.model.label_model.LabelModel'>
# %%
majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
    "accuracy"
]
print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%") #84%

majority_acc = majority_model.score(L=L_test, Y=Y_test, metrics=["precision", "recall", "f1"], tie_break_policy="random")
majority_acc

label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
    "accuracy"
]
print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%") #86%

#%% Let's briefly confirm that the labels the `LabelModel` produces are indeed probabilistic in nature.
# The following histogram shows the confidences we have that each data point has the label SPAM.
# The points we are least certain about will have labels close to 0.5.

def plot_probabilities_histogram(Y):
    plt.hist(Y, bins=10)
    plt.xlabel("Probability of SPAM")
    plt.ylabel("Number of data points")
    plt.show()

probs_train = label_model.predict_proba(L=L_train)
probs_train
plot_probabilities_histogram(probs_train[:, SPAM])

# %% Filtering out unlabeled data points
from snorkel.labeling import filter_unlabeled_dataframe

df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=probs_train, L=L_train
)
df_train_filtered[:5]
df_train[:5]
probs_train_filtered

#%% use Logistic Regression for classification 
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(1, 5))
X_train = vectorizer.fit_transform(df_train_filtered.text.tolist())
X_test = vectorizer.transform(df_test.text.tolist())
X_test

#turn labels with probabilities to one label
from snorkel.utils import probs_to_preds

preds_train_filtered = probs_to_preds(probs=probs_train_filtered)
preds_train_filtered

from sklearn.linear_model import LogisticRegression

sklearn_model = LogisticRegression(C=1e3, solver="liblinear")
sklearn_model.fit(X=X_train, y=preds_train_filtered)
print(f"Test Accuracy: {sklearn_model.score(X=X_test, y=Y_test) * 100:.1f}%")

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
import pandas as pd
import re

ABSTAIN = -1
# HAM = 0
# SPAM = 1
PRM = 1
NONPRM =0
fp="/Users/robinlu/NLP/data_programming/snorkel-tutorials/"
dtp="/Users/robinlu/NLP/data_programming/data/"

import pickle
# pickle.dump([df_train, df_test],open(fp+"/spam/spamdata.pickle","wb"))
# [df_train,df_test] = pickle.load(open(fp+"spam/spamdata.pickle", "rb"))
dfall=pickle.load(open(dtp+"dfall.pickle", "rb"))
dfall.shape
dfall.index
dfall[:3]
# We pull out the label vectors for ease of use later
# print (df_train[:3])
# Y_test = df_test.label.values


#%%2022-03-26 12:22:06
#%% keyword match

def keyword_lookup(x, keywords, label):
    if any(word in x.turn_text.lower() for word in keywords):
        return label
    return ABSTAIN

def make_keyword_lf(keywords, label):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )

#%% Pattern-matching LFs (regular expressions)

def regex_lookup(x, pattern, label):
    regex = re.compile(pattern, re.I)
    if re.search(regex, x.turn_text):
        return label
    return ABSTAIN

def gen_regex_lf(pattern, label):
    return LabelingFunction(
        name=f"regex pattern: {pattern}",
        f=regex_lookup,
        resources=dict(pattern=pattern, label=label),
    )
    
# regexpt1=gen_regex_lf(pattern=pt1, label=PRM)

#%%2022-03-26 11:11:49 checking rules
import yaml

with open(dtp+"prmdict.yml", "r") as f:
    dt=yaml.safe_load(f)
dt.keys()
['PRP1_WILL', 'PRP1_CAN', 'PRP2_WILL', 'PRP3_WILL', 'FILLER', 'VB_HELP', 'VB_ACTION_AGENT', 
 'VB_ACTION_AGENT-', 'VB_ACTION_CLIENT', 'SUBJ_THING', 'MDN_WILL', 'VBN_ACTION', 'SUBJ_PERSON']

def testpt(pt, s):
    regex = re.compile(pt, re.I)
    return re.search(regex, s)



# promise patterns
#subj: agent
[RP1_WILL|PRP3_WILL|PRP1_CAN].*(VB_HELP)?
[RP1_WILL|PRP3_WILL|PRP1_CAN].*(VB_HELP)?VB_ACTION_AGENT

pt0=r"|".join(re.escape(x) for x in dt['PRP1_WILL']+dt['PRP3_WILL']+dt['PRP1_CAN'])
pt1=pt0+r".*"+r'|'.join(re.escape(x+"+") for x in dt['VB_HELP'])
pt1
testpt(pt1, "we can help you")

regexpt0=gen_regex_lf(pattern=pt0, label=PRM)
kw_VB_ACTION_AGENT = make_keyword_lf(keywords=dt['VB_ACTION_AGENT'], label=PRM)


# other
"We can definitely get your started on a new plan today."
[RP1_WILL|PRP3_WILL|PRP1_CAN].*(get you started)

#subj: client
PRP2_WILL 
PRP2_WILL VB_ACTION_CLIENT 

#subj: things
MDN_WILL 
MDN_WILL VBN_ACTION
[k (that) MDN_WILL v for k,v in dt["SUBJ_THING"]]

#subj: other person, e.g. technician|they
SUBJ_PERSON ("will"|"is going to"|"are going to") (verb not sure)

#not promise
#subj: agent
[RP1_WILL|PRP3_WILL|PRP1_CAN].*(VB_HELP)?VB_ACTION_AGENT-

#clause preceded by "if"
("if") [RP1_WILL|PRP3_WILL|PRP1_CAN].*(VB_HELP)?

#speak is R





# c)  Heuristic LFs
# @labeling_function()
# def short_comment(x):
#     """Ham comments are often short, such as 'cool video!'"""
#     return HAM if len(x.text.split()) < 5 else ABSTAIN

@labeling_function()
def speakertype(x):
    #Speaker_R is not PRM
    return NONPRM if x.speaker == "Speaker_R" else ABSTAIN

#%% 2. Combining Labeling Function Outputs with the Label Model
lfs = [
    regex_0,
    speakertype
]

lfs = [
    keyword_vbs0,
    regexpt1,
    speakertype
]



#%% from raw data to label matrix
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=dfall)
type(L_train)
L_train[:2]
df0=pd.DataFrame(L_train, columns=["regex", "speaker"])
df0.shape
df1=df0[df0.regex==1]
l=df1.index.tolist()
l
df2=dfall.iloc[l]
df2.to_csv(dt+"df2.csv")
L_train0=L_train
pickle.dump([L_train0,df2],open(dt+"exp1.pickle","wb"))

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
from snorkel.labeling.model.label_model import LabelModel

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)
type(label_model) #<class 'snorkel.labeling.model.label_model.LabelModel'>
# %% when gold labels are provided
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
plot_probabilities_histogram(probs_train[:, PRM])

# %% Filtering out unlabeled data points
from snorkel.labeling import filter_unlabeled_dataframe

df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=dfall, y=probs_train, L=L_train
)
df_train_filtered.shape
df_train_filtered.index
df_train_filtered[:5]

probs_train_filtered[:5]

#%% supervised classification 


#%% google search
python re cannot process flags argument with a compiled pattern

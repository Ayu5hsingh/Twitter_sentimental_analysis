from lib2to3.pgen2.tokenize import tokenize
import pandas as pd
from transformers import AutoTokenizer
from scipy.special import softmax
import tensorflow as tf
from transformers import AutoModelForSequenceClassification
import numpy 
tweet_words=[]


if __name__=='__main__':
    #getting the data
    data = pd.read_csv("scraped_tweets.csv", error_bad_lines=False)
    

def getting_text():
    j=0
    for i in data["text"][j]:
        j=j+1
        sentimental(data["text"][j])
        

def sentimental(textt):
    
    for words in textt.split(" "):
        if words.startswith('@') and len(words)>1:
            words="@user"
        elif words.startswith('http'):
            words="http"
        tweet_words.append(words)
    tweet_proc=" ".join(tweet_words)

    # Load the model adn tokenizer
    roberta="cardiffnlp/twitter-roberta-base-sentiment"
    model= AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    labels= ["Negative","Neutral","Positive"]


    # Sentimental Analysis
    encoded_tweet = tokenizer(tweet_proc,return_tensors='pt')
    output = model(**encoded_tweet)
    print("************************")
    print(data["text"][0])
    print("************************")
    print("************************")
    print("******Output************")
    print("************************")
    scores = output[0][0].detach()

    for i in range(len(scores)):
        l= labels[i]
        s= scores[i]
        print(l,s)





getting_text()
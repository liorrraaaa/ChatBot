import pathlib
import random
import pickle
import os
from nltk.stem.porter import *
from nltk import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import operator
from nltk.corpus import stopwords
import math
import wikipedia as wk
import warnings
warnings.filterwarnings("ignore")


def create_tf_dict(list_part):
    list_part = list_part.lower()
    stopword = stopwords.words('english')
    stopword += ["else", "like", "go", "get", "let", "instead", "nothing", "feel", "feeling", "feelings"]
    tokens = word_tokenize(list_part)
    tokens = [w for w in tokens if w.isalpha() and w not in stopword]

    # get term frequencies in a more pythonic way
    token_set = set(tokens)
    tf_dict = {t: tokens.count(t) for t in token_set}

    # normalize tf by number of tokens
    for t in tf_dict.keys():
        tf_dict[t] = tf_dict[t] / len(tokens)

    return tf_dict


def create_tfidf(tf, idf):
    tf_idf = {}
    for t in tf.keys():
        tf_idf[t] = tf[t] * idf[t]

    return tf_idf


def extract_important_terms(list_part):
    idf_dict = {}
    doc = list_part

    tf_doc = create_tf_dict(doc)
    vocab = set(tf_doc.keys())

    for term in vocab:
        temp = ['x' for voc in tf_doc.keys() if term in voc]
        idf_dict[term] = math.log((1) / (1 + len(temp)))

    tf_idf_doc = create_tfidf(tf_doc, idf_dict)
    doc_term_weights = sorted(tf_idf_doc.items(), key=lambda x: x[1], reverse=True)
    terms = doc_term_weights[:10]
    term_list = []
    for i in terms:
        term_list.append(i[0])
    return term_list


# Removes newlines and other unwanted things
def clean_up_text(text):
    clean_text = text.replace("\n", "")
    clean_text = clean_text.replace("\ufeff", "")
    return clean_text


def welcome_member(first, name):
    welcome_back = ["Welcome back, " + name + ".", "Nice to see you again, " + name + "."]
    welcome_new = ["Welcome " + name + ". How are feeling?"]
    if first == "false":
        return random.choice(welcome_back)
    elif first == "true":
        return random.choice(welcome_new)


def hello_message(user_response, name):
    welcome_input = ("hello", "hi", "greetings", "sup", "what's up", "hey")
    welcome_response = ["Hello there. How are you, " + name + "?", "Hi " + name + ". How are you?"]

    for word in user_response.split():
        if word.lower() in welcome_input:
            return random.choice(welcome_response)


def feeling_response(user_response):
    analysis = sentiment_analysis(user_response)

    if analysis[0] == "pos":
        return "Last time we spoke you were feeling great! How are you feeling today?"
    elif analysis[0] == "neu":
        return "Last time we spoke you were feeling alright. How are you feeling today?"
    elif analysis[0] == "neg":
        return "Last time we spoke you weren't feeling too good. How are you feeling today?"


def sentiment_analysis(sentence):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(sentence)
    del vs['compound']
    return max(vs.items(), key=operator.itemgetter(1))


def emotional_response(user_resp):
    analysis = sentiment_analysis(user_resp)  # get pos, neu, or neg
    pos_response = ["OK! Anything you want to talk about?", "I see! What else do you want to talk about?"]
    neu_response = ["Very well. What's going on?", "I see. Why do you feel this way?", "Alright. What's going on?"]
    neg_response = ["I am sorry to hear that. What's going on?", "Oh no. What's going on?"]

    if analysis[0] == "pos":
        return "Emma : " + random.choice(pos_response)
    elif analysis[0] == "neu":
        return "Emma : " + random.choice(neu_response)
    elif analysis[0] == "neg":
        return "Emma : " + random.choice(neg_response)


def StemTokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def get_best_answer(list, stem_user_response):
    new_list = []
    for part in list:
        term_list = StemTokens(extract_important_terms(part))
        for term in term_list:
            if term in stem_user_response:
                new_list.append(part)
    if not new_list:
        return ""
    else:
        return random.choice(new_list)


def generate_response(user_response, list):
    analysis = sentiment_analysis(user_response)
    more = ["OK. Tell me more.", "OK. Anything else?"]

    if "tell me about" in user_response:
        print("Let me see...")
        return wiki_data(user_response)

    if "how are you" in user_response:
        return "I am good! How are you?"

    # Stem user_response
    user_response = word_tokenize(user_response)
    stem_user_response = StemTokens(user_response)

    if analysis[0] == "neg" or analysis[0] == "neu":
        answer = get_best_answer(list, stem_user_response)
        if answer != "":
            return "Here's a tip to help you. " + answer
    else:
        return "What else is on your mind?"

    return random.choice(more)

# wikipedia search
def wiki_data(text):
    reg_ex = re.search('tell me about (.*)', text)
    try:
        if reg_ex:
            topic = reg_ex.group(1)
            wiki = wk.summary(topic, sentences = 3)
            return wiki
    except Exception as e:
            return "Emma : No content has been found"


def print_lines(text):
    list = text.split(' ')
    counter = 0
    word = word_tokenize(text)
    for i in list:
        if counter == 15:
            counter = 0
            print(i)
        else:
            print(i, end=" ")
        counter += 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    flag = True
    welcome_bool = False
    fact_bool = False
    user_name = ""
    user_dict = {}
    combined_fact = ""


    # COMBINED FACTS
    with open(pathlib.Path.cwd().joinpath("combined_facts"), 'r') as f:
        raw_combined_facts = f.read()

    clean_combined_facts = clean_up_text(raw_combined_facts)
    combined_facts_list = clean_combined_facts.split(';')
    while "" in combined_facts_list:
        combined_facts_list.remove("")

    # COMBINED
    with open(pathlib.Path.cwd().joinpath("combined_tips"), 'r', encoding='utf-8') as f:
        raw_combined = f.read()

    clean_combined = clean_up_text(raw_combined)
    combined_list = clean_combined.split(';')
    while "" in combined_list:
        combined_list.remove("")

    # AFFIRMATIONS
    with open(pathlib.Path.cwd().joinpath("affirmations"), 'r') as f:
        raw_affirmations = f.read()

    clean_affirmations = clean_up_text(raw_affirmations)
    affirmations_list = clean_affirmations.split('.')
    while "" in affirmations_list:
        affirmations_list.remove("")


    print("\nMy name is Emma Bot and I am here to help you with your anxious feelings. If you want to exit, type 'bye'.")
    print("Emma : What is your name?")
    print("> ", end="")
    user_name = input()

    user_dict = {'name': user_name, 'first_time': '', 'feeling': ''}

    # if userfile already exists, is a returning user!
    if os.path.isfile(user_name + ".pickle"):
        # load
        with open(user_name + ".pickle", 'rb') as handle:
            user_dict = pickle.load(handle)
            user_dict["first_time"] = "false"
    # this file does not exist yet, new user!
    else:
        user_dict["first_time"] = "true"

    # Emma introduction message
    print("Emma : " + welcome_member(user_dict.get("first_time"), user_dict.get("name")) + " ")

    # Welcome back returning user, or greet new user
    if user_dict.get("first_time") == "false":
        print("Emma : " + feeling_response(user_dict.get("feeling")))  # last time we spoke, you were feeling...
    print("> ", end="")
    user_response = input()
    user_dict["feeling"] = user_response  # Update feeling in user_dict
    print(emotional_response(user_response))

    while flag:
        print("> ", end="")
        user_response = input()
        user_response = user_response.lower()
        if user_response not in ['bye', 'shutdown', 'exit', 'quit']:
            if user_response == 'thanks' or user_response == 'thank you':
                print("Emma : You are welcome.. What else is on your mind?")
            else:
                if hello_message(user_response, user_dict.get("name")) is not None:
                    print("Emma : " + hello_message(user_response, user_dict.get("name")))
                    welcome_bool = True
                else:
                    if welcome_bool:
                        print(emotional_response(user_response))
                        welcome_bool = False
                    else:
                        if fact_bool:
                            print("Emma : Would you like to know why this could be happening?", end="")
                            print()
                            print("> ", end="")
                            response = input()
                            response = response.lower()
                            print("Emma : ", end="")
                            if response not in ["no", "no thanks", "no thank you", "nope", "nah"]:
                                print_lines(combined_fact)
                                print()
                            else:
                                print("OK, what else is on your mind?")
                            fact_bool = False
                        else:
                            print("Emma : ", end="")
                            user_dict["problem"] = user_response
                            print_lines(generate_response(user_response, combined_list))
                            print()
                            combined_fact = get_best_answer(combined_facts_list, user_response)
                            if combined_fact != "":
                                fact_bool = True

        else:
            flag = False
            print("Emma : Till next time, hope I helped. Here is an affirmation for you:" + "\n")
            print_lines(random.choice(affirmations_list))
            print("\n")
            # dump
            with open(user_name + ".pickle", 'wb') as handle:
                pickle.dump(user_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

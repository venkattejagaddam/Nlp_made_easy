from urllib.request import urlopen
from bs4 import BeautifulSoup
import heapq
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk
from heapq import nlargest
from string import punctuation
from spacy.lang.en.stop_words import STOP_WORDS
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from gensim.summarization.summarizer import summarize
from textblob import TextBlob
import spacy
import streamlit as st
from spacy import displacy
from PIL import Image
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

# Web Scraping Pkg


# NLP pkgs

# sumy pkg

# spacy packages
# Pkgs for Normalizing Text
# Import Heapq for Finding the Top N Sentences

# nltk packages


# sumy function for summarizer
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence)for sentence in summary]
    result = ' '.join(summary_list)
    return result


# spacy summarizzer
def spacy_summarizer(raw_docx):
    nlp = spacy.load('en_core_web_sm')
    raw_text = raw_docx
    docx = nlp(raw_text)
    stopwords = list(STOP_WORDS)
    # Build Word Frequency # word.text is tokenization in spacy
    word_frequencies = {}
    for word in docx:
        if word.text not in stopwords:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    # Sentence Tokens
    sentence_list = [sentence for sentence in docx.sents]

    # Sentence Scores
    sentence_scores = {}
    for sent in sentence_list:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if len(sent.text.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]

    summarized_sentences = nlargest(
        7, sentence_scores, key=sentence_scores.get)
    final_sentences = [w.text for w in summarized_sentences]
    summary = ' '.join(final_sentences)
    return summary

# nltk summarizer


def nltk_summarizer(raw_text):
    stopWords = set(stopwords.words("english"))
    word_frequencies = {}
    for word in nltk.word_tokenize(raw_text):
        if word not in stopWords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    sentence_list = nltk.sent_tokenize(raw_text)
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(
        7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    return summary
# tokens and lemmas


def text_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)

    # tokens = [token.text for token in docx]
    allData = [('"Tokens":{},\n"Lemma":{}'.format(
        token.text, token.lemma_))for token in docx]
    return allData

# entity analyzer


# def entity_analyzer(my_text):
#     nlp = spacy.load('en')
#     docx = nlp(my_text)
#     entities = [(entity.text, entity.label_)for entity in docx.ents]
#     return entities


# get text from url
def get_text(raw_url):
    page = urlopen(raw_url)
    soup = BeautifulSoup(page)
    fetched_text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return fetched_text

# time estimation


def readingTime(mytext):
    nlp = spacy.load('en_core_web_sm')
    total_words = len([token.text for token in nlp(mytext)])
    estimatedTime = total_words/200.0
    return estimatedTime


def main():
    """ NLP APP WITH STREAMLIT"""
    st.title("NLP Made Easy")
    #st.subheader("Natural Language Processing on the Go")

    # Sidebar
    side_select = st.sidebar.radio(
        "Choose your option", ("Tokenization", "NER_by_Given_Text", "NER_by_url", "Sentiment_Analysis", "Summarizer"))
    # tokenization
    if(side_select == "Tokenization"):
        st.subheader("Tokenization of given text")
        message = st.text_area("Enter your text", "Type Here")
        if st.button("Start Analyzation"):
            nlp_result = text_analyzer(message)
            st.json(nlp_result)

    # entity
    elif(side_select == "NER_by_Given_Text"):
        nlp = spacy.load('en_core_web_sm')
        st.subheader("Named Entity Recognition from given Text")
        raw_text = st.text_area("Enter Text Here", "Type Here")
        if st.button("Analyze"):
            docx = nlp(raw_text)
            html = displacy.render(docx, style="ent")
            html = html.replace("\n\n", "\n")
            st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
        # st.markdown("Extract entities from your Text")
        # message = st.text_area("Enter ur text", "Type Here")
        # if st.button("Extract"):
        #     nlp_result = entity_analyzer(message)
        #     st.json(nlp_result)

    # NER by url
    elif(side_select == "NER_by_url"):
        nlp = spacy.load('en_core_web_sm')
        st.subheader("Analysis on Text From URL")
        raw_url = st.text_input("Enter URL Here", "Type here")
        text_preview_length = st.slider("Length to Preview", 50, 100)
        if st.button("Analyze"):
            if raw_url != "Type here":
                result = get_text(raw_url)
                len_of_full_text = len(result)
                len_of_short_text = round(len(result)/text_preview_length)
                st.success("Length of Full Text::{}".format(len_of_full_text))
                st.success("Length of Short Text::{}".format(
                    len_of_short_text))
                st.info(result[:len_of_short_text])
                summarized_docx = spacy_summarizer(result)
                docx = nlp(summarized_docx)
                html = displacy.render(docx, style="ent")
                html = html.replace("\n\n", "\n")
                st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
    # sentiment
    elif(side_select == "Sentiment_Analysis"):
        st.subheader("Sentiment of your Text")
        message = st.text_area("Enter text", "Type Here")
        if st.button("Analyze"):
            blob = TextBlob(message)
            result_sentiment = blob.sentiment
            # st.success(result_sentiment)
            st.write("polarity:", result_sentiment.polarity)
            st.write("subjectivity:", result_sentiment.subjectivity)
            image = Image.open('1.png')
            st.image(image, width=300)

    # summarizer
    elif(side_select == "Summarizer"):
        st.subheader("Summarization of your Text")
        message = st.text_area("Enter text", "Type Here")
        # summary_options = st.selectbox(
        #   "Choose your summarizer", ("gensim", "sumy", "spacy", "nltk"))
        # if st.button("Summarize"):
        st.markdown(
            "Select any of the summarizer or select multiple to compare: ")
        if st.checkbox('sumy'):
            st.text("Using sumy...")
            summary_result = sumy_summarizer(message)
            st.success(summary_result)
            st.write("Time required to read summarized text:",
                     readingTime(summary_result), "min")
        #if st.checkbox('gensim'):
         #   st.text("Using gensim summarizer")
          #  summary_result = summarize(message)
           # st.success(summary_result)
            #st.write("Time required to read summarized text:",
             #        readingTime(summary_result), "min")
        if st.checkbox('spacy'):
            st.text("Using spacy summarizer...")
            summary_result = spacy_summarizer(message)
            st.success(summary_result)
            st.write("Time required to read summarized text:",
                     readingTime(summary_result), "min")
        if st.checkbox('nltk'):
            st.text("Using spacy summarizer...")
            summary_result = nltk_summarizer(message)
            st.success(summary_result)
            st.write("Time required to read summarized text:",
                     readingTime(summary_result), "min")


if __name__ == '__main__':
    main()

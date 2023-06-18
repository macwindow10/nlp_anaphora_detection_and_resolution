import os
import io
import random
from flask import Flask, render_template, Response, send_file
import nltk
from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget
from nltk import pos_tag, word_tokenize, RegexpParser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from gensim import models
import spacy
from spacy import displacy
import visualise_spacy_tree
from spacy import displacy

matplotlib.use('Agg')

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# load w2v from pre-built Google data
w2v = models.word2vec.Word2Vec()
# download bin.gz from: https://code.google.com/archive/p/word2vec/
w2v = models.KeyedVectors.load_word2vec_format(
    "D:\\pythonprojects\\GoogleNews-vectors-negative300\\GoogleNews-vectors-negative300.bin",
    binary=True)
w2v_vocab = set(w2v.index_to_key)
print("Loaded {} words in vocabulary".format(len(w2v_vocab)))


def remove_stop_words(word_tokens, stop_words):
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


def words_similarity_matrix(nouns):
    words = nouns
    similarities = np.zeros((len(words), len(words)), dtype=np.float_)
    for idx1, word1 in enumerate(words):
        for idx2, word2 in enumerate(words):
            # note KeyError is possible if word doesn't exist
            # print('word1: ', word1);
            # print('word2: ', word2);
            sim = w2v.similarity(word1, word2)
            similarities[idx1, idx2] = sim

    df = pd.DataFrame.from_records(similarities, columns=words)
    df.index = words

    f, ax = plt.subplots(1, 1, figsize=(10, 6))
    cmap = plt.cm.Blues
    mask = np.zeros_like(df)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(df, cmap=cmap, mask=mask, square=True, ax=ax)
    _ = plt.yticks(rotation=90)
    plt.xlabel('Words')
    _ = plt.xticks(rotation=45)
    _ = plt.title("Similarities between words")
    # plt.show()
    return f


def generate_features_extraction_trees(input_sentence):
    sentences = input_sentence.split('.')
    c = 1
    for sentence in sentences:
        if len(sentence) < 10:
            continue
        print('sentence: ', sentence)
        sentence_word_tokens = word_tokenize(sentence)
        sentence_pos_tagged = nltk.pos_tag(sentence_word_tokens)
        # print(f'sentence pos tagged: {sentence_pos_tagged}')
        # Extract all parts of speech from any text
        chunker = RegexpParser("""
                                   NP: {<DT>?<JJ>*<NN>}
                                   P: {<IN>}           
                                   V: {<V.*>}          
                                   PP: {<p> <NP>}          
                                   VP: {<V> <NP|PP>*}
                                   """)
        output = chunker.parse(sentence_pos_tagged)
        output_str = str(output)
        # print("After Extracting\n", output_str)
        cf = CanvasFrame()
        t = Tree.fromstring(output_str)
        tc = TreeWidget(cf.canvas(), t)
        cf.add_widget(tc, 10, 10)  # (10,10) offsets
        cf.print_to_file('tree_{}.ps'.format(c))
        # cf.destroy()
        # output.draw()
        # os.system("gswin64c -sDEVICE=pdfwrite -o tree.pdf tree.ps")
        os.system(
            "gswin64c -dBATCH -dEPSCrop -dEPSFitPage -sDEVICE=png16m -r300  -dNOPAUSE -o tree_{}.png tree_{}.ps".format(
                c, c))
        c = c + 1


# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    input_sentence = "The system must provide a user-friendly interface for creating and editing documents. " \
                     "Users should be able to easily navigate to the editing tool they need. " \
                     "This will help to ensure that document can be created quickly and efficiently." \
                     " It should also provide access to a range of fonts and formatting options. "

    generate_features_extraction_trees(input_sentence)

    stop_words = set(stopwords.words('english'))
    # print(f'stop_words: {stop_words}')

    word_tokens = word_tokenize(input_sentence)
    # print(f'word_tokens: {word_tokens}')

    filtered_sentence = remove_stop_words(word_tokens, stop_words)
    # print(f'filtered_sentence: {filtered_sentence}')

    pos_tagged = nltk.pos_tag(filtered_sentence)
    # print(f'pos_tagged: {pos_tagged}')

    pronouns = []
    nouns = []
    pronouns_nouns = []
    for item in pos_tagged:
        # print(item[1])
        if item[1] == 'NN':
            nouns.append(item[0])
        elif item[1] == 'PRP':
            pronouns.append(item[0])

    pronouns_nouns = pronouns + nouns
    pronouns_nouns = [item.lower() for item in pronouns_nouns]

    print(f'nouns: {nouns}')
    print(f'pronouns: {pronouns}')
    print(f'pronouns and nouns: {pronouns_nouns}')

    # Prepare your training data and labels
    X_train = ["The system must provide a user-friendly interface for creating and editing documents.",
               "Users should be able to easily navigate to the editing tool they need.",
               "This will help to ensure that document can be created quickly and efficiently.",
               "It should also provide access to a range of fonts and formatting options."]
    y_train = [0, 0, 0, 1]  # 0 represents nouns, 1 represents "it"

    # Convert training data to feature vectors
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)

    # Prepare your test data
    X_test = ["It should also provide access to a range of fonts and formatting options."]

    # Convert test data to feature vectors
    X_test_vectorized = vectorizer.transform(X_test)

    # Perform predictions
    y_pred = model.predict(X_test_vectorized)
    # Print the predicted labels
    print("Predicted labels:", y_pred)

    # Print the corresponding pronouns
    predicted_nouns = [pronouns_nouns[label] for label in y_pred]
    print("Predicted nouns:", predicted_nouns)

    resolved_sentence = ''
    for sentence in X_train:
        # print(sentence)
        words = sentence.split(' ')
        for word in words:
            if word.strip().capitalize() in pronouns:
                resolved_sentence = resolved_sentence + \
                                    ' <mark style="background-color: yellow; color: black;">' + \
                                    predicted_nouns[0] + '</mark>'
            else:
                resolved_sentence = resolved_sentence + ' ' + word.strip()
        resolved_sentence = resolved_sentence + '\r\n'
    print(f'resolved_sentence: {resolved_sentence}')

    return render_template('index.html',
                           input_sentence=input_sentence,
                           word_tokens=word_tokens,
                           filtered_sentence=filtered_sentence,
                           pos_tagged=pos_tagged,
                           nouns=nouns,
                           pronouns=pronouns,
                           predicted_nouns=predicted_nouns,
                           resolved_sentence=resolved_sentence)


@app.route('/features_extraction_tree.png/<id>', methods=['GET'])
def features_extraction_tree_png(id):
    return send_file("tree_{}.png".format(id), mimetype='image/gif')


@app.route('/plot_words_similarity_matrix.png/<nouns>', methods=['GET'])
def plot_words_similarity_matrix_png(nouns):
    # fig = create_figure()
    print('nouns: ', nouns)
    list = nouns.replace("[", "") \
        .replace("]", "") \
        .replace("'", "") \
        .replace(" ", "") \
        .split(",")
    print('list: ', list)
    fig = words_similarity_matrix(list)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)
    return fig


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()

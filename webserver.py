import io
import random
from flask import Flask, render_template, Response
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from main import remove_stop_words, words_similarity_matrix
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    input_sentence = "The system must provide a user-friendly interface for creating and editing documents. " \
                     "Users should be able to easily navigate to the editing tool they need. " \
                     "This will help to ensure that document can be created quickly and efficiently." \
                     " It should also provide access to a range of fonts and formatting options. "

    stop_words = set(stopwords.words('english'))
    print(f'stop_words: {stop_words}')

    word_tokens = word_tokenize(input_sentence)
    print(f'word_tokens: {word_tokens}')

    filtered_sentence = remove_stop_words(word_tokens, stop_words)
    print(f'filtered_sentence: {filtered_sentence}')

    pos_tagged = nltk.pos_tag(filtered_sentence)
    print(f'pos_tagged: {pos_tagged}')

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
                resolved_sentence = resolved_sentence + ' ' + predicted_nouns[0]
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


@app.route('/plot_words_similarity_matrix.png')
def plot_words_similarity_matrix_png():
    # fig = create_figure()
    fig = words_similarity_matrix(["Coca_Cola", "Pepsi", "pepsi", "cola", "Microsoft"])
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

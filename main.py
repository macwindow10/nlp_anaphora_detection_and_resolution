import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from gensim import models
import numpy as np
from sklearn.manifold import TSNE


# load w2v from pre-built Google data
w2v = models.word2vec.Word2Vec()
# download bin.gz from: https://code.google.com/archive/p/word2vec/
w2v = models.KeyedVectors.load_word2vec_format(
    "D:\\pythonprojects\\GoogleNews-vectors-negative300\\GoogleNews-vectors-negative300.bin",
    binary=True)
w2v_vocab = set(w2v.index_to_key)
print("Loaded {} words in vocabulary".format(len(w2v_vocab)))

raw_words_of_interest = ['Coke', 'Pepsi', 'cola', 'drink',
                         'cool', 'swim', 'swimming', 'thirst',
                         'Microsoft', 'Oracle',
                         'smartphone', 'cruise']
# some other random stuff we chould throw in...
#                    'King', 'Queen', 'person', 'walking', 'dancing', 'news', 'food', 'kitchen', 'house']

words_of_interest = []
for woi in raw_words_of_interest:
    for word, _ in w2v.most_similar(woi):
        words_of_interest.append(word)

words_of_interest = list(set(words_of_interest))

vectors = []
for word in set(words_of_interest):
    vectors.append(w2v[word])

vectors = np.vstack(vectors)  # turn vectors into a 2D array <words x 300dim>

model = TSNE(n_components=2, random_state=0)
X_tsne = model.fit_transform(vectors)
df_after_tsne = pd.DataFrame.from_records(X_tsne, columns=['x', 'y'])
df_after_tsne['labels'] = words_of_interest

# calculate similarity from a target word to all words, to use as our colour
target_word = "smartphone"
similarities = []
for woi in words_of_interest:
    similarity = min(max(0, w2v.similarity(target_word, woi)), 1.0)
    similarities.append(similarity)

# plot the T-SNE layout for words, darker words means more similar to our target
plt.figure(figsize=(12, 8))
plt.xlim((min(X_tsne[:, 0]), max(X_tsne[:, 0])))
plt.ylim((min(X_tsne[:, 1]), max(X_tsne[:, 1])))
for idx in range(X_tsne.shape[0]):
    x, y = X_tsne[idx]
    label = words_of_interest[idx]
    color = str(min(0.6, 1.0 - similarities[idx]))  # convert to string "0.0".."1.0" as greyscale for mpl
    # plt.annotate(s=label, xy=(x, y), color=color)
    # plt.annotate(s=label, xy=(x, y), weight=int(similarities[idx]*1000)) # use weight

plt.tight_layout()
_ = plt.title("Word similarity (T-SNE) using vectors from {} words\nColoured by similarity to '{}'".format(
        len(words_of_interest),
        target_word))
plt.show()

'''
if __name__ == '__main__':
    example_1 = "The system must provide a user-friendly interface for creating and editing documents. Users should " \
                "be able to easily navigate to the editing tool they need. This will help to ensure that document can " \
                "be created quickly and efficiently. It should also provide access to a range of fonts and formatting " \
                "options. "
    print(f'example_1: {example_1}')

    stop_words = set(stopwords.words('english'))
    print(f'stop_words: {stop_words}')

    word_tokens = word_tokenize(example_1)
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
predicted_pronouns = [pronouns_nouns[label] for label in y_pred]
print("Predicted pronouns:", predicted_pronouns)

resolved_sentence = ''
for sentence in X_train:
    # print(sentence)
    words = sentence.split(' ')
    for word in words:
        if word.strip().capitalize() in pronouns:
            resolved_sentence = resolved_sentence + ' ' + predicted_pronouns[0]
        else:
            resolved_sentence = resolved_sentence + ' ' + word.strip()
    resolved_sentence = resolved_sentence + '\r\n'
print(f'resolved_sentence: {resolved_sentence}')
'''

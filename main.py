import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def remove_stop_words(word_tokens, stop_words):
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence

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
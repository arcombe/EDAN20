import transition
import dparser
import conll
import time
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV


def extract(feature_names, sentences):
    X_l = []
    y_l = []

    for sentence in sentences:
        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'

        X, y = extract_features_sent(stack, queue, graph, feature_names, sentence, True)
        X_l.extend(X)
        y_l.extend(y)

    return X_l, y_l


def extract_features_sent(stack, queue, graph, feature_names, sentence, train):
    # We pad the sentence to extract the context window more easily

    X = list()
    y = list()

    while queue:
        prev_stack = stack
        prev_queue = queue
        stack, queue, graph, trans = dparser.reference(stack, queue, graph)

        x = list()
        for i in feature_names:
            if (i == 'stack0_POS'):
                if len(prev_stack) >= 1:
                    x.append(prev_stack[0]['postag'])
                else:
                    x.append('nill')
            elif (i == 'stack1_POS'):
                if len(prev_stack) >= 2:
                    x.append(prev_stack[1]['postag'])
                else:
                    x.append('nill')
            elif (i == 'stack2_POS'):
                if len(prev_stack) >= 3:
                    x.append(prev_stack[2]['postag'])
                else:
                    x.append('nill')
            elif (i == 'stack0_word'):
                if len(prev_stack) >= 1:
                    x.append(prev_stack[0]['form'])
                else:
                    x.append('nill')
            elif (i == 'stack1_word'):
                if len(prev_stack) >= 2:
                    x.append(prev_stack[1]['form'])
                else:
                    x.append('nill')
            elif (i == 'stack2_word'):
                if len(prev_stack) >= 3:
                    x.append(prev_stack[2]['form'])
                else:
                    x.append('nill')
            elif (i == 'queue0_POS'):
                if len(prev_queue) >= 1:
                    x.append(prev_queue[0]['postag'])
                else:
                    x.append('nill')
            elif (i == 'queue1_POS'):
                if len(prev_queue) >= 2:
                    x.append(prev_queue[1]['postag'])
                else:
                    x.append('nill')
            elif (i == 'queue2_POS'):
                if len(prev_queue) >= 3:
                    x.append(prev_queue[2]['postag'])
                else:
                    x.append('nill')
            elif (i == 'queue0_word'):
                if len(prev_queue) >= 1:
                    x.append(prev_queue[0]['form'])
                else:
                    x.append('nill')
            elif (i == 'queue1_word'):
                if len(prev_queue) >= 2:
                    x.append(prev_queue[1]['form'])
                else:
                    x.append('nill')
            elif (i == 'queue2_word'):
                if len(prev_queue) >= 3:
                    x.append(prev_queue[2]['form'])
                else:
                    x.append('nill')
            elif (i == 'can-re'):
                if len(prev_stack) >= 1:
                    x.append(True)
                else:
                    x.append(False)
            elif (i == 'can-la'):
                if len(prev_stack) >= 1 and len(prev_queue) >= 1:
                    x.append(True)
                else:
                    x.append(False)

        X.append(dict(zip(feature_names, x)))
        y.append(trans)
        if (not train):
            break

    return X, y


def predict(test_sentences, feature_names, f_out):
    for test_sentence in test_sentences:

        X_test_dict, y_test = extract_features_sent(test_sentence, w_size, feature_names, False)
        y_test_predicted = []
        for x_test in X_test_dict:
            if len(y_test_predicted) == 1:
                x_test['c_n1'] = y_test_predicted[-1]
            elif len(y_test_predicted) > 1:
                x_test['c_n2'] = y_test_predicted[-2]
                x_test['c_n1'] = y_test_predicted[-1]
            # Vectorize the test sentence and one hot encoding
            # print('x:', x_test)
            # print('y_hat:', y_test_predicted)
            x_test = vec.transform(x_test)
            # Predicts the chunks and returns numbers
            y_test_predicted.append(classifier.predict(x_test)[0])
        # Appends the predicted chunks as a last column and saves the rows
        rows = test_sentence.splitlines()
        rows = [rows[i] + ' ' + y_test_predicted[i] for i in range(len(rows))]
        for row in rows:
            f_out.write(row + '\n')
        f_out.write('\n')
        # exit()
    f_out.close()


if __name__ == '__main__':
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']

    feature_names_1 = ['stack0_POS', 'stack0_word', 'queue0_POS', 'queue0_word', 'can-re', 'can-la']
    feature_names_2 = ['stack0_POS', 'stack1_POS', 'stack0_word', 'stack1_word', 'queue0_POS', 'queue1_POS',
                       'queue0_word', 'queue1_word', 'can-re', 'can-la']
    feature_names_3 = ['stack0_POS', 'stack1_POS', 'stack2_POS', 'stack0_word', 'stack1_word', 'stack2_word',
                       'queue0_POS', 'queue1_POS', 'queue2_POS',
                       'queue0_word', 'queue1_word', 'queue2_word', 'can-re', 'can-la']

    train_file = './train.conll'
    test_file = './test.conll'
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']

    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)



    X_dict, y = extract(feature_names_1, formatted_corpus)

    print("Encoding the features...")
    # Vectorize the feature matrix and carry out a one-hot encoding
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dict)
    # The statement below will swallow a considerable memory
    # X = vec.fit_transform(X_dict).toarray()
    # print(vec.get_feature_names())

    training_start_time = time.clock()
    print("Training the model...")
    #classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear')
    classifier = tree.DecisionTreeClassifier()
    #classifier = linear_model.Perceptron(penalty='l2')
    model = classifier.fit(X, y)
    print(model)

    test_start_time = time.clock()
    # We apply the model to the test set
    test_sentences = list(conll_reader.read_sentences(test_corpus))

    # Here we carry out a chunk tag prediction and we report the per tag error
    # This is done for the whole corpus without regard for the sentence structure
    print("Predicting the chunks in the test set...")
    X_test_dict, y_test = extract_features(test_sentences, w_size, feature_names)
    # Vectorize the test set and one-hot encoding
    X_test = vec.transform(X_test_dict)  # Possible to add: .toarray()
    y_test_predicted = classifier.predict(X_test)
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, y_test_predicted)))

    # Here we tag the test set and we save it.
    # This prediction is redundant with the piece of code above,
    # but we need to predict one sentence at a time to have the same
    # corpus structure
    print("Predicting the test set...")
    f_out = open('out', 'w')
    predict(test_sentences, feature_names, f_out)

    end_time = time.clock()
    print("Training time:", (test_start_time - training_start_time) / 60)
    print("Test time:", (end_time - test_start_time) / 60)




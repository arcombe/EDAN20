import regex as re
import sys
import pickle
import math

def unigrams(words):
    dict = pickle.load(open('unigrams.idx', 'rb'))
    sum = {}
    count = 0

    for key in dict:
        count = count + dict[key]


    print('Unigram model')
    print('=========================')
    print('wi C(wi) #words P(wi)')
    print('=========================')

    for word in words:
        cwi = dict[word]
        sum[word] = cwi / count
        print(word + ' ' + str(cwi) + ' ' + str(count) + ' ' + str(sum[word]))

    print('=========================')
    prod_uni = 1

    for key in sum:
        prod_uni = prod_uni * sum[key]

    entropy = -1 * math.log(prod_uni, 2) / len(sum)

    print('Prod. unigrams:   ' + str(prod_uni))
    print('Geometric mean prob.:     ' + str(prod_uni ** (1/len(sum))))
    print('Entropy rate:     ' + str(entropy))
    print('Perplexity:    ' + str(2 ** entropy))


def unigrams_2(words):
    dict = pickle.load(open('unigrams.idx', 'rb'))
    count = 0

    for key in dict:
        count = count + dict[key]

    return dict[words[0]] / count



def bigrams(words):
    dict_bi = pickle.load(open('bigrams.idx', 'rb'))
    dict_uni =pickle.load(open('unigrams.idx', 'rb'))
    sum = {}

    print('Bigram model')
    print('=============================================================================')
    print('wi    wi+1    Ci,i+1   C(i)  P(wi+1|wi)')
    print('=============================================================================')


    for x in range(len(words)):
        if x < len(words) - 1:
            word_1 = words[x]
            word_2 = words[x + 1]
            uni = dict_uni[word_1]
            tuple = (word_1, word_2)
            try:
                bi = dict_bi[tuple]
                p = bi / uni
                sum[tuple] = p
            except KeyError:
                bi = 0
                p = 0
                extra = unigrams_2([word_2])
                sum[tuple] = extra

            if p == 0:
                print(word_1 + ' ' + word_2 + ' ' + str(bi) + ' ' + str(uni) + ' ' + str(p) + ' *backoff: ' + str(extra))
            else:
                print(word_1 + ' ' + word_2 + ' ' + str(bi) + ' ' + str(uni) + ' ' + str(p))

    print('=============================================================================')

    prod_uni = 1

    for key in sum:
        prod_uni = prod_uni * sum[key]

    entropy = -1 * math.log(prod_uni, 2) / len(sum)

    print('Prod. unigrams:   ' + str(prod_uni))
    print('Geometric mean prob.:     ' + str(prod_uni ** (1 / len(sum))))
    print('Entropy rate:     ' + str(entropy))
    print('Perplexity:    ' + str(2 ** entropy))

if __name__ == '__main__':

    text = sys.stdin.read()

    words = re.findall('\p{L}+', text)

    unigrams(words)
    print ()
    bigrams(words)


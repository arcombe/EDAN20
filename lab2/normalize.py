from programs.tokenizer import *
import regex as re

def normalize(text):
    words = tokenize3(text)
    text = ''

    on_sentance = False
    sentence = ''

    for word in words:
        m = re.match(r'[A-Ö]', word)
        if not on_sentance and m != None and len(m.group()) == 1:
            on_sentance = True
            sentence = '<s> ' + word
        elif on_sentance and re.match('[.!?]', word):
            on_sentance = False
            sentence = sentence + ' </s>'
            text = text + sentence
        elif on_sentance and re.match('\p{L}', word):
            sentence = sentence + ' ' + word

    return text.lower()



if __name__ == '__main__':
    text = sys.stdin.read()
    text = normalize(text)
    print(text)
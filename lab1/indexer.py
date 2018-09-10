import os
import re
import pickle
import sys
import math

def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    :param dir:
    :param suffix:
    :return: the list of file names
    """

    files = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            files.append(file)
    return files

def get_word(word):

    return re.sub('[^A-Öa-ö]+', '', word).lower()

def index_file(file):

    print(file)
    if not os.path.exists(file):
        return

    text = ''
    dict = {}

    word_count = 0
    with open(file, 'r') as f:

        for line in f:
            text = text+line

            words = line.split(' ')
            for word in words:
                key = get_word(word)
                if key not in dict:
                    dict[key] = []
                word_count = word_count + 1

    print(word_count)
    print(len(dict))

    quit(0)

    max = len(text)

    text = text.lower()



    for key in dict.keys():
        for m in re.finditer(key, text):
            if m.start() != 0 and re.match('[^A-Öa-ö]+', text[m.start() - 1]) != None \
                    and m.end() < max and re.match('[^A-Öa-ö]+', text[m.end()]) != None:

                dict[key].append(m.start())

    pickle.dump(dict, open(file.replace('txt', 'idx'), "wb"))

def create_master(root, files):

    master = {}

    for file in files:
        f = os.path.join(root, file.replace('txt', 'idx'))
        dict = pickle.load(open(f, "rb"))
        for key in dict:
            if key not in master:
                master[key] = {}

            master[key][file] = dict[key]

    pickle.dump(master, open('master.idx', "wb"))

def test_master(root):
    dict = pickle.load(open('master.idx', 'rb'))
    print(dict['samlar'])
    print(dict['ände'])

def tf_idf(master, term, files, N):
    nbr_files_with_term = len(master[term])

    sum_files = {}
    for file in files:
        sum_files[file] = 0
    for key in master:
        for f in master[key]:
            sum_files[f] = sum_files[f] + len(master[key][f])

    for file in files:
        if file in master[term]:
            print(len(master[term][file]))
            count = len(master[term][file]) / sum_files[file]
            res = count * math.log((N / nbr_files_with_term), 10)
        else:
            res = 0.0
        print(file + " " + str(res))


if __name__ == '__main__':

    root = sys.argv[1]

    files = get_files(root, '.txt')
    N = len(files)


    if not os.path.exists('master.idx'):

        for file in files:
            index_file(os.path.join(root, file))

        create_master(root, files)

    else:
        master = pickle.load(open('master.idx', 'rb'))
        tf_idf(master, 'gås', files, N)




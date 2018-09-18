import os
import regex as re
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

    if not os.path.exists(file):
        return

    text = ''
    dict = {}


    with open(file, 'r') as f:

        for line in f:
            text = text + line

            # for word in words:
            #     key = get_word(word)
            #     if key not in dict:
            #         dict[key] = []
            #     word_count = word_count + 1

    max = len(text)

    text = text.lower()

    for m in re.finditer('\p{L}+', text):
        if m.group() not in dict:
            dict[m.group()] = [m.start()]
        else:
            dict[m.group()].append(m.start())

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

def test_master():
    dict = pickle.load(open('master.idx', 'rb'))
    print(dict['samlar'])
    print(dict['ände'])

def tf_idf(master, term, file, N):
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

def tf_idf_2(master, term, file, N, sum):
    nbr_files_with_term = len(master[term])

    if file in master[term]:
        count = len(master[term][file]) / sum
        res = count * math.log((N / nbr_files_with_term), 10)
    else:
        res = 0.0

    return res

def dot(file_a, file_b):
    sum_a_b = 0
    sum_a = 0
    sum_b = 0

    for i in range(0, len(file_a)):
        sum_a_b = sum_a_b + file_a[i] * file_b[i]
        sum_a = sum_a + file_a[i] * file_a[i]
        sum_b = sum_b + file_b[i] * file_b[i]

    return sum_a_b / (math.sqrt(sum_a) * math.sqrt(sum_b))

def cos_sim(master, files, N):

    dict = {}

    sum = {}

    for file in files:
        sum[file] = 0

    for key in master:
        for file in files:
            if file in master[key]:
                sum[file] = sum[file] + len(master[key][file])

    for file in files:
        dict[file] = []

    for key in master:
        for file in files:
            dict[file].append(tf_idf_2(master, key, file, N, sum[file]))

    for file_a in files:
        for file_b in files:
            print(dot(dict[file_a], dict[file_b]), file_a, file_b)




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
        #tf_idf(master, 'et', files, N)
        cos_sim(master, files, N)




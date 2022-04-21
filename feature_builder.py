import pandas as pd
import time
from multiprocessing import Process
from multiprocessing import Manager
import numpy as np
from pandarallel import pandarallel
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
pandarallel.initialize()

class FeatureBuilderBase():
    def __init__(self):
        self.module_name = "FeatureBuilderBase"

    def initialize(self, params):
        for key, value in params:
            setattr(self, key, value)

    # user-index map: user_name --> i
    def get_user2index(self, data):
        userMap = dict()
        for i, user in enumerate(data.name.unique()):
            userMap[user] = i
        return userMap

    # tweet-index map: tweet_text --> j
    def get_tweet2index(self, data):
        tweetMap = dict()
        for i, tweet in enumerate(data.postTweet.unique()):
            tweetMap[tweet] = i
        return tweetMap

    # construct user-tweet matrix: userTweet[i, j] = cnt
    def get_bimatrix(self, user2index, tweet2index, data):
        userTweet = np.zeros((len(user2index), len(tweet2index)))
        for user, tweet in data[['name', 'postTweet']].iloc[::-1].values:
            userTweet[user2index[user], tweet2index[tweet]] += 1
        return userTweet

class MFFeatureBuilder(FeatureBuilderBase):
    def __init__(self, processed_data, mode, num_process=40):
        super(MFFeatureBuilder).__init__()
        self.data = processed_data
        self.build_key_matrix(self.data)
        self.mode = mode
        self.num_process = num_process

        self.supporting_modes = ["multiply", "m_smooth"]
        if self.mode not in self.supporting_modes:
            raise NotImplementedError("Only support modes: {}".format(self.supporting_modes))

    # construct keyword list {keywords}
    def get_keywordList(self, data):
        keywordList = []
        for tweet in data.postTweet:
            keywordList += tweet.split()
        keywordList = set(keywordList)
        return keywordList

    # get user-key matrix: user --> keywords count
    def get_user2keywords(self, data, keyword_list, user):
        tempKey = []
        tempCount = dict.fromkeys(keyword_list, 0)
        for tweet in data[data.name == user].postTweet:
            tempKey += tweet.split()

        for word in set(tempKey):
            tempCount[word] = tempKey.count(word)
        return list(tempCount.values())

    # construct user-keyword Matrix: for every user
    def get_users2keywords(self, user2index, data, keyword_list):
        userKey = pd.DataFrame(list(user2index.keys()), columns=['name'])
        tic = time.time()
        userKey['dist'] = userKey.name.parallel_apply(lambda x: self.get_user2keywords(data, keyword_list, x))
        userKey = np.array(userKey.dist.values.tolist())
        return userKey

    # get tweet-key matrix: tweet --> keywords count
    def get_tweet2keywords(self, keyword_list, tweet):
        tempKey = tweet.split()
        tempCount = dict.fromkeys(keyword_list, 0)

        for word in set(tempKey):
            tempCount[word] = tempKey.count(word)
        return list(tempCount.values())

    # construct tweet-keyword Matrix: for every tweet
    def get_tweets2keywords(self, tweet2index, keyword_list):
        tweets2keywords = pd.DataFrame(list(tweet2index.keys()), columns=['tweet'])
        tic2 = time.time()
        tweets2keywords['dist'] = tweets2keywords.tweet.parallel_apply(lambda x: self.get_tweet2keywords(keyword_list, x))
        tweets2keywords = np.array(tweets2keywords.dist.values.tolist())
        return tweets2keywords

    # interpolation function
    def phi(self, nz_index, tweet2index, index, r):
        if index in nz_index:
            return 1
        s = 0
        for i in nz_index:
            s += np.exp(- r * np.linalg.norm(np.array(tweet2index[index, :]) - np.array(tweet2index[i, :]), 2) ** 2) / 4
        if s < 0.2:
            return 0
        else:
            return s

    # son-process function
    def interpolation(self, result, bimatrix, tweets2keywords, k, num_process):
        for i in range(bimatrix.shape[0]):
            if i % num_process == k:
                print('process', k, '{} / {}'.format(i, bimatrix.shape[0]))
                nz_index = np.where(bimatrix[i, :] > 0)[0]
                index = np.arange(bimatrix.shape[1])
                result[i] = np.vectorize(self.phi, \
                                         excluded=['nz_index', 'tweetKey'])(nz_index=nz_index,
                                                                            tweet2index=tweets2keywords,
                                                                            index=index, r=2)

    # Message Similarity (M-Module): Second Step
    def Mmodule(self, bimatrix, tweets2keywords, num_process):
        """
        K assigns the number of processes
        """
        manager = Manager()
        result = manager.dict()

        plist = []
        for k in range(num_process):
            temp = Process(target=self.interpolation,
                           args=(result, bimatrix, tweets2keywords, k, num_process))
            plist.append(temp)

        for i in plist:
            i.start()
        for i in plist:
            i.join()

        new_features = []
        for _, j in sorted(dict(result).items(), key=lambda x: x[0]):
            new_features.append(j)
        new_features = np.array(new_features)

        return new_features

    # First Step Process
    def build_key_matrix(self, data):
        # get Maps
        self.user2index, self.tweet2index = self.get_user2index(data), self.get_tweet2index(data)
        # get biMatrix
        self.userTweet = self.get_bimatrix(self.user2index, self.tweet2index, data)
        # get keyList
        self.keyword_list = self.get_keywordList(data)
        # userKey
        self.users2keywords = self.get_users2keywords(self.user2index, data, self.keyword_list)
        # tweetKey
        self.tweets2keywords = self.get_tweets2keywords(self.tweet2index, self.keyword_list)
        # bimatrix
        self.bimatrix = self.get_bimatrix(self.user2index, self.tweet2index, data)

    def build_index_mapping_only(self):
        # get Maps
        self.user2index, self.tweet2index = self.get_user2index(self.data), self.get_tweet2index(self.data)
        # get keyList
        self.keyword_list = self.get_keywordList(self.data)
        # userKey
        self.users2keywords = self.get_users2keywords(self.user2index, self.data, self.keyword_list)
        # tweetKey
        self.tweets2keywords = self.get_tweets2keywords(self.tweet2index, self.keyword_list)

    def build(self):
        self.build_key_matrix(self.data)
        if self.mode == "multiply":
            # normalize by 2-norm
            users2keywords = self.users2keywords / (self.users2keywords ** 2).sum(axis=1).reshape(-1, 1) ** 0.5
            tweets2keywords = self.tweets2keywords / (self.tweets2keywords ** 2).sum(axis=1).reshape(-1, 1) ** 0.5
            return users2keywords @ tweets2keywords.T
        elif self.mode == "m_smooth":
            features = self.Mmodule(self.bimatrix, self.tweets2keywords, self.num_process)
            return features
        else:
            raise NotImplementedError(self.module_name)


# wget http://nlp.stanford.edu/data/glove.6B.zip
# unzip glove.6B.zip

class TfidfEmbeddingVectorizer(object):
    def __init__(self, X):
        self.word2vec = self.build_w2v(X)
        self.word2weight = None
        if len(self.word2vec) > 0:
            self.dim = len(self.word2vec[next(iter(self.word2vec))])
        else:
            self.dim = 0
        self.fit(X)

    def build_w2v(self, X):
        GLOVE_6B_50D_PATH = "/Users/lijinning/PycharmProjects/Polarization/dataset/w2v/glove/glove.6B.50d.txt"
        glove_small = {}
        all_words = set(w for words in X for w in words)
        with open(GLOVE_6B_50D_PATH, "rb") as infile:
            for line in infile:
                parts = line.split()
                word = parts[0].decode("utf-8")
                if (word in all_words):
                    nums = np.array(parts[1:], dtype=np.float32)
                    glove_small[word] = nums
        return glove_small

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    def transform_one(self, x):
        return self.transform([x])[0]


if __name__ == "__main__":
    model = TfidfEmbeddingVectorizer([["happy", "deadline", "china"], ["what", "cheap", "chips"]])
    print(model.transform([["happy", "deadline", "china"], ["what", "cheap", "chips"]]))
    print(model.transform_one(["happy", "deadline", "china"]))







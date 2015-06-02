from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import LeaveOneOut as LOO
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import random
import pprint

class classifierObject:

    def __init__(self):
        #global data structures
        self.mapping = {'Recreation': 1, 'Transportation': 2, 'Business': 3,\
                    'Public-Safety': 4, 'Social-Services': 5, 'Environment': 6,\
                    'Health': 7, 'City-Government': 8, 'Education': 9, 'Housing-Development': 10}
        self.rmapping={v:k for k,v in self.mapping.items()}
        self.lines = [i.strip().split(',') for i in open('ny_dump','r').readlines()]
        random.shuffle(self.lines)
        #instantiate vectorizer
        self.vc = CV(token_pattern='[a-z]{2,}', binary=True) 
        #self.vc = TV(token_pattern='[a-z]{2,}', binary=True)
        #######################
        self.field = []
        self.label = []
        self.cate = []
        sum_all = {}
        sum_use = {}
        ctr = 0
        for l in self.lines:
            #if ctr==10: #why?
            #    break;
            if len(l)>6:
                self.field.append(' '.join(l[5:]))
                self.label.append(self.mapping[l[3]])
                self.cate.append(l[3])
            #    ctr += 1 #why?
        self._train()
        return

    def _train(self):
        label = np.array(self.label)
        self.vc = CV(analyzer='char_wb', ngram_range=(2,5), min_df=1, token_pattern='[a-z]{2,}')
        vector = self.vc.fit_transform(self.field).toarray()
        #print len(vc.get_feature_names())
        #print vc.get_feature_names()
        fold = 10
        #idx = LOO(len(vector))
        idx = StratifiedKFold(self.label, n_folds=fold)

        preds = []
        a_sum = []
        ctr = 0
        '''
        in general, from experiments we see:
            -using description>mixed>column names
            -GNB>MNB~=SBVM>RF
            -whole word>n-gram-ed vector
        '''
        #clf = DT(criterion='entropy', random_state=0)
        self.clf = RFC(n_estimators=100, criterion='entropy')
        #clf = GNB()
        #clf = MNB()
        #clf = SVC(C=0.1,kernel='linear')
        tmp = 0
        l = []
        p = [] #array to track the predictions with highest acc
        for train, test in idx:
            train_data = vector[train]
            train_label = label[train]
            test_data = vector[test]
            test_label = label[test]
            self.clf.fit(train_data, train_label)
            preds = self.clf.predict(test_data)
            #print test_data
            #print clf.predict_proba(test_data)
            #preds.append(clf.predict(test_data))
            #if pred != test_label:
            #    ctr += 1
            #    print 'inst', i+1, '%d:%d'%(test_label,pred)
            acc = accuracy_score(test_label, preds)
            a_sum.append(acc)
            print acc
            if acc>tmp:
                tmp = acc
                #l = test_label[:] #true label
                #p = preds[:]      #predicted label
        print 'ave acc:', np.mean(a_sum)
        print 'std:', np.std(a_sum)

    def predictLabel(self,data):
        vc = CV(vocabulary=self.vc.get_feature_names()) 
        sample = vc.fit_transform([data]).toarray()
        print sample
        return self.rmapping[self.clf.predict(sample)[0]]
        #return "none"

    def getLabelBruteForce(self, data):
        self.label = np.array(self.label)
        self.field.append(data)
        #vector = self.vc.fit_transform(self.field).toarray()
        idx = 0
        tag = self.label[0]
        tmp = vector[-1]
        dist = sum(abs(tmp-vector[0]))/sum(vector[0])
        i = 1
        for i in xrange(len(vector)-1):
            d = sum(abs(tmp-vector[i]))/sum(vector[i])
            print "label:" + self.rmapping[tag] + ", d=" + str(d) + ", min_dist=" + str(dist)
            if d<dist:
                dist = d
                idx = i
                tag = self.label[i]
        return self.rmapping[tag]

    def getTopKFeaturesAndCounts(self, idx):
        return
        

if __name__=="__main__":
    co = classifierObject()
    #print co.getLabelBruteForce("hello world what should i be writing here, i don't know")
    label= co.predictLabel("hello world what should i be writing here, i don't know")
    print label

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.cross_validation import LeaveOneOut as LOO
from sklearn.cross_validation import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import AdaBoostClassifier as Ada
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as CM
from sklearn import tree
from sklearn.preprocessing import normalize
import numpy as np
import pylab as pl

mapping = {'Recreation': 1, 'Transportation': 2, 'Business': 3, 'Public-Safety': 4, 'Social-Services': 5, 'Environment': 6, 'Health': 7, 'City-Government': 8, 'Education': 9, 'Housing-Development': 10}
lines = [i.strip().split(',') for i in open('ny_dump','r').readlines()]
data = []
label = []
sum_all = {}
sum_use = {}
for l in lines:
    '''
    1-agency
    2-name
    3-description
    4-category
    5-url id
    6:-field names
    '''
    #sum_all[l[3]] = sum_all.get(l[3],0) + 1
    if len(l)>6:
        #data.append(l[2]+' '+' '.join(l[5:]))
        #data.append(' '.join(l[5:]))
        data.append(l[2])
        label.append(mapping[l[3]])
        sum_use[l[3]] = sum_use.get(l[3],0) + 1
#print sum_all
print sum_use
'''
-break down all the datas into single words (?)
-vectorize the bag of words for each dataset
-train and testing
'''
label = np.array(label)
vc = CV(analyzer='char_wb', ngram_range=(2,5), min_df=1, token_pattern='[a-z]{2,}')
#vc = TV(analyzer='char_wb', ngram_range=(2,3), min_df=1, token_pattern='[a-z]{2,}')
#vc = CV(token_pattern='[a-z]{2,}')
#vc = TV(token_pattern='[a-z]{2,}', binary=True)
vector = vc.fit_transform(data).toarray()
#print len(vc.get_feature_names())
#print vc.get_feature_names()
fold = 10
#idx = LOO(len(vector))
idx = StratifiedKFold(label, n_folds=fold)

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
clf = RFC(n_estimators=50, criterion='entropy')
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
    clf.fit(train_data, train_label)
    preds = clf.predict(test_data)
    #preds.append(clf.predict(test_data))
    #if pred != test_label:
    #    ctr += 1
    #    print 'inst', i+1, '%d:%d'%(test_label,pred)
    acc = accuracy_score(test_label, preds)
    a_sum.append(acc)
    print acc
    if acc>tmp:
        tmp = acc
        l = test_label[:]
        p = preds[:]
#acc = accuracy_score(label, preds)
#print acc
print 'ave acc:', np.mean(a_sum)
print 'std:', np.std(a_sum)

#cm = CM(label, preds)
cm = CM(l, p)
cm = normalize(cm.astype(np.float), axis=1, norm='l1')
#cm /= cm.astype(np.float).sum(axis=1)
fig = pl.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
fig.colorbar(cax)

for x in xrange(len(cm)):
    for y in xrange(len(cm)):
         ax.annotate(str("%.3f"%cm[x][y]), xy=(y,x),
                     horizontalalignment='center',
                     verticalalignment='center')


cls = ['rec','trnsp','busi','safety','svcs','environ','health','gov','edu','housing']
pl.xticks(range(len(cm)),cls)
pl.yticks(range(len(cm)),cls)
pl.title('Confusion matrix')
#pl.colorbar()
pl.ylabel('True label')
pl.xlabel('Predicted label')
pl.show()

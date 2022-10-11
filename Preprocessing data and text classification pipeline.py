#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score


# In[2]:


nltk.download('stopwords')
nltk.download('wordnet')


# In[9]:


df = pd.read_csv('Constraint_Train.csv', sep=',')
df2 = pd.read_csv('Constraint_Test.csv', sep=',')


# In[10]:


df_t=df['tweet']
df_t2=df2['tweet']


# In[11]:


df_t


# In[12]:


df_t2


# # Remove punctuation

# One way of doing this is by looping through the Series with list comprehension and keeping everything that is not in string.punctuation, a list of all punctuation we imported at the beginning with import string.

# Jednym ze sposobów na zrobienie tego jest zapętlenie serii ze zrozumieniem listy i zachowanie wszystkiego, czego nie ma w łańcuchu. Interpunkcja, lista wszystkich znaków interpunkcyjnych, które zaimportowaliśmy na początku za pomocą importowanego ciągu.

# In[13]:


def remove_punctuation(text):
    no_punct="".join([c for c in text if c not in string.punctuation])
    return no_punct


# In[15]:


df_t=df_t.apply(lambda x: remove_punctuation(x))
#df_t2=df_t2.apply(lambda x: remove_punctuation(x))


# In[128]:


df_t


# In[129]:


df_t2


# # Tokenize

# This breaks up the strings into a list of words or pieces based on a specified pattern using Regular Expressions aka RegEx. The pattern I chose to use this time (r'\w') also removes punctuation and is a better option for this data in particular. We can also add.lower() in the lambda function to make everything lowercase.

# To dzieli ciągi na listę słów lub fragmentów w oparciu o określony wzorzec przy użyciu wyrażeń regularnych zwanych RegEx. Wzorzec, który wybrałem tym razem (r '\ w') również usuwa znaki interpunkcyjne i jest lepszą opcją w szczególności dla tych danych. Możemy również dodać .lower () w funkcji lambda, aby wszystkie litery były małe.

# In[16]:


tokenizer=RegexpTokenizer(r'\w+')


# In[17]:


df_t=df_t.apply(lambda x: tokenizer.tokenize(x.lower()))
df_t2=df_t2.apply(lambda x: tokenizer.tokenize(x.lower()))


# In[18]:


df_t


# In[19]:


df_t2


# # Remove stop words

# 
# We imported a list of the most frequently used words from the NL Toolkit at the beginning with from nltk.corpus import stopwords. You can run stopwords.word(insert language) to get a full list for every language. There are 179 English words, including ‘i’, ‘me’, ‘my’, ‘myself’, ‘we’, ‘you’, ‘he’, ‘his’, for example. We usually want to remove these because they have low predictive power. There are occasions when you may want to keep them though. Such as, if your corpus is very small and removing stop words would decrease the total number of words by a large percent.

# Zaimportowaliśmy listę najczęściej używanych słów z NL Toolkit na początku z hasłem stopwords z importu nltk.corpus. Możesz uruchomić stopwords.word (wstaw język), aby uzyskać pełną listę dla każdego języka. Istnieje 179 angielskich słów, na przykład „i”, „me”, „my”, „self”, „my”, „you”, „he”, „his”. Zwykle chcemy je usunąć, ponieważ mają niską moc predykcyjną. Są jednak sytuacje, w których możesz chcieć je zachować. Na przykład, jeśli Twój korpus jest bardzo mały, a usunięcie słów pomijanych zmniejszyłoby całkowitą liczbę słów o duży procent.

# In[20]:


def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words


# In[21]:


df_t=df_t.apply(lambda x: remove_stopwords(x))
df_t2=df_t2.apply(lambda x: remove_stopwords(x))


# In[22]:


df_t


# In[23]:


df_t2


# # Stemming & Lemmatizing

# Both tools shorten words back to their root form. Stemming is a little more aggressive. It cuts off prefixes and/or endings of words based on common ones. It can sometimes be helpful, but not always because often times the new word is so much a root that it loses its actual meaning. Lemmatizing, on the other hand, maps common words into one base. Unlike stemming though, it always still returns a proper word that can be found in the dictionary. I like to compare the two to see which one works better for what I need. I usually prefer Lemmatizer, but surprisingly, this time, Stemming seemed to have more of an affect.

# Oba narzędzia skracają słowa z powrotem do ich pierwotnej formy. Stemming jest trochę bardziej agresywny. Odcina przedrostki i / lub zakończenia słów na podstawie wspólnych. Czasami może być pomocne, ale nie zawsze, ponieważ często nowe słowo jest tak silnym rdzeniem, że traci swoje rzeczywiste znaczenie. Z drugiej strony, lematyzacja polega na odwzorowaniu popularnych słów na jedną bazę. Jednak w przeciwieństwie do robienia słów, zawsze zwraca właściwe słowo, które można znaleźć w słowniku. Lubię porównywać te dwa, aby zobaczyć, który z nich działa lepiej dla tego, czego potrzebuję. Zwykle wolę Lemmatizer, ale tym razem Stemming wydawał się mieć większy wpływ.

# In[24]:


lemmatizer=WordNetLemmatizer()

def word_lemmatizer(text):
    lem_text = [lemmatizer.lemmatize(i) for i in text]
    return lem_text


# In[25]:


df_t=df_t.apply(lambda x: word_lemmatizer(x))
df_t2=df_t2.apply(lambda x: word_lemmatizer(x))


# In[26]:


stemmer=PorterStemmer()

def word_stemmer(text):
    stem_text=" ".join([stemmer.stem(i) for i in text])
    return stem_text


# In[27]:


df_t=df_t.apply(lambda x: word_stemmer(x))
df_t2=df_t2.apply(lambda x: word_stemmer(x))


# In[28]:


df_t


# In[29]:


df_t2


# In[30]:


df['tweet']=df_t


# In[31]:


df2['tweet']=df_t2


# In[32]:


train=df
train


# In[52]:


x_train,x_test,y_train,y_test=train_test_split(train['tweet'],train['label'], test_size=0.25, random_state=7,shuffle=True)


# In[53]:


pd.DataFrame(y_train)


# In[60]:


tf=TfidfVectorizer(stop_words='english', max_df=0.75)


# In[61]:


tfidf_train=tf.fit_transform(x_train)


# In[62]:


tfidf_test=tf.fit_transform(x_test)


# In[63]:


pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)


# In[58]:


pac=PassiveAggressiveClassifier()
pac.fit(vec_train, y_train)


# In[64]:


y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[147]:


test=df2
test


# # Converting to Numeric Data
# 

# To make TF-IDF vectorizer efficient, we have to specify rather large vocabulary(max_features). It causes very very large dataset dimensionality. Usually this causes RAM issues or computational time issues on the model training step. Therefore, sklearn.feature_extraction.text.TfidfVectorizer returns sparse matrix as the output, which is much more memmory efficient and computational time efficient for some classifier models, which are able to deal with sparse matrices. But some of further preprocessing steps, are not able to deal with this, so the next steps will be very memory-consuming. Because of that I have chosen only 10 000 words vocabulary.

# Aby wektoryzator TF-IDF był wydajny, musimy określić dość obszerny słownik (max_features). Powoduje to bardzo dużą wymiarowość zbioru danych. Zwykle powoduje to problemy z pamięcią RAM lub problemy z czasem obliczeniowym na etapie uczenia modelu. Dlatego sklearn.feature_extraction.text.TfidfVectorizer zwraca rzadką macierz jako wynik, co jest znacznie bardziej wydajne pod względem pamięci i czasu obliczeniowego dla niektórych modeli klasyfikatorów, które są w stanie radzić sobie z rzadkimi macierzami. Jednak niektóre z dalszych kroków przetwarzania wstępnego nie są w stanie sobie z tym poradzić, więc następne kroki będą wymagały bardzo dużej ilości pamięci. Z tego powodu wybrałem tylko 10 000 słów.

# Now that we’ve done all the preprocessing required, we’ll convert the data to numeric data. As mentioned before this is done using sklearn. The bag-of-words model just counts the number of times a word appears in each document. The tf-idf matrix makes the observation that some words naturally appear more than others, and this can have an undesired effect on the analysis. For example, in the context of the news articles we have, the word report may appear a lot in all the articles, but this word does not necessarily make it obvious what the article is about. On the other hand, the word fire may not appear much, but it is highly informative when it does. For this reason tf-idf offsets the counts of the words by the number of articles a word appears in.
# The creation of bag-of-words and tf-idf matrices are similar, and one of the first decisions to make is which words to use in the vocabulary. Often this can be created automatically, by using the words that appear in the articles. Though it is recommended to choose a cut off point by picking a maximum number of features; as words that appear very infrequently will not be very informative. This can easily be done by setting the argument _max_features_ in the sklearncode that we'll detail soon. We'll set this to 1000, though it can go as high as 20000 if you have a very large corpus. An alternative approach if you have a specialised application is to use a predetermined vocabulary set as this can lead to a more informative vocabulary.
# To create the matrices, we use the sklearn objects CountVectorizer for creating a bag-of-words model and TfidfVectorizer to create a tf-idf matrix. Once the fit_transform method has been applied, a sparse matrix of the form required will be returned. In the sparse matrix, each row is a nonzero entry of the matrix, and the columns correspond to: article_num, vocab_index, word_count. The words that correspond to each vocab index can be found by using the method get_feature_names, as we demonstrate below.
# 
# 

# Teraz, gdy wykonaliśmy wszystkie wymagane wstępne przetwarzanie, przekonwertujemy dane na dane liczbowe. Jak wspomniano wcześniej, odbywa się to za pomocą sklearn. Model worka słów po prostu zlicza, ile razy słowo pojawia się w każdym dokumencie. Macierz tf-idf pozwala zaobserwować, że niektóre słowa pojawiają się naturalnie częściej niż inne, co może mieć niepożądany wpływ na analizę. Na przykład w kontekście artykułów z wiadomościami, które posiadamy, słowo raport może pojawiać się często we wszystkich artykułach, ale to słowo niekoniecznie wyjaśnia, o czym jest artykuł. Z drugiej strony, słowo ogień może nie pojawiać się zbyt wiele, ale jest bardzo pouczające, kiedy już się pojawia. Z tego powodu tf-idf kompensuje liczbę słów o liczbę artykułów, w których występuje słowo.
# Tworzenie macierzy bag-of-words i tf-idf jest podobne, a jedną z pierwszych decyzji, które należy podjąć, jest to, które słowa użyć w słowniku. Często można to utworzyć automatycznie, używając słów pojawiających się w artykułach. Chociaż zaleca się wybranie punktu odcięcia poprzez wybranie maksymalnej liczby cech; ponieważ słowa, które pojawiają się bardzo rzadko, nie będą zbyt pouczające. Można to łatwo zrobić, ustawiając argument _max_features_ w sklearncode, który wkrótce opiszemy. Ustawimy to na 1000, chociaż może wzrosnąć nawet do 20000, jeśli masz bardzo duży korpus. Alternatywnym podejściem, jeśli masz wyspecjalizowaną aplikację, jest użycie wcześniej określonego zestawu słownictwa, ponieważ może to prowadzić do bardziej pouczającego słownictwa.
# Aby utworzyć macierze, używamy obiektów sklearn CountVectorizer do tworzenia modelu worka słów i TfidfVectorizer do tworzenia macierzy tf-idf. Po zastosowaniu metody fit_transform zostanie zwrócona rzadka macierz wymaganej postaci. W rzadkiej macierzy każdy wiersz jest niezerowym wpisem macierzy, a kolumny odpowiadają: numer_artykułu, indeks_ocisku, liczba_słów. Słowa, które odpowiadają każdemu indeksowi vocab, można znaleźć za pomocą metody get_feature_names, jak pokazano poniżej.

# In[41]:


counter = sklearn.feature_extraction.text.CountVectorizer(max_features = 100)


# In[42]:


df_t1 = counter.fit_transform(df_t)


# In[43]:


print(df_t1)


# In[44]:


df_t22= counter.fit_transform(df_t2)
print(df_t22)


# Most models requires features normalization to (0, 1) range in order to converge faster

# In[45]:


tf_counter = sklearn.feature_extraction.text.TfidfVectorizer(max_features = 100)


# In[46]:


tfidf = tf_counter.fit_transform(df_t)


# In[47]:


print(tfidf)


# In[48]:


tfidf2 = tf_counter.fit_transform(df_t2)
print(tfidf2)


# In[ ]:


pac=PassiveAggressiveClassifier()
pac.fit(vec_train, y_train)


# In[49]:


y=tf_counter.get_feature_names()
y


# # Creating models

# In[52]:


from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import numpy as np
rng = np.random.RandomState(1)
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from eli5 import transform_feature_names
import eli5


# In[33]:


logit = LogisticRegression(C=5e1, solver='lbfgs', multi_class='multinomial', random_state=17, n_jobs=4)


# In[34]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)


# In[35]:


valid = pd.read_csv('Constraint_Val.csv', sep=',')


# In[36]:


train_val = pd.concat([train, valid])


# In[37]:


print(train_val)


# In[38]:


train_val['label']


# In[39]:


sns.countplot(train_val['label']);
plt.title('Train+val: Target distribution');


# In[41]:


text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=150000)


# In[42]:


get_ipython().run_cell_magic('time', '', "X_train_text = text_transformer.fit_transform(train_val['tweet'])\nX_test_text = text_transformer.transform(test['tweet'])")


# In[43]:


print(X_train_text)


# In[44]:


print(X_test_text)


# In[45]:


get_ipython().run_cell_magic('time', '', "cv_results = cross_val_score(logit, X_train_text, train_val['label'], cv=skf, scoring='f1_micro')")


# In[46]:


print(cv_results)


# In[47]:


cv_results, cv_results.mean()


# It's nice to see that cross-validation is more or less stable across folds. Let's train the model on train + val.

# In[48]:


get_ipython().run_cell_magic('time', '', "logit.fit(X_train_text, train_val['label'])")


# In[53]:


eli5.show_weights(estimator=logit, 
                  feature_names= list(text_transformer.get_feature_names()),
                 top=(50, 5))


# In[55]:


test_preds = logit.predict(X_test_text)


# In[56]:


pd.DataFrame(test_preds, columns=['label']).head()


# In[57]:


pd.DataFrame(test_preds, columns=['label']).to_csv('logit_tf_idf_starter_submission.csv',
                                                  index_label='id')


# In[80]:


df


# In[ ]:





# In[111]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(train, test_size=0.3)


# In[112]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(analyzer = "word", max_features=10000)

X_tfidf_train = tfidf_vectorizer.fit_transform(train['tweet'])
X_tfidf_test = tfidf_vectorizer.transform(test['tweet'])


# In[116]:


print(X_tfidf_train)


# In[117]:


print(X_tfidf_test)


# In[118]:


y=train['label']
y_test = test['label']


# In[75]:


get_ipython().system('pip install imbalanced-learn ')


# In[120]:


from sklearn.calibration import CalibratedClassifierCV
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.svm import LinearSVC

iht = InstanceHardnessThreshold(random_state=0, n_jobs=11,
                                 estimator=CalibratedClassifierCV(
                                     LinearSVC(C=100, penalty='l1', max_iter=100, dual=False)
                                 ))
X_resampled, y_resampled = iht.fit_resample(X_tfidf_train, y)
print(sorted(Counter(y_resampled).items()))


# In[60]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print_confusion_matrix(confusion_matrix, 
                           class_names, 
                           figsize = (15,15), 
                           fontsize=12,
                           ylabel='True label',
                           xlabel='Predicted label'):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)


# In[61]:


def evaluate_model(model, X, y, X_test, y_test, target_names=None):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    scores_test = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
    
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    print("Accuracy test: %0.2f (+/- %0.2f)" % (scores_test.mean(), scores_test.std()))
    
    print("Test classification report: ")
    if target_names is None:
        target_names = model.classes_
    print(classification_report(y_test, model.predict(X_test), target_names=target_names))
    print("Test confusion matrix: ")
    print_confusion_matrix(confusion_matrix(y_test, model.predict(X_test)), class_names=target_names)


# In[ ]:


mb = MultinomialNB()
mb.fit(X_selected, y_resampled)
evaluate_model(mb, X_selected, y, X_test_selected, y_test)


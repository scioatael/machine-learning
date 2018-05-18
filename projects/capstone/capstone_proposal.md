# Machine Learning Engineer Nanodegree
## Capstone Proposal
Krzysztof Rutczynski  
18.05.2018

## Proposal

**How good is your Medium article?**

Predicting the populatity of news article on a self-publishing website Medium. 


### Domain Background

In this project I analyze the datasets of a year worth of [Medium](www.medium.com) articles to **predict the *popularity* of a given article**. 

Medium is an online publishing platform, which is known for high quality articles from various fields like politics, technlogy, culture. 
Medium has been founded in 2012 by Twitter's CEO as a complementary "tool" to publish news longer than Twitters 140 characters. 
In the meantime a lot of authors both amateur and professionals alike publish via Medium, which has grown to be the leading social journalism platform

The problem being solved is related to **natural language processing** and **text classification**. I will be analyzing the characterisits of news articles where also textual features of the articles may be of importance. 
Getting connection between the piece of text and its popularity can have many practical applications for publishers and consumers alike e.g:  
- discovering current hot topics or trends in public opinion, 
- proactively promoting or highlighting the  valuable articles, 
or conversely for publishers and authors: 
- applying the models or learning from the models to increase the chance of article being successful.  
 
I am a keen subsciber to technology areas on Medium, e.g. "Towards Data Science" or "AI. My motivation to explore the Medium article dataset stems from my interests in finding a high quality article as well as exploring the properties that valuables articles shall have. 



### Problem Statement

The task that I will be solving is the **prediction of number of recommendation** for an article published on Medium platform. 
Given the historical data about the articles (see next section) and the number of recommendation their received, we would like to predict what is the potential number of recommendation that the particular article will receive once published. 

This is a classical *predictive modelling* problem where, given a historical labeled data (ground truth), we will create a model to serve a prediction for unseen data of similar type (new article)


### Datasets and Inputs

The challenge has been defined as a Kaggle competition available and the dataset has been given by the contest organisation team. 

Link: https://www.kaggle.com/c/how-good-is-your-medium-article

The training set (`train.json`) comprises of 62313 articles published on Medium and the affiliate websites between 2012 and July 1st 2017. 
The test set (`test.json`) comprises of 34645 articles published on Medium from 1st July 2017 till March 3, 2018.

Each of the JSON object contain the following fields:
 - `_id and url` – URL of a published article
 - `published` – publication time
 - `title`
 - `author` – author's name, Twitter and Medium accounts
 - `content` – HTML-content of the article
 - `meta_tags` - contain additional info describing an article (see details below)

Please find in addition some key or potentially relevant meta tags: 
 - `description`
 - `twitter:data1` - estimated reading time
 - authors website

Please see an annex for a detailed example of available meta tags. 

The target variable for the problem is the **number of recommendations** (a.k.a. *claps*) that the given article has received. 
This value is contained in the separate file train_log1p_recommends.csv.
The number of recommendation is given under the log1p-transformation: **`log1p(x) = log(1+x)`**. 



### Solution Statement

The task is to predict the *number of recommendations* (which we consider as a continuous value) based on the *content* and *metadata* that we possess about corresponding articles. Since we have a labeled data, we can apply a **supervised learning** approach to train a model that will predict the target value given the arcictle content and characteristics. Since our target value is a continous, we will choose the **regression model**. Due to the fact that we have too few samples (<100K) I will not attempt to use a popular neural networks approach. Instead I will try with [*Ridge Regression*](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) or [*SVR*](http://scikit-learn.org/stable/modules/svm.html#svm-regression), which are state of the art algorithm for this problem.  

Most of the features of the input data are textual: `content`, article `name`, `description`, `tags`. I will evaluate a range of transformation techniques to convert the textual features into numerical (vectors), including  *Bag of Words*, *TF-IDF* and *word embeddings*. 

I will also explore the HTML `content` field to extract additional features, e.g. *tags* used by author to describe the article.  
An EDA (exploratory data analysis) will be performed to help find additional features of the input data that can be used for building a model, e.g. statistical information about the text like:
- readibility (Automated Readibility Index) [ARI]
- amount of links in the article 
- length of the articles 


### Benchmark Model

I will benefit from the fact that the problem has been announced as a Kaggle challenge, thus it can be benchmarked with existing results from other competitors. In addition on the competition website there are some [kernels](https://www.kaggle.com/kashnitsky/ridge-countvectorizer-baseline) published which can be regarded as a baseline model for the problem [KAG]

The baseline solution for regression is a linear regression model and baseline text transformation technique is a bag of words. 
These two techniques combined will also consitute to my baseline model, which I will further down optimise. 


### Evaluation Metrics

The solution will be evaluated by submitting the result set of number of recommendation given to the test articles. 

The imposed evaluation metric in this kaggle competition is **MAE (Mean Absolute Error)**. [MAE]

This is a common metric to measure accuracy for continuous variables with a negatively oriented-score.
Is it an average over a test sample of absolute difference between the prediction and actual observation. 
![MAE Formula](https://cdn-images-1.medium.com/max/800/1*OVlFLnMwHDx08PHzqlBDag.gif "MAE Formula")

This metric is useful (in comparison to *RMSE*) when we want a better interpretability of comparison between two results and is invariant to distribution of error magnitudes. *RMSE* in turn is more sentistive if we have a error distribution with larger variable.        

### Project Design

As already outlined, the task that I will be solving is the **prediction of number of recommendation for an article published on Medium platform**. 

My assumption is that the key to success in reaching the lowest MAE will be the following factors:
- careful preprocessing of the data, including the text data cleanup
- exploratory data analysis and careful selection of right or engineering of new features that can boost the overall score 

The following paragraph outlines the steps taken to tackle the problem as well as provide a deep dive to certain methods and algorithms used. 

#### 1. Visual inspection of the input data
 First stage of the project constitute of inspecting input data with the goal of identifying potential features characterising good article.
 What of the learnings from this stage is that the content field contains a HTML representation of content which we would like to clean up from HTML tags before further processing.  
#### 2. Data preprocessing
 I will preprocess the list of JSON files and convert it into pandas dataframe for further processing. I will also merge the target label to enable the exploratory data analysis. A necessary cleanup will be done on the content field. 
#### 3. Exploratory data analysis
 I will perform EDA to better understand the statistical characteristics of article and its metadata. Some of the question I would like if popularity of the article is related to:
 - article length - for this I will count number of words, or use the ready feature from `meta_data` field:
    >    "twitter:label1": "Reading time",
    >    "twitter:data1": "3 min read", 
 - specific tags selected by the author (where the user can search for certain tags)
 - authors name / reputation
 - amount of links in the article
 - some temporal considerations (e.g. latest article getting more recommendations)
 I will also try to explore the text by preparing tag cloud to pickup most common topics.

#### 4. Further Feature engineering 
##### 4.1 Text preprocessing
  I will preprocess `content` field for further processing, mostly using NTLK library [NLTK]: 
    - removing popular words from English language (stop words) - although the article base is multilingual (including chinese, I will first assume that most of the articles are written in English)
    - punctuation
    This preprocessing will help in further steps (BoW, RAKE) and also to reduce the dimensionality of textual features (dictionary)
 - Automated Readibility Index
  I will apply a ARI method, which is assessing a complexity of the written text by identifying the age /class of the student that could cope with particular text. The formula for ARI is the following: 
  > ![ARI formula](http://www.readabilityformulas.com/graphics/ariformula.jpg)

  In addition I will apply one of the keyword creation technique (RAKE) to check most relevant keywords per document and compare them with the keywords selected by the author. 
  > RAKE:
  - Remove all stop words from the text( eg for, the, are, is , and etc.)

  - create an array of candidate keywords which are set of words separated by stop words

  - find the frequency of the words. (stemming might be used.)

  - find the degree of the each word. Degree of a word is the number of how many times a word is used by other candidate keywords

  - for every candidate keyword find the total frequency and degree by summing all word's scores.

  - finally degree/frequency gives the score for being keyword. 
 
##### 4.2 Word and document representation
  I am planning to evaluate the following representation of textual features: 
  - Bag of Words 
  - Term Frequency/Inverse Document Frequency (TF/IDF)
  - Word embeddings: word2vec, fasttext
  
  The idea of all methods for word and/or document representation is to transform the sting values of text into numerical representation in such a way to still keep as much as possible from the original text characteristics either from statistical (BoW, TF/IDF) or semantic point of view (word embeddings). 
  - Bag of Words - this representation is a sparse vector of a dimension of the whole dictionary, where each word in a dictionary has its position and the value indicated an amount of occurences of this word in a sentence

    >   I am planning to use a `CountVectorizer` from scikit-learn python library for BoW implementation
    >   ```python
    >       cv = CountVectorizer(max_features=50000)
    >       X_train_counts = cv.fit_transform(train["content"])
    >
    >   ```
  - TFIDF - Term frequency / Invert document frequence takes the BoW concept further and assign a weight to a word taking into account both its frequency in the document, but also what is the freuqency of the word in all other documents. Rare words will be more outstanding thanks to this method. 
    >   I will be using sklearn `TfIdfTranformer` on top of previously mentioned `CountVectorizer`. Taking the weights of unique words into consideration should generally improve the model scoring
    >   
    >    ```python
    >        tfidf = TfidfTransformer()
    >       X_train = tfidf.fit_transform(X_train_counts)
    >   ```
  - Word2Vec/Fasttext 
    > I will use a `gensim` python library for working with text representations including all popular word embedding methods. 
    > An word embedding is a dense vector representation of the word that contains also a semantic relations, where words which are similar are close to each other in the vector space e.g.: 
    ![word2vec](https://www.tensorflow.org/images/linear-relationships.png) 
    credit:https://www.tensorflow.org/tutorials/word2vec

    >For the embeddings I will investigate a methods to represent a whole document based on the embeddings of words contained in the document. 
    >Two approached I will explore are: 
    >    - to compute an mean value of all word vectors contained in the text
    >    - same as above but with consideration of word frequency (tf/idf weighted)
    > Alterntive approach would be to use a document embedding using [**doc2vec**](https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e). This approach still needs to be investigated.

#### 5. Model selection and training
   I will select some of above described features based on the results of initial exploratory data analysis and train a regression model. 
   
   I am planning to start with `RidgeRegression` from `sklearn` and try also SVR, but depending on the amount of observations SVR may be considerably slower.
   - [Ridge Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) 
    
   - [SVR](http://scikit-learn.org/stable/modules/svm.html#svm-regression) 

#### 6. Hyperparameter tuning and feature optimisation
   Based on the initial result I will experiment with differnt values of training parameters as well as various features / feature represations (eg. with or withour word embeddings)

#### 7. scoring and evaluation on Kaggle
   In the final step I will evaluate my results of `MAE` inside the kaggle platform. 


### References

[MAE] https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d

[ARI] https://medium.com/the-mission/after-10-000-data-points-we-figured-out-how-to-write-a-perfect-medium-post-58c41c314f6a

[KAG] https://www.kaggle.com/kashnitsky/ridge-countvectorizer-baseline

https://www.kaggle.com/c/how-good-is-your-medium-article

https://www.kaggle.com/kashnitsky/ridge-countvectorizer-baseline

http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

http://scikit-learn.org/stable/modules/svm.html#svm-regression

https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e


-----------

### Annex - Example of article metadata

```json
 "meta_tags": {
        "viewport": "width=device-width, initial-scale=1",
        "title": "How fast can a camera get? – What comes to mind – Medium",
        "referrer": "unsafe-url",
        "description": "Answer this question yourself. What’s the highest rated camera in terms of its frame capturing speed? You will basically search for the term high-speed camera. Then you will see a list of cameras…",
        "theme-color": "#000000",
        "og:title": "How fast can a camera get? – What comes to mind – Medium",
        "og:url": "https://medium.com/what-comes-to-mind/how-fast-can-a-camera-get-215fe5403a9b",
        "og:image": "https://cdn-images-1.medium.com/max/1200/1*6UZIfXhTsBLYea8yK0MALQ.jpeg",
        "fb:app_id": "542599432471018",
        "og:description": "When light practically stood still…",
        "twitter:description": "When light practically stood still…",
        "twitter:image:src": "https://cdn-images-1.medium.com/max/1200/1*6UZIfXhTsBLYea8yK0MALQ.jpeg",
        "author": "Vaibhav Khulbe",
        "og:type": "article",
        "twitter:card": "summary_large_image",
        "article:publisher": "https://www.facebook.com/medium",
        "article:author": "https://medium.com/@vaibhavkhulbe",
        "robots": "index, follow",
        "article:published_time": "2017-05-06T08:16:30.776Z",
        "twitter:creator": "@vaibhav_khulbe",
        "twitter:site": "@Medium",
        "og:site_name": "Medium",
        "twitter:label1": "Reading time",
        "twitter:data1": "3 min read",
        "twitter:app:name:iphone": "Medium",
        "twitter:app:id:iphone": "828256236",
        "twitter:app:url:iphone": "medium://p/215fe5403a9b",
        "al:ios:app_name": "Medium",
        "al:ios:app_store_id": "828256236",
        "al:android:package": "com.medium.reader",
        "al:android:app_name": "Medium",
        "al:ios:url": "medium://p/215fe5403a9b",
        "al:android:url": "medium://p/215fe5403a9b",
        "al:web:url": "https://medium.com/what-comes-to-mind/how-fast-can-a-camera-get-215fe5403a9b"
```



# Sentiment Analysis

## References used in while developing this project
### Personal notes
- Learning about text mining:<br>https://github.com/pranigopu/textMining
- Learning about browser extension:<br>https://github.com/pranigopu/browserExtensions
- Learning TensorFlow:<br>https://github.com/pranigopu/learningTensorFlow

### Conceptual information references
- Defining artificial intelligence:<br>https://www.britannica.com/technology/artificial-intelligence
- Defining structured data:<br>https://www.tibco.com/reference-center/what-is-structured-data
- Steps in text mining:<br>https://www.lexalytics.com/lexablog/text-analytics-functions-explained

### External source codes & tutorials
#### Sentiment analysis basics
- https://asperbrothers.com/blog/sentiment-analysis-in-python/
- https://techvidvan.com/tutorials/python-sentiment-analysis/
- https://monkeylearn.com/blog/opinion-mining/
- https://pub.towardsai.net/sentiment-analysis-opinion-mining-with-python-nlp-tutorial-d1f173ca4e3c#1531

#### Web scraping & crawling
- https://towardsdatascience.com/scraping-tripadvisor-text-mining-and-sentiment-analysis-for-hotel-reviews-cc4e20aef333
- https://github.com/inistory/Yelp-restaurant-review-sentiment-analysis/blob/master/Yelp_crawler.ipynb
- https://www.youtube.com/watch?v=6jhwS5JmAbc
- https://www.octoparse.com/blog/web-data-crawling-bag-of-words-for-data-mining

#### Chrome extension
- https://www.freecodecamp.org/news/how-to-create-and-publish-a-chrome-extension-in-20-minutes-6dc8395d7153/

#### Chrome extension using Python backend
- https://medium.com/@oaishi.faria/connecting-chrome-extension-with-python-backend-912d1d0db26
- https://towardsdatascience.com/building-a-serverless-chrome-extension-f684740e1ffc
- https://morioh.com/p/0e3b33fe9851
- https://pythonspot.com/create-a-chrome-plugin-with-python/

#### Sentiment analysis training model data using LSTM neural network
- https://asperbrothers.com/blog/sentiment-analysis-in-python/

### Documentation guidelines & references
#### Software requirements specification examples
- https://senior.ceng.metu.edu.tr/2016/fixit/SRS.pdf
- https://www.koreascience.or.kr/article/JAKO202002761602619.pdf
- http://www.cse.aucegypt.edu/~rafea/SATA/Reports/RequirementsSpecifications-Rafea-3.pdf

#### Creating Overall Report Format:
https://www.rcciit.org/students_projects/projects/cse/2018/GR20.pdf

### Reference material for natural language processing (using Python)
- _"Natural Language Processing Recipes"_ by Akshay Kulkarni & Adarsha Shivananda

### Group Publisher for Chrome Web Store Related:
1. For group email ID and related managing dashboard:<br>https://groups.google.com/g/hbp142325cs

## Defining the problem
- What is the sentiment behind a text?
- What sentiments are directed towards what aspects?
- How can we obtain the answers to the above through machine learning?

### Elaborating on the terms
#### Sentiment
Sentiment refers to
- Expression of emotion (or lack thereof)
- Expression of approval (or lack thereof)

##### Identifying sentiment without identifying its target
Sentiment presupposes a target (i.e. an entity or event) for the sentiment. In other words, sentiment is always sentiment about something. However, natural languages allow for more generalized expressions of sentiment i.e. they allow for the identification of a sentiment without having to specify the target of the sentiment, which does exist, but may exist in any form. In other words, we can identify the sentiment (either approximately or accurately) without having to specify what the sentiment is directed towards.
##### Context and sequence
The context in which a linguistic unit is used can play a significant role in determining the linguistic unit's meaning and sentiment. Here, context can be defined as the combination of
- The known circumstances surrounding the linguistic unit
- Other linguistic units neighboring the linguistic unit
- The domain of discussion (ex. subject doubts, hotel reviews, movie critiques...)

Evaluating each unit of the text in appropriate context can be key in determining the meaning and sentiment of the text.
Secondly, the sequence in which linguistic units appear, particularly words, can greatly impact the meaning and sentiment of the given text, particularly in more analytical  languages like English. The sequence can help determine
- Emphases
- Subjects and objects of a sentence
- Topic of a sentences
- Relations between aspects text and modifiers
- Link between entities, events and actions

#### Machine learning
##### General overview
Machine learning is a subset of artificial intelligence. Artificial intelligence, in general terms, is a machine's capacity to perform tasks that used to require sentient intelligence. In more precise terms, artificial intelligence is a machine's capacity to reason, discover meaning in data, generalize observations or learn to past experiences .
<br><br>
Machine learning is artificial intelligence that learns from available data, and adapts accordingly. In other words, in machine learning, the machine's behavior does not rely only on programmed instructions, but also on the machine's interpretation of available data. Here, programmed instructions define the machine's approach to interpreting the data, but not the machine's ultimate behavior.

##### Need for machine learning
###### Computation power
Increasing computation power increases the scalability of an application. Machine learning harnesses the computational power of a computer, allowing for applications that process a much greater quantity of data in much lesser time, compared to human labor. Processing more data is helps in the following:
- Increasing the probability of find underlying patterns
- Decreasing the effect of outliers or abnormal data
- Identifying more variables and classes

Furthermore, being able to process greater quantities of data in much lesser time can help in creating a more useful and usable application.

###### Necessity to adapt to data
Natural language is extremely diverse in terms of concepts and expressions, and varies greatly depending on context. Furthermore, the number of permutations and combinations in which concepts and expressions may appear are practically infinite, and the effects of these permutations and combinations on meaning are often significant.
<br><br>
However, language is based on rules, which may be clearly defined or implicit in usage. Usage of language usually follows discernible patterns that human speakers learn to identify over time, through experience and education. Similarly, in order to interpret natural language texts accurately, using available data to shape a computer's models of natural language meaning seems to be the best approach.
<br><br>
Note that this does not consider the validity of the source of the data. For example, the frontpage of a hotel's website may contain overwhelmingly positive reviews, while online review sections may contain a more diverse range of reviews.

##### Case specific
###### Requirement
In our case, we need the machine to interpret texts that are written in English, and thereby determine:
- The overall underlying sentiment
- The underlying sentiments with respect to aspects

The broad area of machine learning involved here is natural language processing (NLP), which includes sentiment analysis (our area of focus). The logical, algorithmic and programming approach to sentiment analysis is discussed later.

### Elaborating on relevant NLP concepts
#### Overview
NLP is a subfield of both linguistics and artificial intelligence. It is the study of computational<sup>[1](#f1)</sup> methods to process, analyze and interpret natural language data. It involves two broad components:
1.	Natural language understanding (NLU)
2.	Natural language generation (NLG)

For this project, the focus will be on NLU, which deals with the computer's reading comprehension i.e. the ability of the computer to gather, structure and analyze the elements of a natural language source. In this project, we will only focus on digital text sources.

#### NLU approach
##### Text mining
###### Overview
Text mining or text analytics is the process of transforming unstructured text into a structured, normalized data. Structured data  implies
1.	Standardized format (i.e. format uniformly used for every use case)
2.	Well-defined structure
3.	Accessible to humans and programs

Data normalization is the organization of data into a more logical structure, which involves:
1.	Removing redundancies
2.	Data munging (i.e. transforming the data into a more usable format)
3.	Grouping data according to shared characteristics

###### Purpose
Structuring and normalizing data makes complex processes like machine learning more efficient and effective, since machine learning relies on data quality and accessibility. For example, structuring and normalizing data can help in:
- Iterating through elements more efficiently (ex. iterating through words)
- Dealing with necessary data (by removing redundancies)
- Generalizing a computational method for a standardized format
- Efficiently accessing necessary metadata (ex. word index, word count)

###### Dealing with ambiguity
Ambiguity in natural language can appear in the following forms:
1.	Lexical ambiguity (also called semantic ambiguity)
2.	Syntactic ambiguity
3.	Referential ambiguity
## Solution approach
### Breakdown
The different parts of our solution are covered in the following order:
1.	Data gathering
  - Identifying appropriate online sources
  -	Web scraping methods
  -	Data storage and usage
2.	Analyzing the overall sentiments for given texts
  -	Text mining
  -	Analyzing sentiment
    -	Conceptual method
    -	Machine learning algorithm
    -	Programming
3.	Analyzing aspects for given texts
  -	Text mining
    -	Conceptual method
    -	Machine learning algorithm
    -	Programming
4.	Analyzing relationship between aspects and sentiments
5.	Scaling up sentiment analysis
6.	Packaging the application

### Domain of focus
For the purpose of simplicity, the sentiment analysis is focused on online reviews of hotels. This domain was chosen because hotels depend on reputation and customer satisfaction, and must aim to understand their customers' needs quickly and carefully, which is why sentiment analysis of hotel reviews is a potentially useful application.
<br><br>
Furthermore, hotels must not only gather the overall sentiments of their customers, but also gather their sentiments regarding the different aspects of hotel services, such as housekeeping, furnishing, amenities, etc., so that they can obtain information they can act upon more precisely.

## Data gathering
### Overview
Before discussing the methods of analyzing text, we must discuss the methods of acquiring the necessary data. Digital textual data can be available in many forms, such as:
- Plain text
- Rich text
- HTML elements
- Photographs

Since the focus of our project is on analyzing online hotel reviews, we will be gathering data from websites. Hence, we will focus on gathering data from plain text and HTML elements.

### Identifying appropriate online sources
For an online source to be appropriate for our purpose, it needs to be:
- Directly related to the hotel business
- Contain a well-defined reviews section

### Web scraping methods
Web scraping is the extraction of data from websites, which can be done manually or through a program. Our focus is on creating programs to perform web scraping and automatically detect and retrieve reviews.

# POC-for-NLP-Project

# Sentiment Analysis
Sentiment analysis is one of the most common NLP problems. The goal is to analyze a text and predict whether the underlying sentiment is positive, negative or neutral. What can you use it for? Here are a few ideas - measure sentiment of customer feedback, survey responses, social media, and movie reviews!

# Natural Language Processing libraries:

## nltk 
  Is the most popular Python package for Natural Language processing, it provides algorithms for importing, cleaning, pre-processing text data in human language and then apply computational linguistics algorithms like sentiment analysis.
## gensim 
  Is a robust open-source vector space modeling and topic modeling toolkit implemented in Python. It uses NumPy, SciPy and optionally Cython for performance.
## spaCy 
  Is an extremely optimized NLP library that is meant to be operated together with deep learning frameworks such as TensorFlow or PyTorch.
  
# Project Workflow

![image](https://user-images.githubusercontent.com/28219393/89193559-0e519c80-d59e-11ea-817f-de19b5c148aa.png)


## Data Preprocessing/Text Preprocessing
- Natural Language Processing (NLP) is all about leveraging tools, techniques and algorithms to process and understand natural language-based data, which is usually unstructured like text, speech and so on.

![image](https://user-images.githubusercontent.com/28219393/89193845-7acc9b80-d59e-11ea-85f9-2f2d0d49c2b2.png)


- Pre-processing of text data is done in order to extract better features from the cleaned data
- Text cleaning is task specific
    Case, punctuations, markup, languages, hyphens, section markers, spellings etc. all need to be handled based on context
    
 # Information Extraction

The figure shows the architecture for a simple information extraction system. It begins by processing a document using several of the procedures: first, the raw text of the document is split into sentences using a sentence segmenter, and each sentence is further subdivided into words using a tokenizer. Next, each sentence is tagged with part-of-speech tags, which will prove very helpful in the next step, named entity detection. In this step, we search for mentions of potentially interesting entities in each sentence. Finally, we use relation detection to search for likely relations between different entities in the text.

![image](https://user-images.githubusercontent.com/28219393/89194626-853b6500-d59f-11ea-9631-4aab895d01b4.png)







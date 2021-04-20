# netflix-recommendation
A movie recommendation system using a small Netflix movie database
# How it works
After deleting rows with missing values, movie descriptions (taken from netflix_titles.csv) are transformed to vectors using either a sklearn CountVectorizer or a TfIdfVectorizer and then matching is done using Jaccard similarity "boolean" or TFIDF "tf-idf" cosine similarity. The original project (this is taken from NKUA Data Mining Course, Department of Informatics and Telecommunications) requested some extra implementations, which may be made over time. Possible improvements include using the IMDB movie ratings database as well. 

Book Recommendation System:
website link-http://bookrecm.herokuapp.com/

We have used natural language preprocessing and machine learning to create an algorithm which gives out the most similar books to the user's input. The dataset for books was obtained from kaggle and was then used at a smaller scale. This model is then put out on web by web2py making use of the streamlit module of python and is further set up on the heroku server. Our website also provides the reliable links to purchase the books of choice.

Note: As the server didn't allow more than a certain amount of memory, we were forced to take a small number of samples resulting in lower accuracy in similarity. However the model works fine even for larger samples as observed on the local server.



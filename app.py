import streamlit as st
import pickle as pkl
import pandas as pd
import requests
import re
import string
import nltk
nltk.download('stopwords')
stopwd = nltk.corpus.stopwords.words('english')
def clean_text(text):
    
    text= text.lower() # Lowercasing the text
    text = re.sub('Mail <svaradi@sprynet.com> for translation. ', '', text) # Removing unknown mail
    text = re.sub('-',' ',text.lower())   # Replacing `x-x` as `x x`
    text = re.sub(r'@\S+', '', text) # Removing mentions
    text = re.sub(r'http\S+', '', text) # Removing Links
    text = re.sub(f'[{string.punctuation}]', '', text) # Remove punctuations
    text = re.sub(f'[{string.digits}]', '', text) # Remove numbers
    text = re.sub(r'\s+', ' ', text) # Removing unnecessary spaces
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text) # Removing single characters
    
    words = nltk.tokenize.word_tokenize(text,language="english", preserve_line=True)
    text = " ".join([i for i in words if i not in stopwd and len(i)>2]) # Removing the stop words

    return text.strip()


# Function to load data
def load_data(file_name):
    return pkl.load(open(file_name, 'rb'))

# Function to load titles, dataframe, model, vectorizer, and encoder
load_titles = lambda: load_data('titles.pkl')
load_dataframe = lambda: load_data('dataframe.pkl')
load_model = lambda modelname: pkl.load(open(f'{modelname}.pkl', 'rb'))
load_vectorizer = lambda: load_data('Cvectorizer.pkl')
load_encoder = lambda: load_data('encoder.pkl')


# HTML template for displaying genre predictions
html_string = '''
<p style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif">
    <span style="font-weight: bold;font-size: x-large;">{} Genre:</span>
    <strong style="color: rgb(110,181,47)">"{}"</strong>
</p>
'''

# Streamlit configuration
st.set_page_config(layout='centered', page_title='Movie Genre Predictor',
                   page_icon="🎬", initial_sidebar_state='auto')

# Header and caption
st.header("Movie Genre Predictor")


# Model accuracies
#accuracies = [
 #   {"Model": "Logistic Regression", "Train Accuracy": 0.9468998860022146, "Test Accuracy": 0.5159040590405904},
  #  {"Model": "BernoulliNB", "Train Accuracy": 0.9134487458883257, "Test Accuracy": 0.5271955719557195},
   # {"Model": "MultinomialNB", "Train Accuracy": 0.9319468590753393, "Test Accuracy": 0.5165313653136532},
    #{"Model": "Support Vector", "Train Accuracy": 0.9908692943292223, "Test Accuracy": 0.4830811808118081}
#]

# Model codes
model_codes = {"Logistic Regression": "lgc",
               "SVM": "svc", "Multinomial Naive Bayes": "mnb", "Bernoulli Naive Bayes": "bnb"}

# Columns for model selection and accuracies
#col1, col2 = st.columns(2)
 #with col1:
  #  model = st.selectbox("Select the model to use", list(model_codes.keys()), key="model_select")
#with col2:
 #   df = pd.DataFrame(accuracies)
  #  df.index = df["Model"]
   # df.drop("Model", axis=1, inplace=True)
    #st.write(df)

# Radio button for input selection
menu_id = st.radio("Select an option to enter the input", ("Search Movie", "Manually write title and description"))

if menu_id == "Search Movie":
    # Load titles and display options
    titles = load_titles()
    option = st.selectbox('Select the movie Name:', tuple(titles[::10]), key="movie_select")
    
    # Load data and display information
    df = load_dataframe()
    row = df[df['TITLE'] == option]
    glbtitle, glbdescription, glbimg, actual_genre = row["TITLE"].values[0], row["DESCRIPTION"].values[0], row["Poster_Link"].values[0], row["GENRE"].values[0]

    # Display information in columns
    col1, col2 = st.columns(2)
    with col1:
        st.header(f"Title: {glbtitle}")
        st.write(f"Description: {glbdescription}")

    with col2:
        try:
            image = requests.get(glbimg).content
            st.image(image, width=200)
        except Exception as e:
            st.error("Failed to load Poster")

    # Predict genre on button click
            
    if st.button('Predict Genre'):
        with st.spinner("Predicting Genres..."):
            unique_predictions = set()
            for model_name, model_code in model_codes.items():
                model = load_model(model_code)
                result = model.predict(load_vectorizer().transform([clean_text(glbtitle + " " + glbdescription)]))
                result = load_encoder().inverse_transform(result)[0]
                unique_predictions.add(result.strip().title())
        
        st.markdown(html_string.format("Predicted Genres (Unique)", ", ".join(unique_predictions)), unsafe_allow_html=True)
        st.markdown(html_string.format("Actual Genre", actual_genre.strip().title()), unsafe_allow_html=True)
else:
    # Manual input for title and description
    title = st.text_input('Enter the movie title:')
    desc = st.text_area('Enter the movie Description:')

    # Predict genres on button click
    if st.button('Predict Genre'):
        with st.spinner("Predicting Genres..."):
            unique_predictions = set()
            for model_name, model_code in model_codes.items():
                model = load_model(model_code)
                result = model.predict(load_vectorizer().transform([clean_text(str(title) + " " + str(desc))]))
                result = load_encoder().inverse_transform(result)[0]
                unique_predictions.add(result.strip().title())
        
        st.markdown(html_string.format("Predicted Genre:", ", ".join(unique_predictions)), unsafe_allow_html=True)


# About the project section
st.header("About the Project")
st.markdown("""
<style>
.footer {
    position: relative;
    left: 0;
    bottom: 0;
    width: 100%;
    color: black;
    text-align: left;
}
.git {
    padding-top: 15px;
}
</style>
<div class="footer">
    This project is made using Streamlit and deployed on Heroku. <br>
    The movies in the search bar are taken from IMDB with the actual genres.<br>
    
</div>
""", unsafe_allow_html=True)

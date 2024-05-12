

st.write('# Sentiment Classifier')
st.write('---')
st.subheader('Enter your text and hashtags to analyze sentiment')
# User input
text = st.text_area("Enter your text:", height=100)
hash = st.hash_area("Enter hashtags:",height = 100)

text = text 
clean_text(str(text))
if st.button("Analyze Sentiment"):
  text_vector = vectorizer.transform([text])
  y_predict = model.predict(text_vector)
  if(y_predict == 0):
    st.write("You are feeling positive emotions, Live, Love, Laugh!")
  elif(y_predict == 1):
    st.write("You are feeling nothing, Better drink a cup of coffee!")
  else:
    st.write("You are feeling plenty of negative emotions, Eat a snack bar!")

    

# pip install streamlit
import pickle
import streamlit as st
# streamlit is aliased as st

# Load the Saved model

lr = pickle.load(open('lr_model.pkl','rb'))  # rb = read binary
dt = pickle.load(open('dt_model.pkl','rb'))
knn = pickle.load(open('knn_model.pkl','rb'))
rf = pickle.load(open('rf_model.pkl','rb'))

st.header('Iris Flower Classification ML Web App')

ml_model = ['Logistic Regression','KNeighbors Classifier','Decision Tree Classifier',
            'RandomForest Classifier']

option = st.sidebar.selectbox('Select one of the ML Model',ml_model)


sl = st.slider('Sepal Length',0.0,10.0)
sw = st.slider('Sepal Width',0.0,10.0)
pl = st.slider('Petal Length',0.0,10.0)
pw = st.slider('Petal Width',0.0,10.0)

test = [[sl,sw,pl,pw]]

st.write('Test data',test)

# Calling the ML Model for Prediction

if st.button('Run Classifier'):
    if option== 'Logistic Regression':
        st.success(lr.predict(test)[0])
    elif option== 'KNeighbors Classifier':
        st.success(knn.predict(test)[0])
    elif option== 'Decision Tree Classifier':
        st.success(dt.predict(test)[0])
    else:
        st.success(rf.predict(test)[0])



# To run the file in terminal, write the following
# streamlit run ml_cls_app.py
# To stop the server, write the following
# Ctrl + C
# To clear the screen
# cls




import streamlit as st
import pickle

from hello import dt_model

lin_model=pickle.load(open('dt_model.pkl','rb'))

def classify(num):
    if num==1:
        return 'you get the credit card'
    elif num ==0:
        return 'you can not get the credit card'

def main():
    st.title("Streamlit Tutorial")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Iris Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Decision tree']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)

    re=st.slider('reports', 0.0, 6.0)
    ac=st.slider('active', 0.0, 35.0)
    ma=st.slider('major', 0.0, 1.0)
    inc=st.slider('income', 0.0, 12.0)
    exp=st.slider('expenditure', 0.0, 1000.0)
    inputs=[[re,ac,ma,inc,exp]]
    if st.button('Classify'):
        if option=='Linear Regression':
            st.success(classify(dt_model.predict(inputs)))
        # elif option=='Logistic Regression':
        #     st.success(classify(log_model.predict(inputs)))
        # else:
        #    st.success(classify(svm.predict(inputs)))


if __name__=='__main__':
    main()

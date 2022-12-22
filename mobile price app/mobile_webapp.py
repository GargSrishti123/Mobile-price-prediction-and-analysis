import streamlit as st
import pickle
import numpy as np



dt = pickle.load(open('dt_model.pkl','rb'))
rf= pickle.load(open('rf_model.pkl','rb'))
xgb = pickle.load(open('xgboost_model.pkl','rb'))
ada = pickle.load(open('adaboost_model.pkl','rb'))


st.title("Mobile Price Prediction Web App")
html_temp = """
    <div style="background-color:lightgreen ;padding:8px">
    <h2 style="color:black;text-align:center;">Mobile Price Prediction</h2>
    </div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

activities = ['Decision Tree','Random Forest','XGBoost','AdaBoost']
option = st.sidebar.selectbox('Which regression model would you like to use?',activities)
st.subheader(option)

st.write("""###### For brand selecton select numbers from 0-7 as follows:
            0:Motorola, 1:One PLus, 2:Oppo, 3:Poco, 4:Realme, 5:Redmi, 6:Samsung, 7:Vivo """)

st.write("""###### For Price Range select numbers from 0-3 as follows:
            0:High_range, 1:Low_range, 2:Medium_range, 3:Very_high_range """)


brand = [0,1,2,3,4,5,6,7]
brand_option = st.sidebar.selectbox("Choose the Brand",brand)
#brand_option=float(brand_option)

p_range = [0,1,2,3]
prange_option = st.sidebar.selectbox("Choose the price range",p_range)
#prange_option = float(prange_option)

RAM = [2,3,4,6,8,12,16]
ram_option = st.sidebar.selectbox("Choose the RAM available in GB",RAM)
#ram_option = float(ram_option)
#st.subheader(ram_option)

cam_pixel = st.slider('Select Main Camera pixel', 8, 200)
front_pixel = st.slider('Select Front Camera pixel', 2, 64)
star = st.slider('Select star ratings', 1.0, 5.0)
user = st.slider('Select users rated', 0, 30000)
battery = st.slider('Select Battery (mAh)', 2000, 7000)
display = st.slider('Select display size in centimeters', 15.0, 17.0)


storage = [16,32,64,128,256,512]
storage_option= st.sidebar.selectbox("Choose the storage available in GB",storage)
#st.subheader(ram_option)

inputs=[[brand_option,storage_option,battery,cam_pixel,front_pixel,display,ram_option,star,user,prange_option]]

# st.write("type for brand",type(brand_option))
# st.write("type for storage0",type(storage_option))
# st.write("type for battery",type(battery))
# st.write("type for cp",type(cam_pixel))
# st.write("type for sel_p",type(front_pixel))
# st.write("type for display",type(display))
# st.write("type for ram",type(ram_option))
# st.write("type for star",type(star))
# st.write("type for user",type(user))
# st.write("type for price range",type(prange_option))

if st.button('Predict'):
    if option=='Decision Tree':
        st.success(dt.predict(inputs))
    elif option=='Random Forest':
        st.success(rf.predict(inputs))
    elif option=='XGBoost':
        st.success(xgb.predict(inputs))
    else:
        st.success(ada.predict(inputs))

import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import pickle
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

equip_dict = {'10003540':0,
                'P-1000-N002':2,
                'P-1000-N006':5,
                'P-3000-N006':7,
                'P-ZCBM-N002':9}
model = pickle.load(open('xgb_model.pkl', 'rb'))
df = pd.read_csv('Equipment_data.csv')

def finding_id(mydict,x):
    i=1
    for key in mydict.keys():
        if x==key:
            y = (mydict[key])
            return y
        elif i==len(mydict):
            return -1
        i=i+1



st.title("Predictive Maintenance\n\n\n")
st.sidebar.selectbox('Select Plant',['Plant A'])
status = st.sidebar.radio('Select:',('All Equipment','Equipment Number'))

if status=='All Equipment':
    pass
else:
    Equipment_ID = st.sidebar.selectbox('',list(equip_dict.keys()))

if(st.sidebar.button('Predict')):

    col1,col2 ,col3,col4= st.columns(4) 
    col1.success("Performance Forecast")
    col1.text("All Ok")
    col2.success("Maintenance Schedule")
    col2.text("All Ok")
    col3.success("Operating Parameters")
    col3.text("1 Item need attention")
    col4.success("Risk Of Breakdown")
    col4.text("1 Risk Detected")

    if status=='All Equipment':
        duedate={'Equipment Number':[],'Maintenance Due in (days)':[]}
        for e_id in equip_dict.keys():
            equip_id = finding_id(equip_dict,e_id)
            equip_data = df[df['EQUIPMENT_NUMBER']==equip_id].iloc[-1,:].values.reshape(1,-1)
            prediction = model.predict(equip_data)
            duedate['Maintenance Due in (days)'].append(abs(int(prediction)))
            duedate['Equipment Number'].append(e_id)

        st.dataframe(duedate)
#            st.success('For Equipment {} - Next Maintenance Order Due in {} days.'.format(e_id,abs(int(prediction))))
    else:
        equip_id = finding_id(equip_dict,Equipment_ID)
        equip_data = df[df['EQUIPMENT_NUMBER']==equip_id].iloc[-1,:].values.reshape(1,-1)
        prediction = model.predict(equip_data)
        st.text('Next Maintenance Order Due in {} days.'.format(abs(int(prediction))))



    
# Python script that runs with streamlit to create time series visualizations

import streamlit as st
import h5py
import numpy as np
import matplotlib.pyplot as plt
from config import load_config
paths = load_config(dataset_key='all')

f = h5py.File(paths['raw_data'] / 'data_preprocessed_view.hdf5','r')
time = f['Time']
wavelength = f['Wavelength']
channel_1 = f['Channel1']
channel_2 = f['Channel2']
channel_3 = f['Channel3']

ch1_dff = f['Ch1_dff']
ch2_dff = f['Ch2_dff']
ch3_dff = f['Ch3_dff']

# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="wide", page_title="HSFP data visualizations")
st.title('HSFP data visulizations')

# Define function for plotting spectrum, time, and intensity as a spectrogram
def spectrogram(data, x, y, c):
    f, ax = plt.subplots()
    i = ax.imshow(np.transpose(data), aspect='auto', vmin=c[0], vmax=c[1], extent=[np.min(time), np.max(time), 699, 400], cmap='gray')
    ax.set(xlabel='Time (sec)', ylabel='Wavelength (nm)', xlim=[x[0],x[1]], ylim=[y[0],y[1]])
    ax.grid(False)
    f.colorbar(i,ax=ax)
    st.pyplot(f)

# Set up spectrogram plots
row1_1, row1_2 = st.columns(2)
with row1_1:
    st.subheader('Intensity from camera sensor')
    x = st.slider("Select time range to display", value=[float(time[0]), float(time[-1])])
    y = st.slider("Select wavelength range to display", value=[400, 700])
with row1_2:
    st.write('Adjust each slider to tune the x, y, and colorbar scales of the spectrograms')

row2_1, row2_2, row2_3 = st.columns(3)
with row2_1:
    st.subheader('Channel 1 - 473nm excitation')
    c1 = st.slider('Select colorbar channel 1', value=[0, 2000])
    spectrogram(channel_1, x, y, c1)
with row2_2:
    st.subheader('Channel 2 - 561nm excitation')
    c2 = st.slider('Select colorbar channel 2', value=[0, 2000])
    spectrogram(channel_2, x, y, c2)
with row2_3:
    st.subheader('Channel 3 - 405nm excitation')
    c3 = st.slider('Select colorbar channel 3', value=[0, 2000])
    spectrogram(channel_3, x, y, c3)


# Define function for plotting spectrum, time, and dF/F as a spectrogram
def dff(data, x, y, c):
    f, ax = plt.subplots()
    i = ax.imshow(np.transpose(data), aspect='auto', vmin=c[0], vmax=c[1], extent=[np.min(time), np.max(time), 699, 400], cmap='seismic')
    ax.set(xlabel='Time (sec)', ylabel='Wavelength (nm)', xlim=[x[0],x[1]], ylim=[y[0],y[1]])
    ax.grid(False)
    f.colorbar(i,ax=ax)
    st.pyplot(f)

# Set up spectrogram plots for dff
row3_1, row3_2 = st.columns(2)
with row3_1:
    st.subheader('dF/F (%)')
    x = st.slider("Select time range to display for dF/F spectrograms", value=[float(time[0]), float(time[-1])])
    y = st.slider("Select wavelength range to display for dF/F spectrograms", value=[400, 700])
with row3_2:
    st.write('Adjust slider to tune the x, y, and colorbar scales of the spectrograms')

row4_1, row4_2, row4_3 = st.columns(3)
with row4_1:
    st.subheader('Channel 1 - 473nm excitation')
    c1 = st.slider('dF/F colorbar channel 1', value=[-10, 10])
    dff(ch1_dff, x, y, c1)
with row4_2:
    st.subheader('Channel 2 - 561nm excitation')
    c2 = st.slider('dF/F colorbar channel 2 excitation', value=[-10, 10])
    dff(ch2_dff, x, y, c2)
with row4_3:
    st.subheader('Channel 3 - 405nm excitation')
    c3 = st.slider('dF/F colorbar channel 3 excitation', value=[-10, 10])
    dff(ch3_dff, x, y, c3)



# Define function for plotting time series data
def timeseries(data,dff,w):
    # f,ax = plt.subplots()
    # ax.plot(time[:], data[:,w-400], linewidth=1)
    # ax.set(xlabel='Time (sec)', ylabel='Intensity', title='Wavelength: '+str(w)+'nm', xlim=[x2[0],x2[1]])
    # st.pyplot(f)
    f,ax = plt.subplots()
    ax.plot(time[:], dff[:,w-400], linewidth=1)
    ax.set(xlabel='Time (sec)', ylabel='dF/F (%)', xlim=[x2[0],x2[1]])
    st.pyplot(f)

# Set up plotting for time series 
row5_1, row5_2 = st.columns(2)
with row5_1:
    st.subheader('dF/F time series plots')
with row5_2:
    x2 = st.slider("Select time range", value=[float(time[0]), float(time[-1])])

row6_1, row6_2, row6_3 = st.columns(3)
with row6_1:
    w1 = st.slider('Select wavelength for channel 1', 400, 699)
    #w1 = st.multiselect('Select wavelengths to disply for channel 1',[450, 480, 535, 610])
    timeseries(channel_1,ch1_dff,w1)

with row6_2:
    w2 = st.slider('Select wavelength for channel 2', 400, 699)
    #w2 = st.multiselect('Select wavelengths to disply for channel 2',[450, 480, 535, 610])
    timeseries(channel_2,ch2_dff,w2)

with row6_3:
    w3 = st.slider('Select wavelength for channel 3', 400, 699)
    #w3 = st.multiselect('Select wavelengths to disply for channel 3',[450, 480, 535, 610])
    timeseries(channel_3,ch3_dff,w3)
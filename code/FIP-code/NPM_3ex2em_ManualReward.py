# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:34:14 2023

For 3CS-US Probablisic Pavlovian Conditioning

TrialType_
1:CS1 Rewarded
2-10:CS1 UnRewarded
11-15:CS2 Rewarded
16-20CS2 UnRewarded
21-29:CS3 Rewarded
30: CS3 UnRewarded

@author: kenta.hagihara
"""
#%% setup/imports
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import csv
import glob
import re
from scipy.optimize import curve_fit
import json
import pandas as pd
from scipy.stats import sem

import PreprocessingFunctions2 as pf

#plt.close('all')
# SaveDir=r''
# AnalDir=r'...\behavior'

session_id = 'FIP_836732_2025-12-03_10-23-43'

SaveDir = r'C:\output_data\results\results_' + session_id
AnalDir = r'C:\output_data' + os.sep + session_id + os.sep + 'behavior'
# AnalDir = r'C:\output_data\FIP_836732_2025-12-03_10-23-43\behavior'

os.makedirs(SaveDir, exist_ok=True)

# manually enter when the first reward trial that should actually be counted happened
first_rew_idx = 23           # 0 for 836733 12/3/25
                            # 23 for 836732 12/3/25
# for visualization
#Roi2Vis=[0,1,2]
Roi2Vis = [0]
AllPlot=0

# params for pre-processing
nFrame2cut = 100  #crop initial n frames
sampling_rate = 20 #individual channel (not total)
kernelSize = 1 #median filter
degree = 4 #polyfit
b_percentile = 0.70 #To calculare F0, median of bottom x%

sampling_rate=20
StimPeriod = 0.5 #sec for visualization`
preW=100 #nframes for PSTH
LickWindow=5.0 #sec window length for Consummatory/Omission licks

#%% load raw data
try: 
    file1  = glob.glob(AnalDir + os.sep + "FIP_DataIso_*")[0]
    
except Exception as e:
    file1  = glob.glob(AnalDir[0:-8] + 'fib' + os.sep + "FIP_DataIso_*")[0]
    file2 = glob.glob(AnalDir[0:-8] + 'fib' + os.sep + "FIP_DataG_*")[0]
    file3 = glob.glob(AnalDir[0:-8] + 'fib' + os.sep + "FIP_DataR_*")[0]   
    subjectID = AnalDir.split("\\")[3]
    
else:
    file2 = glob.glob(AnalDir + os.sep + "FIP_DataG_*")[0]
    file3 = glob.glob(AnalDir + os.sep + "FIP_DataR_*")[0]
    subjectID = AnalDir.split("\\")[3]

with open(file1) as f:
    reader = csv.reader(f)
    datatemp = np.array([row for row in reader])
    data1 = datatemp[1:,:].astype(np.float32)
    #del datatemp
    
with open(file2) as f:
    reader = csv.reader(f)
    datatemp = np.array([row for row in reader])
    data2 = datatemp[1:,:].astype(np.float32)
    #del datatemp
    
with open(file3) as f:
    reader = csv.reader(f)
    datatemp = np.array([row for row in reader])
    data3 = datatemp[1:,:].astype(np.float32)
    #del datatemp


TSfiles = glob.glob(AnalDir + os.sep + "TS_*")
TSdict = {}

for file_i in range(len(TSfiles)):
    fullpath_i=glob.glob(AnalDir + os.sep + "TS_*")[file_i]
    file_i_name=os.path.basename(fullpath_i)
    match = re.search(r'TS_(.*?)_',file_i_name)
    key = match.group(1)
    
    
    with open(fullpath_i, newline='') as file:
        csv_reader = csv.reader(file)
        try:
            has_header = csv.Sniffer().has_header(file.read(1024))
        except:
            has_header = False
                
    with open(fullpath_i) as f:
        reader = csv.reader(f)
        if has_header:
            next(reader) #skip header
        datatemp = np.array([row for row in reader])
        TSdict[key] = datatemp.astype(np.float32)
        
#%% Trim out initial manual rewards
TSdict['Reward'] = TSdict['Reward'][first_rew_idx:, :]


#%% Adjust recording end time
# in case acquisition halted accidentally
Length = np.amin([len(data1),len(data2),len(data3)])

data1 = data1[0:Length] #iso       Time*[TS,ROI0,ROI1,ROI2,..]
data2 = data2[0:Length] #signal
data3 = data3[0:Length] #Stim

PMts= data2[:,0] #SignalTS
time_seconds = np.arange(len(data1)) /sampling_rate
#%% Preprocess
Ctrl_dF_F=np.zeros((data1.shape[0],data1.shape[1]))
G_dF_F=np.zeros((data1.shape[0],data1.shape[1]))
R_dF_F=np.zeros((data1.shape[0],data1.shape[1]))

for ii in range(data2.shape[1]-1):
    Ctrl_dF_F[:,ii] = pf.tc_preprocess(data1[:,ii+1], nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)
    G_dF_F[:,ii] = pf.tc_preprocess(data2[:,ii+1] , nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)
    R_dF_F[:,ii] = pf.tc_preprocess(data3[:,ii+1] , nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)

#%% Extract frame stamps for trials
TSFramesdict={}

for k in TSdict:
    Temp = TSdict[k]
    TempFrames = np.empty(len(Temp))
    for ii in range(len(Temp)):
        idx = np.argmin(np.abs(data1[:,0] - Temp[ii,0]))
        TempFrames[ii] = idx
    
    TSFramesdict[k] = TempFrames

RewardFrames = TSFramesdict['Reward']
CS1Frames = TSFramesdict['CS1']
CS2Frames = TSFramesdict['CS2']
CS3Frames = TSFramesdict['CS3']
LickFrames = TSFramesdict['Lick']
#%% optional pupil tracking

if bool(glob.glob(AnalDir + os.sep + "PupilTracking*")) == True:
    #print('loading PupilTracking')
    file_Pupil = glob.glob(AnalDir + os.sep + "PupilTracking*")[0]
    file_EyeCam = glob.glob(AnalDir + os.sep + "FaceEyeCamera*.csv")[0]
    
    with open(file_Pupil) as f:
        reader = csv.reader(f)
        datatemp = np.array([row for row in reader])
        data_Pupil = datatemp[1:,:].astype(np.float32)
        del datatemp

    with open(file_EyeCam) as f:
        reader = csv.reader(f)
        datatemp = np.array([row for row in reader])
        data_EyeCam_time = datatemp[1:,:].astype(np.float32)
        del datatemp
        
    Length = np.amin([len(data_EyeCam_time),len(data_Pupil)])

    data_EyeCam_time = data_EyeCam_time[0:Length]
    data_Pupil = data_Pupil[0:Length]
        
    #data_EyeCam_time = (data_EyeCam_time - data_EyeCam_time[0])/1000 #ms to s 
    idx = np.argmin(np.abs(data_EyeCam_time[:] - data1[0,0])) #align to photometry start time
    data_EyeCam_time = (data_EyeCam_time - data_EyeCam_time[idx])/1000 #ms to s
#%% PLOT Entire Session
if AllPlot==1:
    gs = gridspec.GridSpec(6,8)
    plt.figure(figsize=(20, 8))
    plt.subplot(gs[0:3, 0:8])
    
    for ii in range(Ctrl_dF_F.shape[1]):
        plt.plot(time_seconds, Ctrl_dF_F[:,ii]*100 - ii*100, 'blue')
        plt.plot(time_seconds, G_dF_F[:,ii]*100 - ii*100, 'green')
        plt.plot(time_seconds, np.zeros(len(time_seconds))-ii*100,'--k')
        
    plt.plot(LickFrames/20, np.ones(len(LickFrames))*100, marker=3, markersize=10, color=[0, 0, 0, 0.5] ,label='Lick')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('dF/F (%)')
    plt.title("Whole Trace   SubjectID: " + subjectID)
    plt.xlim([0, time_seconds[-1]])
    plt.grid(True)
    
    
    for ii in range(len(RewardFrames)):
        plt.axvspan(RewardFrames[ii]/20, RewardFrames[ii]/20 + StimPeriod, color = [0, 0, 1, 0.4])
    
    for ii in range(len(CS1Frames)):
        plt.axvspan(CS1Frames[ii]/20, CS1Frames[ii]/20 + 20, color = [1, 0, 0, 0.4])
    
    for ii in range(len(CS2Frames)):
        plt.axvspan(CS2Frames[ii]/20, CS2Frames[ii]/20 + 20, color = [0, 1, 0, 0.4]) 
    
    for ii in range(len(CS3Frames)):
        plt.axvspan(CS3Frames[ii]/20, CS3Frames[ii]/20 + 20, color = [1, 0, 1, 0.4]) 
    
    plt.axvspan(RewardFrames[0]/20, RewardFrames[0]/20, color = [0, 0, 1, 0.4],label='Reward')
    plt.axvspan(CS1Frames[0]/20, CS1Frames[0]/20, color = [1, 0, 0, 0.4],label='CS1')
    plt.axvspan(CS2Frames[0]/20, CS2Frames[0]/20, color = [0, 1, 0, 0.4],label='CS2') 
    plt.axvspan(CS3Frames[0]/20, CS3Frames[0]/20, color = [1, 0, 1, 0.4],label='CS3')
    
    plt.legend()
    
    if bool(glob.glob(AnalDir + os.sep + "PupilTracking*")) == True:
        plt.subplot(gs[3, 0:8])
        plt.plot(data_EyeCam_time, data_Pupil, color=[0.4, 0.4, 0.4])   
        plt.ylabel('pixel')
        plt.xlim([0, time_seconds[-1]])
        plt.title('Pupil Diam.')
        plt.xlabel('second')

##
figT=plt.figure('Summary:' + AnalDir,figsize=(16, 16))
gs = gridspec.GridSpec(12,9)
plt.subplot(gs[0:4, 0:9])

for ii_ROI in range(len(Roi2Vis)):
    plt.plot(time_seconds, Ctrl_dF_F[:,Roi2Vis[ii_ROI]]*100 - ii_ROI*100, 'blue')
    plt.plot(time_seconds, G_dF_F[:,Roi2Vis[ii_ROI]]*100 - ii_ROI*100, 'green')
    plt.plot(time_seconds, R_dF_F[:,Roi2Vis[ii_ROI]]*100 - ii_ROI*100, 'magenta')
    plt.plot(time_seconds, np.zeros(len(time_seconds))-ii_ROI*100,'--k')
    
plt.plot(LickFrames/20, np.ones(len(LickFrames))*100, marker=3, markersize=10, color=[0, 0, 0, 0.5] ,label='Lick')

plt.xlabel('Time (seconds)')
plt.ylabel('dF/F (%)')
plt.title("SubjectID: " + subjectID + "  Date: " + os.path.basename(os.path.dirname(AnalDir)))
plt.xlim([0, time_seconds[-1]])
plt.grid(True)


for ii in range(len(RewardFrames)):
    plt.axvspan(RewardFrames[ii]/20, RewardFrames[ii]/20 + StimPeriod, color = [0, 0, 1, 0.4])

for ii in range(len(CS1Frames)):
    plt.axvspan(CS1Frames[ii]/20, CS1Frames[ii]/20 + 1, color = [1, 0, 0, 0.4])

for ii in range(len(CS2Frames)):
    plt.axvspan(CS2Frames[ii]/20, CS2Frames[ii]/20 + 1, color = [0, 1, 0, 0.4]) 

for ii in range(len(CS3Frames)):
    plt.axvspan(CS3Frames[ii]/20, CS3Frames[ii]/20 + 1, color = [1, 0, 1, 0.4]) 

plt.axvspan(RewardFrames[0]/20, RewardFrames[0]/20, color = [0, 0, 1, 0.4],label='Reward')
#plt.axvspan(CS1Frames[0]/20, CS1Frames[0]/20, color = [1, 0, 0, 0.4],label='CS1')
#plt.axvspan(CS2Frames[0]/20, CS2Frames[0]/20, color = [0, 1, 0, 0.4],label='CS2') 
#plt.axvspan(CS3Frames[0]/20, CS3Frames[0]/20, color = [1, 0, 1, 0.4],label='CS3')

plt.legend()

if bool(glob.glob(AnalDir + os.sep + "PupilTracking*")) == True:

    plt.plot(data_EyeCam_time, data_Pupil-(ii_ROI)*100-50, color=[0.4, 0.4, 0.4],label='PupilDiam.')
    plt.xlim([0, time_seconds[-1]])
    plt.legend()
    
#%% define PSTH functions (for multiple traces)
def PSTHmaker(TC, Stims, preW, postW):
    
    cnt = 0
    
    for ii in range(len(Stims)):
        if Stims[ii] - preW >= 0 and  Stims[ii] + postW < len(TC):
            
            A = int(Stims[ii]-preW) 
            B = int(Stims[ii]+postW)
            
            if cnt == 0:
                PSTHout = TC[A:B,:]
                cnt = 1
            else:
                PSTHout = np.dstack([PSTHout,TC[A:B,:]])
        else:
            if cnt == 0:
                PSTHout = np.zeros(preW+postW)
                cnt = 1
            #else:
                #PSTHout = np.dstack([PSTHout, np.zeros(preW+postW)])
    return PSTHout

#%% Define PSTH plotting function
def PSTHplot(PSTH, MainColor, SubColor, LabelStr):
    plt.plot(np.arange(np.shape(PSTH)[1])/20 - preW/sampling_rate, np.mean(PSTH.T,axis=1),label=LabelStr,color = MainColor)
    #plt.plot(np.arange(np.shape(PSTH)[1])/20 - 5, np.mean(PSTH.T,axis=1) + np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0]),color = SubColor, linestyle = "dotted")
    #plt.plot(np.arange(np.shape(PSTH)[1])/20 - 5, np.mean(PSTH.T,axis=1) - np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0]),color = SubColor, linestyle = "dotted")
    y11 =  np.mean(PSTH.T,axis=1) + np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0])
    y22 =  np.mean(PSTH.T,axis=1) - np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0])
    plt.fill_between(np.arange(np.shape(PSTH)[1])/20 - preW/sampling_rate, y11, y22, facecolor=SubColor, alpha=0.5)


#%% Define PSTH baseline subtraction (multi)
#dim0:trial, dim1:time
def PSTH_baseline(PSTH, preW):

    for ii in range(np.shape(PSTH)[2]):
        
        Trace_this = PSTH[:, :, ii]
        Trace_this_base = Trace_this[0:preW,:]
        Trace_this_subtracted = Trace_this - np.mean(Trace_this_base,axis=0)        
        
        if ii == 0:
            PSTHbase = Trace_this_subtracted
        else:
            PSTHbase = np.dstack([PSTHbase,Trace_this_subtracted])
    
    return PSTHbase


#%% Create a range array for the number of reward trials
#% Trial csv handing
ManualRewards = np.arange(len(RewardFrames))

#%% Calculate PSTH for signal around manual rewards
Psth_G_ManRew = PSTHmaker(G_dF_F*100, RewardFrames, 100, 300)
Psth_R_ManRew = PSTHmaker(R_dF_F*100, RewardFrames, 100, 300)
Psth_C_ManRew = PSTHmaker(Ctrl_dF_F*100, RewardFrames, 100, 300)
Psth_G_ManRew_base = PSTH_baseline(Psth_G_ManRew, 100)
Psth_R_ManRew_base = PSTH_baseline(Psth_R_ManRew, 100)
Psth_C_ManRew_base = PSTH_baseline(Psth_C_ManRew, 100)  

        
#%%  Calculate ymin and ymax
ymin=np.empty(len(Roi2Vis)+1)
ymax=np.empty(len(Roi2Vis)+1)
for ii in range(len(Roi2Vis)):
    ymax[ii]=np.max([
    np.max(np.mean(Psth_G_ManRew_base[:,Roi2Vis[ii],:],axis=1)),
    np.max(np.mean(Psth_R_ManRew_base[:,Roi2Vis[ii],:],axis=1))])
    
    ymin[ii]=np.min([
    np.min(np.mean(Psth_G_ManRew_base[:,Roi2Vis[ii],:],axis=1)),
    np.min(np.mean(Psth_R_ManRew_base[:,Roi2Vis[ii],:],axis=1))])

ymax[ii+1]=np.max([
np.max(np.mean(Psth_R_ManRew_base[:,0,:],axis=1)),
np.max(np.mean(Psth_R_ManRew_base[:,0,:],axis=1))])

ymin[ii+1]=np.min([
np.min(np.mean(Psth_R_ManRew_base[:,0,:],axis=1)),
np.min(np.mean(Psth_R_ManRew_base[:,0,:],axis=1))])

#%% PLOT summary of signal during reward
figT=plt.figure('Summary:' + AnalDir, figsize=(16, 16))

    
for ii in range(len(Roi2Vis)):
    plt.subplot(gs[4 + ii*2:4 + ii*2+2, 0:3])
    PSTHplot(Psth_G_ManRew_base[:,Roi2Vis[ii],:].T, "g", "darkgreen", "R+")
    PSTHplot(Psth_C_ManRew_base[:,Roi2Vis[ii],:].T, "b", "darkblue", "Iso_R+")
    # PSTHplot(Psth_G_CS3UR_base[:,Roi2Vis[ii],:].T, "m", "darkmagenta", "R-")
    # PSTHplot(Psth_C_CS3UR_base[:,Roi2Vis[ii],:].T, "k", "k", "Iso_R-")    
    plt.ylim([ymin[ii]*1.1, ymax[ii]*1.1])
    plt.xlim([-5,15])
    plt.grid(True)
    plt.title("Manual Rewards, ROI-Green: " + str(ii))
    plt.xlabel('Time - Tone (s)')
    plt.axvspan(0, 1.0, color = [1, 0, 1, 0.4])
    plt.axvspan(2.0, 2.5, color = [0, 0, 1, 0.4])
    

    plt.subplot(gs[4 + ii*2: 4 + ii*2+2, 3:6])
    PSTHplot(Psth_R_ManRew_base[:,Roi2Vis[ii],:].T, "g", "darkgreen", "R+")
    PSTHplot(Psth_C_ManRew_base[:,Roi2Vis[ii],:].T, "b", "darkblue", "Iso_R+")
    # PSTHplot(Psth_R_CS3UR_base[:,Roi2Vis[ii],:].T, "m", "darkmagenta", "R-")
    # PSTHplot(Psth_C_CS3UR_base[:,Roi2Vis[ii],:].T, "k", "k", "Iso_R-")    
    plt.ylim([ymin[ii]*1.1, ymax[ii]*1.1])
   # plt.ylim([-5, 5])
    plt.xlim([-5,15])
    plt.grid(True)
    plt.title("Manual Rewards, ROI-Red: " + str(ii))
    plt.xlabel('Time - Tone (s)')
    plt.ylabel('dF/F%')
    plt.axvspan(0, 1.0, color = [1, 0, 1, 0.4])
    plt.axvspan(2.0, 2.5, color = [0, 0, 1, 0.4])
    
    
    plt.tight_layout(
        # rect=[0, 0.03, 1, 0.95]
        )
    plt.subplots_adjust(hspace=2.0, wspace=0.4)
#%% print total number of reward trials
print('TotalRews: ' + str(np.sum([len(RewardFrames)])))


#%% Lick Quant

Lick_Rew=[]
Lick_Rew_post=[]
for ii in range(len(RewardFrames)):
    count1 = len([x for x in LickFrames if RewardFrames[ii] < x < RewardFrames[ii]+2*20])
    count2 = len([x for x in LickFrames if RewardFrames[ii]+2*20 < x < RewardFrames[ii]+7*20])
    Lick_Rew=np.append(Lick_Rew,count1) 
    Lick_Rew_post=np.append(Lick_Rew_post,count2) 

aveLick_Rew=np.mean(Lick_Rew)
semLick_Rew=np.std(Lick_Rew)/np.sqrt(len(Lick_Rew))

aveLick_Rew_post=np.mean(Lick_Rew_post)
semLick_Rew_post=np.std(Lick_Rew_post)/np.sqrt(len(Lick_Rew_post))

#%% PLOT lick quantification
figT=plt.figure('Summary:' + AnalDir, figsize=(16,16))

plt.subplot(gs[10:12, 6:9])
#plt.plot(Lick_Rew,label='Anticipatory')
plt.plot(Lick_Rew_post,label='Consummatory Licks')
plt.plot(ManualRewards, Lick_Rew_post[ManualRewards.astype(int)], '.',color='blue',markersize=10, label='Rewarded')
# plt.plot(UnRewardedCS3ind, Lick_CS3_post[UnRewardedCS3ind.astype(int)], '.',color='Red',markersize=10,label='UnRewarded')
plt.xlabel('trial #')
plt.title('Manual Rewards Consummatory Licks:' + str(round(aveLick_Rew_post,2)) + '+-' +str(round(semLick_Rew_post,2)))
plt.legend(fontsize=7)
bottom, top = plt.ylim()
lick_min = np.min(bottom, 0)
plt.ylim((lick_min, top+10))
plt.subplots_adjust(hspace = 0.5, wspace=0.25)
plt.tight_layout()

#%% Save summary image
aDate=os.path.basename(os.path.dirname(AnalDir))
plt.savefig(SaveDir + os.sep + 'Summary_' + subjectID + '_' + aDate + '.pdf')
#if bool(glob.glob(AnalDir + os.sep + "TrialN_*")) == True:

 
#%% dF/F trial-by-trial quant
CSall=np.sort(np.hstack(TSFramesdict['Reward']))

# Resp_Cue = np.empty((len(CSall),Ctrl_dF_F.shape[1])) # Cue:CS onset-offset (1s)
Resp_Rew = np.empty((len(CSall),Ctrl_dF_F.shape[1]))  # Reward: Reward onset to +3s
Resp_Tail = np.empty((len(CSall),Ctrl_dF_F.shape[1]))  # tail:next cue - 2sec to next trial 
Resp_base = np.empty((len(CSall),Ctrl_dF_F.shape[1]))  # base:-2s-0ms
# Resp_Cue_based = np.empty((len(CSall),Ctrl_dF_F.shape[1]))  # baseline subtracted
Resp_Rew_based = np.empty((len(CSall),Ctrl_dF_F.shape[1]))  # 
Resp_Tail_based = np.empty((len(CSall),Ctrl_dF_F.shape[1]))  # 

TC=G_dF_F*100 # At StimRig, GreenChannel

for ii in range(len(CSall)):
    # Resp_Cue[ii,:] = np.mean(TC[int(CSall[ii]+1):int(CSall[ii]+20),:],axis=0)
    Resp_Rew[ii,:] = np.mean(TC[int(CSall[ii]+1):int(CSall[ii]+100),:],axis=0)
    if ii==len(CSall)-1:
        Resp_Tail[ii,:] = np.mean(TC[int(CSall[ii]+200):int(CSall[ii]+240),:],axis=0)
    else:
        Resp_Tail[ii,:] = np.mean(TC[int(CSall[ii+1]-40):int(CSall[ii+1]),:],axis=0)

    Resp_base[ii,:] = np.mean(TC[int(CSall[ii]-40):int(CSall[ii]),:],axis=0)

    #relative to local mean
    # Resp_Cue_based[ii,:] = Resp_Cue[ii,:] - Resp_base[ii,:] 
    Resp_Rew_based[ii,:] = Resp_Rew[ii,:] - Resp_base[ii,:] 
    Resp_Tail_based[ii,:] = Resp_Tail[ii,:] - Resp_base[ii,:] 

  
#%% PLOT trial-by-trial quant


for ROIii in Roi2Vis:
    #[ymin,ymax]=[-30,110]
    
    plt.figure(figsize=(10,5))    
   
    
    #% Time from previous reward
    TimeFromPR = TSFramesdict['Reward']-np.roll(TSFramesdict['Reward'],1)
    TimeFromPR = TimeFromPR/sampling_rate # in seconds
    
    RewResp=np.empty(len(TSFramesdict['Reward']))
    RewResp_base=np.empty(len(TSFramesdict['Reward']))
    
    TC=G_dF_F*100
    
    # for ii in range(len(TSFramesdict['Reward'])):
    #     RewResp[ii]=np.mean(TC[int(TSFramesdict['Reward'][ii]):int(TSFramesdict['Reward'][ii])+60, 1])
    #     RewResp_base[ii]=RewResp[ii] - np.mean(TC[int(TSFramesdict['Reward'][ii]-80):int(TSFramesdict['Reward'][ii])-40, 1])
   
    # Replaced Kenta's code (above, commented out) with this since original code
    # didn't calculate RewResp for each ROI separately.
    TC_col = TC[:, ROIii]
    for ii, reward_frame in enumerate(TSFramesdict['Reward']):
        reward_frame = int(reward_frame)
    
        # Mean response in 60-frame window after reward
        RewResp[ii] = np.mean(TC_col[reward_frame : reward_frame + 60])
    
        # Baseline: mean of 40-frame window 80–40 frames before reward
        baseline = np.mean(TC_col[reward_frame - 80 : reward_frame - 40])
    
        # Baseline-corrected response
        RewResp_base[ii] = RewResp[ii] - baseline
    
   
    
    '''
    plt.figure()    
    plt.scatter(TimeFromPR[1:],RewResp[1:])
    #plt.scatter(TimeFromPR[1:],RewResp_base[1:])
    plt.ylabel('dF/F %')
    plt.xlabel('Time from previous R (s)')
    '''
   
    # Rall=np.sort(np.hstack([Mat_CS3R]))
    # CS3Rind=np.where(np.isin(Rall, Mat_CS3R))[0]
    
    CS3RewFrames = CS3Frames + 40
    tolerance = 5 
    diff_frames = np.abs(RewardFrames[:, None] - CS3RewFrames[None, :])
    mask = np.any(diff_frames <= tolerance, axis=1)
    CS3Rind = np.where(mask)[0]
    
    
    #plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(TimeFromPR[1:],RewResp[1:],c=[0,0,1,0.5],label='Manual_Reward')
    plt.ylabel('dF/F %')
    plt.xlabel('Time from previous R (s)')
    plt.title('ROI ' + str(ROIii) + ': Response vs Time from previous R')
    plt.legend()
    
    #% TrialN 
    #plt.figure()    
    plt.subplot(1,2,2)
    plt.plot(RewResp, 'g')
    plt.ylabel('dF/F %')
    plt.xlabel('#Rewarded Trial')
    plt.title('ROI ' + str(ROIii) + ': Response across rewards')
    
    plt.tight_layout(
        # rect=[0, 0.03, 1, 0.95]
        )
    plt.subplots_adjust(hspace=0.55, wspace=0.15)
    
    plt.savefig(SaveDir + os.sep + 'ROI-' + str(ROIii) + '_prev-trial_summary.pdf')
    



#%% PLOT ReactionTime From reward

RewardTime= TSdict["Reward"]
LickTime= TSdict["Lick"]

RewardedLickFrames = np.empty(len(RewardTime))
RewardRT = np.empty(len(RewardTime))

for ii in range(len(RewardTime)):
    idx = np.argmin(np.abs(LickTime[:,0] - RewardTime[ii,0]))
    if LickTime[idx,0] - RewardTime[ii,0] <= 0:
              idx = idx + 1
    try:
        RewardedLickFrames[ii] = idx
        RewardRT[ii] = LickTime[idx,0] - RewardTime[ii,0]
    except:
        print("skipped: " + str(idx))

plt.figure(figsize=(12, 4))
plt.subplot(1,3,1)
plt.plot(RewardRT/1000)
plt.ylabel('ReactionTime to Rew (s)')
plt.xlabel('Reward#')
plt.title('med RT:' + str(np.median(RewardRT/1000)) + ' (s)')

plt.subplot(1,3,2)
plt.hist(RewardRT/1000)
plt.subplots_adjust(hspace = 0.7, wspace=0.25)
plt.ylabel('count')
plt.xlabel('ReactionTime to Rew (s)')

plt.subplot(1,3,3)
plt.hist(RewardRT[RewardRT<500]/1000)
plt.subplots_adjust(hspace = 0.7, wspace=0.25)
plt.ylabel('count')
plt.xlabel('RT to R <0.5s')

plt.savefig(SaveDir + os.sep + 'reaction_time.pdf')
#to assign to df,
#Anti-Lick responses to CS


#%% PreciseTiming RewardResponse

def find_closest_larger_elements(arr1, arr2):
    arr2_sorted = np.sort(arr2)
    result = []

    for element in arr1:
        # Find the index of the smallest element in arr2_sorted that is larger than element
        idx = np.searchsorted(arr2_sorted, element, side='right')
        
        if idx < len(arr2_sorted):
            result.append(arr2_sorted[idx])
        else:
            result.append(None)  # Or handle the case where no larger element is found

    return np.array(result)


RewardConsumption = find_closest_larger_elements(RewardFrames, LickFrames)

RewardConsumption = [x for x in RewardConsumption if x is not None] # added by Carrie 

Psth_G_RewardC = PSTHmaker(G_dF_F*100, RewardConsumption, 100, 300)
Psth_C_RewardC = PSTHmaker(Ctrl_dF_F*100, RewardConsumption, 100, 300)
Psth_R_RewardC = PSTHmaker(R_dF_F*100, RewardConsumption, 100, 300)
Psth_G_RewardC_base = PSTH_baseline(Psth_G_RewardC, 100)
Psth_C_RewardC_base = PSTH_baseline(Psth_C_RewardC, 100)
Psth_R_RewardC_base = PSTH_baseline(Psth_R_RewardC, 100)

# plt.figure()
# PSTHplot(Psth_R_RewardC_base[:,0,:].T, "darkred", "magenta",[])
# #PSTHplot(Psth_C_RewardC_base[:,0,:].T, "b", "darkblue",[])
# plt.axvspan(0, 2.5, color = [0, 0, 1, 0.2])
# #plt.axvspan(2.0, 2.5, color = [0, 0, 1, 0.4])
# plt.grid(True)
# plt.xlim([-5, 15])
# plt.title('aligned reward response')


#%% PLOT reward aligned response
plt.figure(figsize=(10,5))

# ROI O:
plt.subplot(2, 2, 1)
#PSTHplot(Psth_R_RewardC_base[:,0,:].T, "darkred", "magenta", "red")
PSTHplot(Psth_G_RewardC_base[:,0,:].T, "g", "darkgreen", "green")
PSTHplot(Psth_C_RewardC_base[:,0,:].T, "k", "k", "isos")
plt.ylim([ymin[0]*1.3, ymax[0]*1.3])
plt.xlim([-5,15])
plt.grid(True)
plt.title("ROI 0 Green: aligned reward response")
plt.xlabel('Time from reward delivery (s)')
plt.axvspan(0, 2.5, color = [0, 0, 1, 0.2])
    
plt.subplot(2, 2, 2)
PSTHplot(Psth_R_RewardC_base[:,0,:].T, "darkred", "magenta", "red")
#PSTHplot(Psth_G_RewardC_base[:,0,:].T, "g", "darkgreen", "green")
PSTHplot(Psth_C_RewardC_base[:,0,:].T, "k", "k", "isos")
plt.ylim([ymin[0]*1.3, ymax[0]*1.3])
plt.xlim([-5,15])
plt.grid(True)
plt.title("ROI 0 Red: aligned reward response")
plt.xlabel('Time from reward delivery (s)')
plt.axvspan(0, 2.5, color = [0, 0, 1, 0.2])
    


# # ROI 1:
# plt.subplot(2, 2, 3)
# #PSTHplot(Psth_R_RewardC_base[:,1,:].T, "darkred", "magenta", "red")
# PSTHplot(Psth_G_RewardC_base[:,1,:].T, "g", "darkgreen", "green")
# PSTHplot(Psth_C_RewardC_base[:,1,:].T, "k", "k", "isos")
# plt.ylim([ymin[0]*1.1, ymax[0]*1.1])
# plt.xlim([-5,15])
# plt.grid(True)
# plt.title("ROI 1 Green: aligned reward response")
# plt.xlabel('Time from reward delivery (s)')
# plt.axvspan(0, 2.5, color = [0, 0, 1, 0.2])
    
# plt.subplot(2, 2, 4)
# PSTHplot(Psth_R_RewardC_base[:,1,:].T, "darkred", "magenta", "red")
# #PSTHplot(Psth_G_RewardC_base[:,1,:].T, "g", "darkgreen", "green")
# PSTHplot(Psth_C_RewardC_base[:,1,:].T, "k", "k", "isos")
# plt.ylim([ymin[0]*1.1, ymax[0]*1.1])
# plt.xlim([-5,15])
# plt.grid(True)
# plt.title("ROI 1 Red: aligned reward response")
# plt.xlabel('Time from reward delivery (s)')
# plt.axvspan(0, 2.5, color = [0, 0, 1, 0.2])

plt.tight_layout(
    # rect=[0, 0.03, 1, 0.95]
    )
plt.subplots_adjust(hspace=0.5, wspace=0.15)

plt.savefig(SaveDir + os.sep + 'reward_aligned_response.pdf')




#%% Messing around, larger plot labels

plt.figure(figsize=(6, 2))

for ii_ROI in range(len(Roi2Vis)):
    plt.plot(time_seconds, Ctrl_dF_F[:,Roi2Vis[ii_ROI]]*100 - ii_ROI*100, 'blue')
    plt.plot(time_seconds, G_dF_F[:,Roi2Vis[ii_ROI]]*100 - ii_ROI*100, 'green')
    plt.plot(time_seconds, R_dF_F[:,Roi2Vis[ii_ROI]]*100 - ii_ROI*100, 'magenta')
    plt.plot(time_seconds, np.zeros(len(time_seconds))-ii_ROI*100,'--k')
    
plt.plot(LickFrames/20, np.ones(len(LickFrames))*100, marker=3, markersize=10, color=[0, 0, 0, 0.5] ,label='Lick')

plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('dF/F (%)')
plt.title("SubjectID: " + subjectID + "  Date: " + os.path.basename(os.path.dirname(AnalDir)))
plt.xlim([0, time_seconds[-1]])
plt.grid(True)


for ii in range(len(RewardFrames)):
    plt.axvspan(RewardFrames[ii]/20, RewardFrames[ii]/20 + StimPeriod, color = [0, 0, 1, 0.4])

for ii in range(len(CS1Frames)):
    plt.axvspan(CS1Frames[ii]/20, CS1Frames[ii]/20 + 1, color = [1, 0, 0, 0.4])

for ii in range(len(CS2Frames)):
    plt.axvspan(CS2Frames[ii]/20, CS2Frames[ii]/20 + 1, color = [0, 1, 0, 0.4]) 

for ii in range(len(CS3Frames)):
    plt.axvspan(CS3Frames[ii]/20, CS3Frames[ii]/20 + 1, color = [1, 0, 1, 0.4]) 

plt.axvspan(RewardFrames[0]/20, RewardFrames[0]/20, color = [0, 0, 1, 0.4],label='Reward')

#%% larger plot labels
plt.figure(figsize=(6,2))

    
for ii in range(len(Roi2Vis)):
    plt.subplot(1, 2, 1)
    PSTHplot(Psth_G_ManRew_base[:,Roi2Vis[ii],:].T, "g", "darkgreen", "R+")
   # PSTHplot(Psth_C_ManRew_base[:,Roi2Vis[ii],:].T, "b", "darkblue", "Iso_R+")
    # PSTHplot(Psth_G_CS3UR_base[:,Roi2Vis[ii],:].T, "m", "darkmagenta", "R-")
    # PSTHplot(Psth_C_CS3UR_base[:,Roi2Vis[ii],:].T, "k", "k", "Iso_R-")    
    plt.ylim([ymin[ii]*1.1, ymax[ii]*1.1])
    plt.xlim([-5,15])
    plt.grid(True)
    plt.title("GRAB-DA3m - reward aligned")
    plt.xlabel('Time - Tone (s)')
   # plt.axvspan(0, 2.5, color = [1, 0, 1, 0.4])
    plt.axvspan(0, 2.5, color = [0, 0, 1, 0.4])
    

    plt.subplot(1, 2, 2)
    PSTHplot(Psth_R_ManRew_base[:,Roi2Vis[ii],:].T, "m", "magenta", "R+")
  # PSTHplot(Psth_C_ManRew_base[:,Roi2Vis[ii],:].T, "b", "darkblue", "Iso_R+")
    # PSTHplot(Psth_R_CS3UR_base[:,Roi2Vis[ii],:].T, "m", "darkmagenta", "R-")
    # PSTHplot(Psth_C_CS3UR_base[:,Roi2Vis[ii],:].T, "k", "k", "Iso_R-")    
    plt.ylim([ymin[ii]*1.1, ymax[ii]*1.1])
   # plt.ylim([-5, 5])
    plt.xlim([-5,15])
    plt.grid(True)
    plt.title("D1-MSN OCaMP - reward aligned")
    plt.xlabel('Time - Tone (s)')
    plt.ylabel('dF/F%')
  #  plt.axvspan(0, 1.0, color = [1, 0, 1, 0.4])
    plt.axvspan(0, 2.5, color = [0, 0, 1, 0.4])
    
    
    plt.tight_layout(
        # rect=[0, 0.03, 1, 0.95]
        )
 #   plt.subplots_adjust(hspace=2.0, wspace=3.4)
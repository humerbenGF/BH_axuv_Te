# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:01:27 2024

@author: xiande.feng

This code is an updated version of axuv_Te_function_with_uncertainty_gftools.py

the new code used new coefficient to calculate Te


"""

import sys
sys.path.insert(0, "/home/mean value and uncertainty")
sys.path.insert(0, "/home/mean value and uncertainty - shots-19004-20470")
#import my_manifests as mm
import GF_data_tools as gdt

import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import savgol_filter

from scipy import integrate as spi
from scipy import signal

def dedroop_signal(signal, time, tau):
	dd_deriv = np.gradient(signal, time) + signal*(1./float(tau))
	corrected_signal = spi.cumtrapz(dd_deriv, x=time, initial=0)
	return corrected_signal
#plt.close('all')

#--------------------------------------------------------------------#
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#

def axuv_Te_with_error(shot):

    shot_n = shot
    shot_f = str(1000*(shot_n//1000))
    
    #filter parameter settings
    
    filter_points=101 #101
    filter_order=4
    
    #------get the raw data for different shots, change the diagnostic configuration (data location) at shot 19004
    if shot_n>18399 and shot_n<19004:
        
        F_Al_3um=5.897
        F_Mylar_6um=1.07
        F_Mylar_NO=6.99  

        layer_list = ['digitizer/AXUV051_0060MY/measured_volt',
                      'digitizer/AXUV052_0130MY/measured_volt',
                      'digitizer/AXUV053_0029AL/measured_volt',
                      'digitizer/AXUV054_0000NO/measured_volt',
                      'mirnov_combined/plasma_current_spline']

        q_options = {'experiment':'pi3b',
                    'manifest': 'default',
                    'shot': shot_n,
                    'layers': layer_list, 
                    'nodisplay': True}

        data = gdt.fetch_data.run_query(q_options) # load the data
        axuv_my06, axuv_my13, axuv_Al03, axuv_no_filter, plasma_current = data['waves']

        #----filter the raw signal
        axuv_my06_sig=axuv_my06
        axuv_my06_sig_filter=F_Mylar_6um*savgol_filter(np.asarray(np.asarray(axuv_my06_sig)), filter_points, filter_order, mode='nearest')
        axuv_my13_sig=axuv_my13
        axuv_my13_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_my13_sig)), filter_points, filter_order, mode='nearest')
        axuv_Al03_sig=axuv_Al03
        axuv_Al03_sig_filter=F_Al_3um*savgol_filter(np.asarray(np.asarray(axuv_Al03_sig)), filter_points, filter_order, mode='nearest')
        axuv_no_filter_sig=axuv_no_filter
        axuv_no_filter_sig_filter=F_Mylar_NO*savgol_filter(np.asarray(np.asarray(axuv_no_filter_sig)), filter_points, filter_order, mode='nearest')

    elif shot_n>=19004 and shot_n<19127:  #swap channel 1 and 3
        #  relative calibration factor for each channel, could be stored in a yaml file
        F_Mylar_6um=1/1.08319  
        F_Mylar_13um=1/1.12066
        F_Mylar_22um=1/2.5534
        F_Mylar_NO=1/0.007935
        
        #-------get the raw data of AXUV Te diagnostics from aurora 
        
        layer_list = ['digitizer/AXUV021_0006MY/measured_volt',
                      'digitizer/AXUV022_0013MY/measured_volt',
                      'digitizer/AXUV023_0025MY/measured_volt',
                      'digitizer/AXUV024_0000NO/measured_volt',
                      'mirnov_combined/plasma_current_spline']

        q_options = {'experiment':'pi3b',
                    'manifest': 'default',
                    'shot': shot_n,
                    'layers': layer_list, 
                    'nodisplay': True}

        data = gdt.fetch_data.run_query(q_options) # load the data
        axuv_my06, axuv_my13, axuv_my22, axuv_no_filter, plasma_current = data['waves']

        #----filter the raw signal
        axuv_my06_sig=F_Mylar_6um*np.asarray(axuv_my06)
        axuv_my06_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_my06_sig)), filter_points, filter_order, mode='nearest')
        axuv_my13_sig=F_Mylar_13um*np.asarray(axuv_my13)
        axuv_my13_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_my13_sig)), filter_points, filter_order, mode='nearest')
        axuv_my22_sig=F_Mylar_22um*np.asarray(axuv_my22)
        axuv_my22_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_my22_sig)), filter_points, filter_order, mode='nearest')
        axuv_no_filter_sig=F_Mylar_NO*np.asarray(axuv_no_filter)
        axuv_no_filter_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_no_filter_sig)), filter_points, filter_order, mode='nearest')

    elif shot_n>=19127 and shot_n<19200:  
        
        F_Mylar_6um=1/1.08319   
        F_Mylar_13um=1/10.27267
        F_Mylar_22um=1/23.80258
        F_Mylar_NO=1/0.007935

        layer_list = ['digitizer/AXUV021_0060MY/measured_volt',
                      'digitizer/AXUV022_0013MY/measured_volt',
                      'digitizer/AXUV023_0025MY/measured_volt',
                      'digitizer/AXUV024_0000NO/measured_volt',
                      'mirnov_combined/plasma_current_spline']

        q_options = {'experiment':'pi3b',
                    'manifest': 'default',
                    'shot': shot_n,
                    'layers': layer_list, 
                    'nodisplay': True}

        data = gdt.fetch_data.run_query(q_options) # load the data
        axuv_my06, axuv_my13, axuv_my22, axuv_no_filter, plasma_current = data['waves']

        #----filter the raw signal
        axuv_my06_sig=F_Mylar_6um*np.asarray(axuv_my06)
        axuv_my06_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_my06_sig)), filter_points, filter_order, mode='nearest')
        axuv_my13_sig=F_Mylar_13um*np.asarray(axuv_my13)
        axuv_my13_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_my13_sig)), filter_points, filter_order, mode='nearest')
        axuv_my22_sig=F_Mylar_22um*np.asarray(axuv_my22)
        axuv_my22_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_my22_sig)), filter_points, filter_order, mode='nearest')
        axuv_no_filter_sig=F_Mylar_NO*np.asarray(axuv_no_filter)
        axuv_no_filter_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_no_filter_sig)), filter_points, filter_order, mode='nearest')

    elif shot_n>=19200 and shot_n < 20470:  #swap channel 2 and 4 from shot 19200, put diode 2 pin output to amp 4 and diode 4 output to amp2. Set amp2 gain to 104 low noise and all the rest amps to 106 gain low noise mode 
        #  relative calibration factor for each channel, could be stored in a yaml file
        F_Mylar_6um=1/0.9722    #(gain(V/A):97228.21308697807, error (%) 0.5817694172894263)
        F_Mylar_13um=1/1.0639   #(gain(V/A):106393.84585288126, error (%) 4.503145662103653)
        F_Mylar_22um=1/2.4457   #(gain(V/A):244575.09675280697, error (%) 1.8059440273087297)
        F_Mylar_NO=1/0.007935   #(gain(V/A):793.582790500511, error (%) )

        #-------get the raw data of AXUV Te diagnostics from aurora 

        layer_list = ['digitizer/AXUV021_0006MY/measured_volt',
                      'digitizer/AXUV022_0013MY/measured_volt',
                      'digitizer/AXUV023_0025MY/measured_volt',
                      'digitizer/AXUV024_0000NO/measured_volt',
                      'mirnov_combined/plasma_current_spline']

        q_options = {'experiment':'pi3b',
                    'manifest': 'default',
                    'shot': shot_n,
                    'layers': layer_list, 
                    'nodisplay': True}

        data = gdt.fetch_data.run_query(q_options) # load the data
        axuv_my06, axuv_my13, axuv_my22, axuv_no_filter, plasma_current = data['waves']

        #----filter the raw signal
        axuv_my06_sig=F_Mylar_6um*np.asarray(axuv_my06)
        axuv_my06_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_my06_sig)), filter_points, filter_order, mode='nearest')
        axuv_my13_sig=F_Mylar_13um*np.asarray(axuv_my13)
        axuv_my13_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_my13_sig)), filter_points, filter_order, mode='nearest')
        axuv_my22_sig=F_Mylar_22um*np.asarray(axuv_my22)
        axuv_my22_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_my22_sig)), filter_points, filter_order, mode='nearest')
        axuv_no_filter_sig=F_Mylar_NO*np.asarray(axuv_no_filter)
        axuv_no_filter_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_no_filter_sig)), filter_points, filter_order, mode='nearest')


    elif  shot_n >=20470 :  # ch1 10^5 gain, ch2-3, 10^6 gain, ch4 10^4 gain
        #  relative calibration factor for each channel, could be stored in a yaml file
        F_Mylar_6um=1/0.0925    #(gain(V/A):9250.364568925666, error (%) 0.014)

        F_Mylar_13um=1/0.98075  #(gain(V/A):98075.860748291, error (%) 0.016)
        F_Mylar_22um=1/2.45174   #(gain(V/A):245174.9954254315, error (%) 0.021)
        F_Mylar_NO=1/0.0214   #(gain(V/A): 2141.3887119730334, error (%) )

        #-------get the raw data of AXUV Te diagnostics from aurora 

        layer_list = ['digitizer/AXUV021_0006MY/measured_volt',
                      'digitizer/AXUV022_0013MY/measured_volt',
                      'digitizer/AXUV023_0025MY/measured_volt',
                      'digitizer/AXUV024_0000NO/measured_volt',
                      'mirnov_combined/plasma_current_spline']
        # layer_list = ['digitizer/AXUV021_0006MY/measured_volt',
        #               'digitizer/AXUV022_0013MY/measured_volt',
        #               'digitizer/AXUV023_0025MY/measured_volt',
        #               'digitizer/AXUV024_0000NO/measured_volt']
        q_options = {'experiment':'pi3b',
                    'manifest': 'default',
                    'shot': shot_n,
                    'layers': layer_list, 
                    'nodisplay': True}

        data = gdt.fetch_data.run_query(q_options) # load the data
        axuv_my06, axuv_my13, axuv_my22, axuv_no_filter, plasma_current = data['waves']
        #axuv_my06, axuv_my13, axuv_my22, axuv_no_filter = data['waves']

        #----filter the raw signal
        axuv_my06_sig=F_Mylar_6um*np.asarray(axuv_my06)
        axuv_my06_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_my06_sig)), filter_points, filter_order, mode='nearest')
        axuv_my13_sig=F_Mylar_13um*np.asarray(axuv_my13)
        axuv_my13_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_my13_sig)), filter_points, filter_order, mode='nearest')
        axuv_my22_sig=F_Mylar_22um*np.asarray(axuv_my22)
        axuv_my22_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_my22_sig)), filter_points, filter_order, mode='nearest')
        axuv_no_filter_sig=F_Mylar_NO*np.asarray(axuv_no_filter)
        axuv_no_filter_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_no_filter_sig)), filter_points, filter_order, mode='nearest')


    # elif  shot_n >=20939:
    #     #  relative calibration factor for each channel, could be stored in a yaml file
    #     F_Mylar_6um=1/0.0925    #(gain(V/A):9250.364568925666, error (%) 0.014)

    #     F_Mylar_13um=1/0.98075  #(gain(V/A):98075.860748291, error (%) 0.016)
    #     F_Mylar_22um=1/2.45174   #(gain(V/A):245174.9954254315, error (%) 0.021)
    #     F_Mylar_NO=1/0.02141   #(gain(V/A): 2141.3887119730334, error (%) )
        
    #     shotFolder = 'J:/' + str(shot_n).zfill(6) + '/'          #Pi3 data (digitizer data folder) in my case it's J:/ drive
    #     filePath = shotFolder + 'GFDIG0149.CSV'
    #     data = pd.read_csv(filePath, sep=',', header=1, skiprows=2)

    #     data.drop(['Unnamed: 0'], axis =1, inplace=True)
    #     dt = 8E-7
    #     t0 = -0.00115
    #     fs = 1/dt
    #     #cutoff = 10e3  # cut off 10khz
    #     #cutoff = 10e3
    #     #order = 5

    #     DC=0
    #     y = data['Y[0]']
    #     nump = len(y)
    #     t = np.linspace(t0, t0+dt*(nump-1), nump)

    #     y_0 = data['Y[0]']

    #     y_1 = data['Y[1]']

    #     y_2 = data['Y[2]']

    #     y_3 = data['Y[3]']
    #     #y_nn = (y-2048)/(2*2047)

    #     gain=1.0

    #     axuv_my06 = ((y_0/4096)-.5) / gain
    #     axuv_my13 = ((y_1/4096)-.5) / gain
    #     axuv_my22 = ((y_2/4096)-.5) / gain
    #     axuv_no_filter = ((y_3/4096)-.5) / gain
    #     #y_s = butter_lowpass_filtfilt(y_n, cutoff, fs)
    #     axuv_my06 = axuv_my06-axuv_my06[0:200].mean()
    #     axuv_my13 = axuv_my13-axuv_my13[0:200].mean()
    #     axuv_my22 = axuv_my06-axuv_my22[0:200].mean()
    #     axuv_no_filter = axuv_no_filter-axuv_no_filter[0:200].mean()


    #     # axuv_my06 = axuv_my06 + integrate.simpson(axuv_my06,t,dt)/0.022
    #     # print(integrate.simpson(axuv_my06,t)/0.022)
    #     # axuv_my13 = axuv_my13 + integrate.simpson(axuv_my13,t,dt)/0.022
    #     # axuv_my22 = axuv_my22 + integrate.simpson(axuv_my22,t,dt)/0.022
    #     # axuv_no_filter = axuv_no_filter +integrate.simpson(axuv_no_filter, t,dt)/0.022

    #     axuv_my06 = dedroop_signal(axuv_my06, t, 0.022)
    #     axuv_my13 = dedroop_signal(axuv_my13, t, 0.022)
    #     axuv_my22 = dedroop_signal(axuv_my22, t, 0.022)
    #     axuv_no_filter = dedroop_signal(axuv_no_filter, t, 0.022)
        
    #     #print(axuv_my06)
    #     axuv_my06_sig=F_Mylar_6um*np.asarray(axuv_my06)
    #     axuv_my06_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_my06_sig)), filter_points, filter_order, mode='nearest')
    #     axuv_my13_sig=F_Mylar_13um*np.asarray(axuv_my13)
    #     axuv_my13_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_my13_sig)), filter_points, filter_order, mode='nearest')
    #     axuv_my22_sig=F_Mylar_22um*np.asarray(axuv_my22)
    #     axuv_my22_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_my22_sig)), filter_points, filter_order, mode='nearest')
    #     axuv_no_filter_sig=F_Mylar_NO*np.asarray(axuv_no_filter)
    #     axuv_no_filter_sig_filter=savgol_filter(np.asarray(np.asarray(axuv_no_filter_sig)), filter_points, filter_order, mode='nearest')
        
        
    plasma_current_sig = plasma_current
    plasma_current_sig_filter = savgol_filter(np.asarray(np.asarray(plasma_current_sig)), filter_points, filter_order, mode='nearest')
    plasma_current_time = plasma_current.x_axis()
    t = axuv_my06.x_axis()
    
    
    # get the time from one of the raw signal channel
    
    # dt = axuv_my06['dim_meta'][0]['delta']
    # t0 = axuv_my06['dim_meta'][0]['offset']
    # num_points = axuv_my06['dim_meta'][0]['length']
    
    # t = np.linspace(t0, t0+dt*(num_points-1), num_points)


#-----------------------------plot raw axuv signal-------

    plt.figure()
    if shot_n>18399 and shot_n<19004:
        plt.plot(t*1000,axuv_my06_sig,'b',linewidth=2,label ='axuv_Mylar_5.5um')
        plt.plot(t*1000,axuv_my13_sig,'r',linewidth=2,label ='axuv_Mylar_11um')
        
        plt.plot(t*1000,axuv_Al03_sig,'k',linewidth=2,label ='axuv_Al_3um')
        plt.plot(t*1000,axuv_no_filter_sig,'g',linewidth=2,label ='axuv_no filter')
        plt.ylim([-0.1, 0.8])
        plt.xlim([-1, 50])
        plt.xlabel('time(ms)',fontsize= 14 )
        plt.ylabel(' Volt(V)',fontsize= 14 )
        plt.xticks(fontsize= 14 )
        plt.yticks(fontsize= 14 )
        plt.title('shot='+str(shot_n),fontsize=14)
        plt.legend(fontsize=14)
        
    elif shot_n>=19004 and shot_n<19200:
        plt.plot(t*1000,axuv_my06_sig_filter*1e4,'b',linewidth=2,label ='axuv_Mylar_5.5um')
        plt.plot(t*1000,axuv_my13_sig_filter*1e4,'r',linewidth=2,label ='axuv_Mylar_11um')
        plt.plot(t*1000,axuv_my22_sig_filter*1e4,'k',linewidth=2,label ='axuv_my_22um')
        plt.plot(t*1000,axuv_no_filter_sig_filter*1e1,'g',linewidth=2,label ='axuv_no filter x0.001')
        # plt.plot(t*1000,axuv_my06_sig,'b',linewidth=2,label ='axuv_Mylar_5.5um')
        # plt.plot(t*1000,axuv_my13_sig,'r',linewidth=2,label ='axuv_Mylar_11um')
        
        # plt.plot(t*1000,axuv_my22_sig,'k',linewidth=2,label ='axuv_my_22um')
        # plt.plot(t*1000,axuv_no_filter_sig,'g',linewidth=2,label ='axuv_no filter')
        plt.ylim([-0.1, 0.8])
        plt.xlim([-1, 50])
        plt.xlabel('time(ms)',fontsize= 14 )
        plt.ylabel(' current (nA)',fontsize= 14 )
        plt.xticks(fontsize= 14 )
        plt.yticks(fontsize= 14 )
        plt.title('shot='+str(shot_n),fontsize=14)
        plt.legend(fontsize=14)
        

    elif shot_n>=19200 and shot_n < 20470:
        plt.plot(t*1000,axuv_my06_sig_filter*1e4,'b',linewidth=2,label ='axuv_Mylar_5.5um')
        plt.plot(t*1000,axuv_my13_sig_filter*1e4,'r',linewidth=2,label ='axuv_Mylar_11um')
        plt.plot(t*1000,axuv_my22_sig_filter*1e4,'k',linewidth=2,label ='axuv_my_22um')
        plt.plot(t*1000,axuv_no_filter_sig_filter*1e1,'g',linewidth=2,label ='axuv_no filter x0.001')
        #plt.plot(t*1000,axuv_my06_sig*1e4,'b',linewidth=2,label ='axuv_Mylar_5.5um')
        #plt.plot(t*1000,axuv_my13_sig*1e4,'r',linewidth=2,label ='axuv_Mylar_11um')
        
        #plt.plot(t*1000,axuv_my22_sig*1e4,'k',linewidth=2,label ='axuv_my_22um')
        #plt.plot(t*1000,axuv_no_filter_sig*1e1,'g',linewidth=2,label ='axuv_no filterx0.001')
        #plt.ylim([-0.1, 0.8])
        plt.xlim([-1, 20])
        plt.xlabel('time(ms)',fontsize= 14 )
        plt.ylabel(' current (nA)',fontsize= 14 )
        plt.xticks(fontsize= 14 )
        plt.yticks(fontsize= 14 )
        plt.title('shot='+str(shot_n),fontsize=14)
        plt.legend(fontsize=14)
        
    elif shot_n>= 20470:
        plt.plot(t*1000,axuv_my06_sig_filter*1e4,'b',linewidth=2,label ='axuv_Mylar_6.2um')
        plt.plot(t*1000,axuv_my13_sig_filter*1e4,'r',linewidth=2,label ='axuv_Mylar_12.4um')
        plt.plot(t*1000,axuv_my22_sig_filter*1e4,'k',linewidth=2,label ='axuv_my_21um')
        plt.plot(t*1000,axuv_no_filter_sig_filter*1e2,'g',linewidth=2,label ='axuv_no filter x0.01')
        #plt.plot(t*1000,axuv_my06_sig*1e4,'b',linewidth=2,label ='axuv_Mylar_5.5um')
        #plt.plot(t*1000,axuv_my13_sig*1e4,'r',linewidth=2,label ='axuv_Mylar_11um')
         
        #plt.plot(t*1000,axuv_my22_sig*1e4,'k',linewidth=2,label ='axuv_my_22um')
        #plt.plot(t*1000,axuv_no_filter_sig*1e1,'g',linewidth=2,label ='axuv_no filterx0.001')
        plt.ylim([-1000, 2500])
        plt.xlim([-1, 30])
        plt.xlabel('time(ms)',fontsize= 14 )
        plt.ylabel(' current (nA)',fontsize= 14 )
        plt.xticks(fontsize= 14 )
        plt.yticks(fontsize= 14 )
        plt.title('shot='+str(shot_n),fontsize=14)
        plt.legend(fontsize=14)
        
        
        plt.figure()
        #spectrum, ffttime, Zxx = signal.stft(axuv_my06_sig_filter, 1/8e-7,window = 'hann',nperseg=1024, noverlap=512, nfft=4096, detrend='linear')
        
        spectrum, ffttime, Zxx = signal.spectrogram(axuv_my06_sig_filter, 1/8e-7,window = 'hann', nperseg=1024, noverlap=512,nfft=4096, detrend='linear',  return_onesided=True,mode='magnitude',)
        plt.pcolormesh(ffttime, spectrum, np.abs(Zxx)*8,  cmap = 'plasma',vmin=np.abs(Zxx).min(), vmax=np.abs(Zxx).max(), shading='nearest')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.ylim(0, 30000)
        plt.xlim(0, 0.03)
        plt.show()
    # elif shot_n>= 20939:
    #     plt.plot(t*1000,axuv_my06_sig_filter*1e4,'b',linewidth=2,label ='axuv_Mylar_6.2um')
    #     plt.plot(t*1000,axuv_my13_sig_filter*1e4,'r',linewidth=2,label ='axuv_Mylar_12.4um')
    #     plt.plot(t*1000,axuv_my22_sig_filter*1e4,'k',linewidth=2,label ='axuv_my_21um')
    #     plt.plot(t*1000,axuv_no_filter_sig_filter*1e2,'g',linewidth=2,label ='axuv_no filter x0.01')
    #     #plt.plot(t*1000,axuv_my06_sig*1e4,'b',linewidth=2,label ='axuv_Mylar_5.5um')
    #     #plt.plot(t*1000,axuv_my13_sig*1e4,'r',linewidth=2,label ='axuv_Mylar_11um')
         
    #     #plt.plot(t*1000,axuv_my22_sig*1e4,'k',linewidth=2,label ='axuv_my_22um')
    #     #plt.plot(t*1000,axuv_no_filter_sig*1e1,'g',linewidth=2,label ='axuv_no filterx0.001')
    #     plt.ylim([-100, 3500])
    #     plt.xlim([-1, 40])
    #     plt.xlabel('time(ms)',fontsize= 14 )
    #     plt.ylabel(' current (nA)',fontsize= 14 )
    #     plt.xticks(fontsize= 14 )
    #     plt.yticks(fontsize= 14 )
    #     plt.title('shot='+str(shot_n),fontsize=14)
    #     plt.legend(fontsize=14)
        
#-------------------------Te with ratio--------------
    if shot_n>18399 and shot_n<19004:
        # get the ratio of the two mylar filters channel(have only two channel before shot 19004)
        
        Ratio=axuv_my13_sig_filter/axuv_my06_sig_filter
        #11um mylar and 5.5um
        #tebeta=1.2
        Te_forward_model_betaTe1_1=54.6469+995.687*Ratio**1-4424.05*Ratio**2+13990.3*Ratio**3-18590.7*Ratio**4+10297.8*Ratio**5
        #tebeta=5
        Te_forward_model_betaTe1_2=61.4697+697.168*Ratio**1-2760.64*Ratio**2+9265.86*Ratio**3-12565*Ratio**4+7147.02*Ratio**5
        
        #uncertainty
        #tebeta=1.2
        Te_forward_model_betaTe1_p_1=82.064+860.633*Ratio**1-3643.18*Ratio**2+13498.2*Ratio**3-19782.4*Ratio**4+11860.2*Ratio**5

        #tebeta=5
        Te_forward_model_betaTe1_p_2=82.4974+588.201*Ratio**1-1848.99*Ratio**2+7436.32*Ratio**3-10884*Ratio**4+6713.02*Ratio**5

        #tebeta=1.2
        Te_forward_model_betaTe1_m_1=22.0015+1275.88*Ratio**1-5657.26*Ratio**2+16381.9*Ratio**3-20888*Ratio**4+11063.1*Ratio**5
        #tebeta=5
        Te_forward_model_betaTe1_m_2=38.3108+883.562*Ratio**1-3599.76*Ratio**2+10930.2*Ratio**3-14254.6*Ratio**4+7763.57*Ratio**5


        # use the average signal of  Te_forward_model_betaTe1 and Te_forward_model_betaTe2 as the Te
        Te1_mean=(Te_forward_model_betaTe1_1+Te_forward_model_betaTe1_2)/2.0
        # uncertainty analysis
        Te1_p_uncer_filter_1=Te_forward_model_betaTe1_p_1-Te_forward_model_betaTe1_1
        Te1_m_uncer_filter_1=Te_forward_model_betaTe1_1-Te_forward_model_betaTe1_m_1
        Te1_p_uncer_filter_2=Te_forward_model_betaTe1_p_2-Te_forward_model_betaTe1_2
        Te1_m_uncer_filter_2=Te_forward_model_betaTe1_2-Te_forward_model_betaTe1_m_2
        Te1_std_profile=np.array([Te_forward_model_betaTe1_1,Te_forward_model_betaTe1_2]).std(axis=0)
        Te1_p_uncer_filter = (Te1_p_uncer_filter_1+Te1_p_uncer_filter_2)/2
        Te1_m_uncer_filter = (Te1_m_uncer_filter_1+Te1_m_uncer_filter_2)/2
        Te1_p_uncertainty=np.sqrt(Te1_p_uncer_filter**2+Te1_std_profile**2)
        Te1_m_uncertainty=np.sqrt(Te1_m_uncer_filter**2+Te1_std_profile**2)
        
        
        fig, ax1 = plt.subplots(figsize=(12, 9))
        ax2 = ax1.twinx()
        ax1.plot(t*1000,Te1_mean,'r',linewidth=2,label ='AXUV_Te')

        #ax1.plot(t*1000,Te1_mean+Te1_p_uncertainty,'k',linewidth=2,label ='AXUV_Te plus uncertainty')
        #ax1.plot(t*1000,Te1_mean-Te1_m_uncertainty,'b',linewidth=2,label ='AXUV_Te minus uncertainty ')
        ax1.fill_between(t*1000,Te1_mean-Te1_m_uncertainty,Te1_mean+Te1_p_uncertainty,facecolor='green',alpha=0.6) 
        
        #ax2.plot(plasma_current_time*1000,np.array(plasma_current_sig)/1000,'m',linewidth=2,label ='plasma current')
        
        ax1.set_ylim([0,500])
        ax1.set_xlim([0, 18])
        ax2.set_ylim([0,500])
        ax1.set_xlabel('time(ms)',fontsize= 14 )
        ax1.set_ylabel('Te(eV) ',fontsize= 14 )
        ax2.set_ylabel('plasma current(kA) ',fontsize= 14 )
        plt.xticks(fontsize= 14 )
        plt.yticks(fontsize= 14 )
        plt.title('shot='+str(shot_n),fontsize=14)
        ax1.legend(prop={'size': 14},loc='upper left')
        ax2.legend(prop={'size': 14},loc='upper right')
        save_data={'shot number':shot_n,
                   'Mylar11&5.5_axuv_Te (eV)':Te1_mean,
                   'AXUV_Te plus uncertainty':Te1_p_uncertainty,
                   'AXUV_Te minus uncertainty':Te1_m_uncertainty,
                   'axuv_Te_time (ms)':t*1000}

    elif shot_n>=19004 and shot_n < 20470:
        # get the ratio of the two channel from the 3 filters system (have 3 ratios as we have three filters after 19004)
        
        # #11um mylar and 5.5um
        Ratio1=axuv_my13_sig_filter/axuv_my06_sig_filter
        # #22um mylar and 5.5um
        Ratio2=axuv_my22_sig_filter/axuv_my06_sig_filter
        # #22um mylar and 11um
        Ratio3=axuv_my22_sig_filter/axuv_my13_sig_filter
        
        # #11um mylar and 5.5um
        # Ratio1=axuv_my13_sig/axuv_my06_sig
        # #22um mylar and 5.5um
        # Ratio2=axuv_my22_sig/axuv_my06_sig
        # #22um mylar and 11um
        # Ratio3=axuv_my22_sig/axuv_my13_sig
        
        # fitcoe1_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty - shots-19004-20470\mean values\fit_coe_tan.npy')
        # fitcoe2_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty - shots-19004-20470\mean values\fit_coe2_tan.npy')
        # fitcoe3_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty - shots-19004-20470\mean values\fit_coe3_tan.npy')
        
        # fitcoe1_p_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty - shots-19004-20470\Te1&Te2 error-plus\fit_coe_tan.npy')
        # fitcoe2_p_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty - shots-19004-20470\Te1&Te2 error-plus\fit_coe2_tan.npy')
         
        # fitcoe1_m_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty - shots-19004-20470\Te1&T2 error-minus\fit_coe_tan.npy')
        # fitcoe2_m_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty - shots-19004-20470\Te1&T2 error-minus\fit_coe2_tan.npy')
        
        
        # fitcoe3_p_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty - shots-19004-20470\Te3 error-plus\fit_coe3_tan.npy')

        # fitcoe3_m_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty - shots-19004-20470\Te3 error-minus\fit_coe3_tan.npy')
         
        fitcoe1_load = np.load(r'mean value and uncertainty - shots-19004-20470\mean values\fit_coe_tan.npy')
        fitcoe2_load = np.load(r'mean value and uncertainty - shots-19004-20470\mean values\fit_coe2_tan.npy')
        fitcoe3_load = np.load(r'mean value and uncertainty - shots-19004-20470\mean values\fit_coe3_tan.npy')
        
        fitcoe1_p_load = np.load(r'mean value and uncertainty - shots-19004-20470\Te1&Te2 error-plus\fit_coe_tan.npy')
        fitcoe2_p_load = np.load(r'mean value and uncertainty - shots-19004-20470\Te1&Te2 error-plus\fit_coe2_tan.npy')
         
        fitcoe1_m_load = np.load(r'mean value and uncertainty - shots-19004-20470\Te1&T2 error-minus\fit_coe_tan.npy')
        fitcoe2_m_load = np.load(r'mean value and uncertainty - shots-19004-20470\Te1&T2 error-minus\fit_coe2_tan.npy')
        
        
        fitcoe3_p_load = np.load(r'mean value and uncertainty - shots-19004-20470\Te3 error-plus\fit_coe3_tan.npy')

        fitcoe3_m_load = np.load(r'mean value and uncertainty - shots-19004-20470\Te3 error-minus\fit_coe3_tan.npy')
         
        
        
        # AXUV_Te from mylar filter 11/5.5
        
        #tebeta=0.2
        Te_forward_model_betaTe1_1=fitcoe1_load[1][0]+fitcoe1_load[1][1]*Ratio1**1+fitcoe1_load[1][2]*Ratio1**2+fitcoe1_load[1][3]*Ratio1**3+fitcoe1_load[1][4]*Ratio1**4+fitcoe1_load[1][5]*Ratio1**5
        #tebeta=2
        Te_forward_model_betaTe1_2=fitcoe1_load[5][0]+fitcoe1_load[5][1]*Ratio1**1+fitcoe1_load[5][2]*Ratio1**2+fitcoe1_load[5][3]*Ratio1**3+fitcoe1_load[5][4]*Ratio1**4+fitcoe1_load[5][5]*Ratio1**5
       
        Te_forward_model_1=(Te_forward_model_betaTe1_1+Te_forward_model_betaTe1_2)/2.0
      
        # #uncertainty
        # #tebeta=0.2
        
        
        Te_forward_model_betaTe1_p_1=fitcoe1_p_load[1][0]+fitcoe1_p_load[1][1]*Ratio1**1+fitcoe1_p_load[1][2]*Ratio1**2+fitcoe1_p_load[1][3]*Ratio1**3+fitcoe1_p_load[1][4]*Ratio1**4+fitcoe1_p_load[1][5]*Ratio1**5

        # #tebeta=2
        Te_forward_model_betaTe1_p_2=fitcoe1_p_load[5][0]+fitcoe1_p_load[5][1]*Ratio1**1+fitcoe1_p_load[5][2]*Ratio1**2+fitcoe1_p_load[5][3]*Ratio1**3+fitcoe1_p_load[5][4]*Ratio1**4+fitcoe1_p_load[5][5]*Ratio1**5

        # #tebeta=0.2
        Te_forward_model_betaTe1_m_1=fitcoe1_m_load[1][0]+fitcoe1_m_load[1][1]*Ratio1**1+fitcoe1_m_load[1][2]*Ratio1**2+fitcoe1_m_load[1][3]*Ratio1**3+fitcoe1_m_load[1][4]*Ratio1**4+fitcoe1_m_load[1][5]*Ratio1**5

        # #tebeta=2
        Te_forward_model_betaTe1_m_2=fitcoe1_m_load[5][0]+fitcoe1_m_load[5][1]*Ratio1**1+fitcoe1_m_load[5][2]*Ratio1**2+fitcoe1_m_load[5][3]*Ratio1**3+fitcoe1_m_load[5][4]*Ratio1**4+fitcoe1_m_load[5][5]*Ratio1**5


        # # use the average signal of  Te_forward_model_betaTe1 and Te_forward_model_betaTe2 as the Te
        Te1_mean=(Te_forward_model_betaTe1_1+Te_forward_model_betaTe1_2)/2.0
        # # uncertainty analysis
        Te1_p_uncer_filter_1=Te_forward_model_betaTe1_p_1-Te_forward_model_betaTe1_1
        Te1_m_uncer_filter_1=Te_forward_model_betaTe1_1-Te_forward_model_betaTe1_m_1
        Te1_p_uncer_filter_2=Te_forward_model_betaTe1_p_2-Te_forward_model_betaTe1_2
        Te1_m_uncer_filter_2=Te_forward_model_betaTe1_2-Te_forward_model_betaTe1_m_2
        Te1_std_profile=np.array([Te_forward_model_betaTe1_1,Te_forward_model_betaTe1_2]).std(axis=0)
        Te1_p_uncer_filter = (Te1_p_uncer_filter_1+Te1_p_uncer_filter_2)/2
        Te1_m_uncer_filter = (Te1_m_uncer_filter_1+Te1_m_uncer_filter_2)/2
        Te1_p_uncertainty=np.sqrt(Te1_p_uncer_filter**2+Te1_std_profile**2)
        Te1_m_uncertainty=np.sqrt(Te1_m_uncer_filter**2+Te1_std_profile**2)
        
        
        
        # AXUV_Te from mylar filter 22/5.5
        
        #tebeta=0.2
        Te_forward_model_betaTe2_1=fitcoe2_load[1][0]+fitcoe2_load[1][1]*Ratio2**1+fitcoe2_load[1][2]*Ratio2**2+fitcoe2_load[1][3]*Ratio2**3+fitcoe2_load[1][4]*Ratio2**4+fitcoe2_load[1][5]*Ratio2**5
        #tebeta=2
        Te_forward_model_betaTe2_2=fitcoe2_load[5][0]+fitcoe2_load[5][1]*Ratio2**1+fitcoe2_load[5][2]*Ratio2**2+fitcoe2_load[5][3]*Ratio2**3+fitcoe2_load[5][4]*Ratio2**4+fitcoe2_load[5][5]*Ratio2**5
        Te_forward_model_2=(Te_forward_model_betaTe2_1+Te_forward_model_betaTe2_2)/2.0
        
        
        # #uncertainty
        # #tebeta=0.2
        Te_forward_model_betaTe2_p_1=fitcoe2_p_load[1][0]+fitcoe2_p_load[1][1]*Ratio2**1+fitcoe2_p_load[1][2]*Ratio2**2+fitcoe2_p_load[1][3]*Ratio2**3+fitcoe2_p_load[1][4]*Ratio2**4+fitcoe2_p_load[1][5]*Ratio2**5
        # #tebeta=2
        Te_forward_model_betaTe2_p_2=fitcoe2_p_load[5][0]+fitcoe2_p_load[5][1]*Ratio2**1+fitcoe2_p_load[5][2]*Ratio2**2+fitcoe2_p_load[5][3]*Ratio2**3+fitcoe2_p_load[5][4]*Ratio2**4+fitcoe2_p_load[5][5]*Ratio2**5

        # #tebeta=0.2
        Te_forward_model_betaTe2_m_1=fitcoe2_m_load[1][0]+fitcoe2_m_load[1][1]*Ratio2**1+fitcoe2_m_load[1][2]*Ratio2**2+fitcoe2_m_load[1][3]*Ratio2**3+fitcoe2_m_load[1][4]*Ratio2**4+fitcoe2_m_load[1][5]*Ratio2**5
        # #tebeta=2
        Te_forward_model_betaTe2_m_2=fitcoe2_m_load[5][0]+fitcoe2_m_load[5][1]*Ratio2**1+fitcoe2_m_load[5][2]*Ratio2**2+fitcoe2_m_load[5][3]*Ratio2**3+fitcoe2_m_load[5][4]*Ratio2**4+fitcoe2_m_load[5][5]*Ratio2**5


        Te2_p_uncer_filter_1=Te_forward_model_betaTe2_p_1-Te_forward_model_betaTe2_1
        Te2_m_uncer_filter_1=Te_forward_model_betaTe2_1-Te_forward_model_betaTe2_m_1
        Te2_p_uncer_filter_2=Te_forward_model_betaTe2_p_2-Te_forward_model_betaTe2_2
        Te2_m_uncer_filter_2=Te_forward_model_betaTe2_2-Te_forward_model_betaTe2_m_2

        Te2_std_profile=np.array([Te_forward_model_betaTe2_1,Te_forward_model_betaTe2_2]).std(axis=0)

        Te2_p_uncer_filter = (Te2_p_uncer_filter_1+Te2_p_uncer_filter_2)/2
        Te2_m_uncer_filter = (Te2_m_uncer_filter_1+Te2_m_uncer_filter_2)/2


        Te2_p_uncertainty=np.sqrt(Te2_p_uncer_filter**2+Te2_std_profile**2)
        Te2_m_uncertainty=np.sqrt(Te2_m_uncer_filter**2+Te2_std_profile**2)

        
        
        # AXUV_Te from mylar filter 22/11
        Te_forward_model_betaTe3_1=fitcoe3_load[1][0]+fitcoe3_load[1][1]*Ratio3**1+fitcoe3_load[1][2]*Ratio3**2+fitcoe3_load[1][3]*Ratio3**3+fitcoe3_load[1][4]*Ratio3**4+fitcoe3_load[1][5]*Ratio3**5
        #tebeta=2
        Te_forward_model_betaTe3_2=fitcoe3_load[5][0]+fitcoe3_load[5][1]*Ratio3**1+fitcoe3_load[5][2]*Ratio3**2+fitcoe3_load[5][3]*Ratio3**3+fitcoe3_load[5][4]*Ratio3**4+fitcoe3_load[5][5]*Ratio3**5
        Te_forward_model_3=(Te_forward_model_betaTe3_1+Te_forward_model_betaTe3_2)/2.0
        
        # #uncertainty
        # #tebeta=0.2
        Te_forward_model_betaTe3_p_1=fitcoe3_p_load[1][0]+fitcoe3_p_load[1][1]*Ratio3**1+fitcoe3_p_load[1][2]*Ratio3**2+fitcoe3_p_load[1][3]*Ratio3**3+fitcoe3_p_load[1][4]*Ratio3**4+fitcoe3_p_load[1][5]*Ratio3**5
        # #tebeta=2
        Te_forward_model_betaTe3_p_2=fitcoe3_p_load[5][0]+fitcoe3_p_load[5][1]*Ratio3**1+fitcoe3_p_load[5][2]*Ratio3**2+fitcoe3_p_load[5][3]*Ratio3**3+fitcoe3_p_load[5][4]*Ratio3**4+fitcoe3_p_load[5][5]*Ratio3**5

        # #tebeta=0.2
        Te_forward_model_betaTe3_m_1=fitcoe3_m_load[1][0]+fitcoe3_m_load[1][1]*Ratio3**1+fitcoe3_m_load[1][2]*Ratio3**2+fitcoe3_m_load[1][3]*Ratio3**3+fitcoe3_m_load[1][4]*Ratio3**4+fitcoe3_m_load[1][5]*Ratio3**5
        
        # #tebeta=2
        Te_forward_model_betaTe3_m_2=fitcoe3_m_load[5][0]+fitcoe3_m_load[5][1]*Ratio3**1+fitcoe3_m_load[5][2]*Ratio3**2+fitcoe3_m_load[5][3]*Ratio3**3+fitcoe3_m_load[5][4]*Ratio3**4+fitcoe3_m_load[5][5]*Ratio3**5

        Te3_p_uncer_filter_1=Te_forward_model_betaTe3_p_1-Te_forward_model_betaTe3_1
        Te3_m_uncer_filter_1=Te_forward_model_betaTe3_1-Te_forward_model_betaTe3_m_1
        Te3_p_uncer_filter_2=Te_forward_model_betaTe3_p_2-Te_forward_model_betaTe3_2
        Te3_m_uncer_filter_2=Te_forward_model_betaTe3_2-Te_forward_model_betaTe3_m_2

        Te3_std_profile=np.array([Te_forward_model_betaTe3_1,Te_forward_model_betaTe3_2]).std(axis=0)

        Te3_p_uncer_filter = (Te3_p_uncer_filter_1+Te3_p_uncer_filter_2)/2
        Te3_m_uncer_filter = (Te3_m_uncer_filter_1+Te3_m_uncer_filter_2)/2


        Te3_p_uncertainty=np.sqrt(Te3_p_uncer_filter**2+Te3_std_profile**2)
        Te3_m_uncertainty=np.sqrt(Te3_m_uncer_filter**2+Te3_std_profile**2)
       
        plt.figure()
       
        plt.plot(t*1000,Te_forward_model_1,'r',linewidth=2,label ='AXUV_Te_mylar11&5.5')
        #plt.plot(t*1000,Te_forward_model_1+Te1_p_uncertainty,'k',linewidth=2,label ='AXUV_Te plus uncertainty')
        #plt.plot(t*1000,Te_forward_model_1-Te1_m_uncertainty,'b',linewidth=2,label ='AXUV_Te minus uncertainty ')
        plt.fill_between(t*1000,Te_forward_model_1-Te1_m_uncertainty,Te_forward_model_1+Te1_p_uncertainty,facecolor='green',alpha=0.6) 
        
        plt.plot(t*1000,Te_forward_model_2,'k',linewidth=2,label ='AXUV_Te_mylar22&5.5')
        #plt.plot(t*1000,Te_forward_model_2+Te2_p_uncertainty,'k',linewidth=2,label ='AXUV_Te plus uncertainty')
        #plt.plot(t*1000,Te_forward_model_2-Te2_m_uncertainty,'b',linewidth=2,label ='AXUV_Te minus uncertainty ')
        plt.fill_between(t*1000,Te_forward_model_2-Te2_m_uncertainty,Te_forward_model_2+Te2_p_uncertainty,facecolor='green',alpha=0.6) 
        
        plt.plot(t*1000,Te_forward_model_3,'b',linewidth=2,label ='AXUV_Te_mylar22&11')
        #plt.plot(t*1000,Te_forward_model_3+Te3_p_uncertainty,'k',linewidth=2,label ='AXUV_Te plus uncertainty')
        #plt.plot(t*1000,Te_forward_model_3-Te3_m_uncertainty,'b',linewidth=2,label ='AXUV_Te minus uncertainty ')
        plt.fill_between(t*1000,Te_forward_model_3-Te3_m_uncertainty,Te_forward_model_3+Te3_p_uncertainty,facecolor='green',alpha=0.6) 
        
        plt.ylim([0,400])
        plt.xlim([0, 18])
   
        plt.xlabel('time(ms)',fontsize= 14 )
        plt.ylabel('Te(eV) ',fontsize= 14 )
        plt.xticks(fontsize= 14 )
        plt.yticks(fontsize= 14 )
        plt.title('shot='+str(shot_n),fontsize=14)
        plt.legend(fontsize=14)
        
        
        # plt.figure()
        # plt.plot(plasma_current_time*1000,np.array(plasma_current_sig)/1000,'m',linewidth=2,label ='plasma current')
        # plt.xlabel('time(ms)',fontsize= 14 )
        # plt.ylabel('plasma current (kA) ',fontsize= 14 )
        # plt.xticks(fontsize= 14 )
        # plt.yticks(fontsize= 14 )
        # plt.title('shot='+str(shot_n),fontsize=14)
        # plt.legend(fontsize=14)
        
        # plt.ylim([0,600])
        # plt.xlim([0, 25])
        
        
        save_data={'shot number':shot_n,
                   
                   'axuv_Te_time (ms)':t*1000,
                    'axuv_mylar6um_current(nA)':axuv_my06_sig_filter*1e4,
                    'axuv_mylar13um_current(nA)':axuv_my13_sig_filter*1e4,
                    'axuv_mylar22um_current(nA)':axuv_my22_sig_filter*1e4,
                    'axuv_no_filter_current(nA)':axuv_no_filter_sig_filter*1e4,
                   'Ratio of 11um to 5.5um':Ratio1,
                   'Ratio of 22um to 5.5um':Ratio2,
                   'Ratio of 22um to 11um':Ratio3,
                   'Mylar11&5.5_axuv_Te (eV)':Te_forward_model_1,
                   'AXUV_Te1 plus uncertainty':Te1_p_uncertainty,
                   'AXUV_Te1 minus uncertainty':Te1_m_uncertainty,
                   'Mylar22&5.5_axuv_Te (eV)':Te_forward_model_2,
                   'AXUV_Te2 plus uncertainty':Te2_p_uncertainty,
                   'AXUV_Te2 minus uncertainty':Te2_m_uncertainty,
                   'Mylar22&11_axuv_Te (eV)':Te_forward_model_3,
                   'AXUV_Te3 plus uncertainty':Te3_p_uncertainty,
                   'AXUV_Te3 minus uncertainty':Te3_m_uncertainty,
                   }

    elif  shot_n >= 20470 and shot_n < 22472:
        # get the ratio of the two channel from the 3 filters system (have 3 ratios as we have three filters after 19004)
        
        # #12.4um mylar and 6.2um
        Ratio1=axuv_my13_sig_filter/axuv_my06_sig_filter
        # #21um mylar and 6.2um
        Ratio2=axuv_my22_sig_filter/axuv_my06_sig_filter
        # #21um mylar and 12.4um
        Ratio3=axuv_my22_sig_filter/axuv_my13_sig_filter
        
        
        #-----new coefficient after fixing the error in the code
        
        
        # fitcoe1_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty\mean values\fit_coe_tan.npy')
        # fitcoe2_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty\mean values\fit_coe2_tan.npy')
        # fitcoe3_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty\mean values\fit_coe3_tan.npy')
        
        # fitcoe1_p_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty\Te1&Te2 error-plus\fit_coe_tan.npy')
        # fitcoe2_p_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty\Te1&Te2 error-plus\fit_coe2_tan.npy')
         
        # fitcoe1_m_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty\Te1&T2 error-minus\fit_coe_tan.npy')
        # fitcoe2_m_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty\Te1&T2 error-minus\fit_coe2_tan.npy')
        
        
        # fitcoe3_p_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty\Te3 error-plus\fit_coe3_tan.npy')

        # fitcoe3_m_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty\Te3 error-minus\fit_coe3_tan.npy')
         
        fitcoe1_load = np.load(r'mean value and uncertainty\mean values\fit_coe_tan.npy')
        fitcoe2_load = np.load(r'mean value and uncertainty\mean values\fit_coe2_tan.npy')
        fitcoe3_load = np.load(r'mean value and uncertainty\mean values\fit_coe3_tan.npy')
        
        fitcoe1_p_load = np.load(r'mean value and uncertainty\Te1&Te2 error-plus\fit_coe_tan.npy')
        fitcoe2_p_load = np.load(r'mean value and uncertainty\Te1&Te2 error-plus\fit_coe2_tan.npy')
         
        fitcoe1_m_load = np.load(r'mean value and uncertainty\Te1&T2 error-minus\fit_coe_tan.npy')
        fitcoe2_m_load = np.load(r'mean value and uncertainty\Te1&T2 error-minus\fit_coe2_tan.npy')
        
        
        fitcoe3_p_load = np.load(r'mean value and uncertainty\Te3 error-plus\fit_coe3_tan.npy')

        fitcoe3_m_load = np.load(r'mean value and uncertainty\Te3 error-minus\fit_coe3_tan.npy')
         
        # AXUV_Te from mylar filter 12.4/6.2
        
        #tebeta=0.2
        Te_forward_model_betaTe1_1=fitcoe1_load[1][0]+fitcoe1_load[1][1]*Ratio1**1+fitcoe1_load[1][2]*Ratio1**2+fitcoe1_load[1][3]*Ratio1**3+fitcoe1_load[1][4]*Ratio1**4+fitcoe1_load[1][5]*Ratio1**5
        #tebeta=2
        Te_forward_model_betaTe1_2=fitcoe1_load[5][0]+fitcoe1_load[5][1]*Ratio1**1+fitcoe1_load[5][2]*Ratio1**2+fitcoe1_load[5][3]*Ratio1**3+fitcoe1_load[5][4]*Ratio1**4+fitcoe1_load[5][5]*Ratio1**5
       
        Te_forward_model_1=(Te_forward_model_betaTe1_1+Te_forward_model_betaTe1_2)/2.0
      
        # #uncertainty
        # #tebeta=0.2
        
        
        Te_forward_model_betaTe1_p_1=fitcoe1_p_load[1][0]+fitcoe1_p_load[1][1]*Ratio1**1+fitcoe1_p_load[1][2]*Ratio1**2+fitcoe1_p_load[1][3]*Ratio1**3+fitcoe1_p_load[1][4]*Ratio1**4+fitcoe1_p_load[1][5]*Ratio1**5

        # #tebeta=2
        Te_forward_model_betaTe1_p_2=fitcoe1_p_load[5][0]+fitcoe1_p_load[5][1]*Ratio1**1+fitcoe1_p_load[5][2]*Ratio1**2+fitcoe1_p_load[5][3]*Ratio1**3+fitcoe1_p_load[5][4]*Ratio1**4+fitcoe1_p_load[5][5]*Ratio1**5

        # #tebeta=0.2
        Te_forward_model_betaTe1_m_1=fitcoe1_m_load[1][0]+fitcoe1_m_load[1][1]*Ratio1**1+fitcoe1_m_load[1][2]*Ratio1**2+fitcoe1_m_load[1][3]*Ratio1**3+fitcoe1_m_load[1][4]*Ratio1**4+fitcoe1_m_load[1][5]*Ratio1**5

        # #tebeta=2
        Te_forward_model_betaTe1_m_2=fitcoe1_m_load[5][0]+fitcoe1_m_load[5][1]*Ratio1**1+fitcoe1_m_load[5][2]*Ratio1**2+fitcoe1_m_load[5][3]*Ratio1**3+fitcoe1_m_load[5][4]*Ratio1**4+fitcoe1_m_load[5][5]*Ratio1**5


        # # use the average signal of  Te_forward_model_betaTe1 and Te_forward_model_betaTe2 as the Te
        Te1_mean=(Te_forward_model_betaTe1_1+Te_forward_model_betaTe1_2)/2.0
        # # uncertainty analysis
        Te1_p_uncer_filter_1=Te_forward_model_betaTe1_p_1-Te_forward_model_betaTe1_1
        Te1_m_uncer_filter_1=Te_forward_model_betaTe1_1-Te_forward_model_betaTe1_m_1
        Te1_p_uncer_filter_2=Te_forward_model_betaTe1_p_2-Te_forward_model_betaTe1_2
        Te1_m_uncer_filter_2=Te_forward_model_betaTe1_2-Te_forward_model_betaTe1_m_2
        Te1_std_profile=np.array([Te_forward_model_betaTe1_1,Te_forward_model_betaTe1_2]).std(axis=0)
        Te1_p_uncer_filter = (Te1_p_uncer_filter_1+Te1_p_uncer_filter_2)/2
        Te1_m_uncer_filter = (Te1_m_uncer_filter_1+Te1_m_uncer_filter_2)/2
        Te1_p_uncertainty=np.sqrt(Te1_p_uncer_filter**2+Te1_std_profile**2)
        Te1_m_uncertainty=np.sqrt(Te1_m_uncer_filter**2+Te1_std_profile**2)
        
        
        
        # AXUV_Te from mylar filter 21/6.2
        #tebeta=0.2
        Te_forward_model_betaTe2_1=fitcoe2_load[1][0]+fitcoe2_load[1][1]*Ratio2**1+fitcoe2_load[1][2]*Ratio2**2+fitcoe2_load[1][3]*Ratio2**3+fitcoe2_load[1][4]*Ratio2**4+fitcoe2_load[1][5]*Ratio2**5
        #tebeta=2
        Te_forward_model_betaTe2_2=fitcoe2_load[5][0]+fitcoe2_load[5][1]*Ratio2**1+fitcoe2_load[5][2]*Ratio2**2+fitcoe2_load[5][3]*Ratio2**3+fitcoe2_load[5][4]*Ratio2**4+fitcoe2_load[5][5]*Ratio2**5
        Te_forward_model_2=(Te_forward_model_betaTe2_1+Te_forward_model_betaTe2_2)/2.0
        
        
        # #uncertainty
        # #tebeta=0.2
        Te_forward_model_betaTe2_p_1=fitcoe2_p_load[1][0]+fitcoe2_p_load[1][1]*Ratio2**1+fitcoe2_p_load[1][2]*Ratio2**2+fitcoe2_p_load[1][3]*Ratio2**3+fitcoe2_p_load[1][4]*Ratio2**4+fitcoe2_p_load[1][5]*Ratio2**5
        # #tebeta=2
        Te_forward_model_betaTe2_p_2=fitcoe2_p_load[5][0]+fitcoe2_p_load[5][1]*Ratio2**1+fitcoe2_p_load[5][2]*Ratio2**2+fitcoe2_p_load[5][3]*Ratio2**3+fitcoe2_p_load[5][4]*Ratio2**4+fitcoe2_p_load[5][5]*Ratio2**5

        # #tebeta=0.2
        Te_forward_model_betaTe2_m_1=fitcoe2_m_load[1][0]+fitcoe2_m_load[1][1]*Ratio2**1+fitcoe2_m_load[1][2]*Ratio2**2+fitcoe2_m_load[1][3]*Ratio2**3+fitcoe2_m_load[1][4]*Ratio2**4+fitcoe2_m_load[1][5]*Ratio2**5
        # #tebeta=2
        Te_forward_model_betaTe2_m_2=fitcoe2_m_load[5][0]+fitcoe2_m_load[5][1]*Ratio2**1+fitcoe2_m_load[5][2]*Ratio2**2+fitcoe2_m_load[5][3]*Ratio2**3+fitcoe2_m_load[5][4]*Ratio2**4+fitcoe2_m_load[5][5]*Ratio2**5


        Te2_p_uncer_filter_1=Te_forward_model_betaTe2_p_1-Te_forward_model_betaTe2_1
        Te2_m_uncer_filter_1=Te_forward_model_betaTe2_1-Te_forward_model_betaTe2_m_1
        Te2_p_uncer_filter_2=Te_forward_model_betaTe2_p_2-Te_forward_model_betaTe2_2
        Te2_m_uncer_filter_2=Te_forward_model_betaTe2_2-Te_forward_model_betaTe2_m_2

        Te2_std_profile=np.array([Te_forward_model_betaTe2_1,Te_forward_model_betaTe2_2]).std(axis=0)

        Te2_p_uncer_filter = (Te2_p_uncer_filter_1+Te2_p_uncer_filter_2)/2
        Te2_m_uncer_filter = (Te2_m_uncer_filter_1+Te2_m_uncer_filter_2)/2


        Te2_p_uncertainty=np.sqrt(Te2_p_uncer_filter**2+Te2_std_profile**2)
        Te2_m_uncertainty=np.sqrt(Te2_m_uncer_filter**2+Te2_std_profile**2)
        
        
        # AXUV_Te from mylar filter 21/12.4
        #tebeta=0.2
        Te_forward_model_betaTe3_1=fitcoe3_load[1][0]+fitcoe3_load[1][1]*Ratio3**1+fitcoe3_load[1][2]*Ratio3**2+fitcoe3_load[1][3]*Ratio3**3+fitcoe3_load[1][4]*Ratio3**4+fitcoe3_load[1][5]*Ratio3**5
        #tebeta=2
        Te_forward_model_betaTe3_2=fitcoe3_load[5][0]+fitcoe3_load[5][1]*Ratio3**1+fitcoe3_load[5][2]*Ratio3**2+fitcoe3_load[5][3]*Ratio3**3+fitcoe3_load[5][4]*Ratio3**4+fitcoe3_load[5][5]*Ratio3**5
        Te_forward_model_3=(Te_forward_model_betaTe3_1+Te_forward_model_betaTe3_2)/2.0
        
        # #uncertainty
        # #tebeta=0.2
        Te_forward_model_betaTe3_p_1=fitcoe3_p_load[1][0]+fitcoe3_p_load[1][1]*Ratio3**1+fitcoe3_p_load[1][2]*Ratio3**2+fitcoe3_p_load[1][3]*Ratio3**3+fitcoe3_p_load[1][4]*Ratio3**4+fitcoe3_p_load[1][5]*Ratio3**5
        # #tebeta=2
        Te_forward_model_betaTe3_p_2=fitcoe3_p_load[5][0]+fitcoe3_p_load[5][1]*Ratio3**1+fitcoe3_p_load[5][2]*Ratio3**2+fitcoe3_p_load[5][3]*Ratio3**3+fitcoe3_p_load[5][4]*Ratio3**4+fitcoe3_p_load[5][5]*Ratio3**5

        # #tebeta=0.2
        Te_forward_model_betaTe3_m_1=fitcoe3_m_load[1][0]+fitcoe3_m_load[1][1]*Ratio3**1+fitcoe3_m_load[1][2]*Ratio3**2+fitcoe3_m_load[1][3]*Ratio3**3+fitcoe3_m_load[1][4]*Ratio3**4+fitcoe3_m_load[1][5]*Ratio3**5
        
        # #tebeta=2
        Te_forward_model_betaTe3_m_2=fitcoe3_m_load[5][0]+fitcoe3_m_load[5][1]*Ratio3**1+fitcoe3_m_load[5][2]*Ratio3**2+fitcoe3_m_load[5][3]*Ratio3**3+fitcoe3_m_load[5][4]*Ratio3**4+fitcoe3_m_load[5][5]*Ratio3**5

        Te3_p_uncer_filter_1=Te_forward_model_betaTe3_p_1-Te_forward_model_betaTe3_1
        Te3_m_uncer_filter_1=Te_forward_model_betaTe3_1-Te_forward_model_betaTe3_m_1
        Te3_p_uncer_filter_2=Te_forward_model_betaTe3_p_2-Te_forward_model_betaTe3_2
        Te3_m_uncer_filter_2=Te_forward_model_betaTe3_2-Te_forward_model_betaTe3_m_2

        Te3_std_profile=np.array([Te_forward_model_betaTe3_1,Te_forward_model_betaTe3_2]).std(axis=0)

        Te3_p_uncer_filter = (Te3_p_uncer_filter_1+Te3_p_uncer_filter_2)/2
        Te3_m_uncer_filter = (Te3_m_uncer_filter_1+Te3_m_uncer_filter_2)/2


        Te3_p_uncertainty=np.sqrt(Te3_p_uncer_filter**2+Te3_std_profile**2)
        Te3_m_uncertainty=np.sqrt(Te3_m_uncer_filter**2+Te3_std_profile**2)
       
        plt.figure()
       
        plt.plot(t*1000,1.0*Te_forward_model_1,'r',linewidth=2,label ='AXUV_Te_mylar12.4&6.2um')
        # #plt.plot(t*1000,Te_forward_model_1+Te1_p_uncertainty,'k',linewidth=2,label ='AXUV_Te plus uncertainty')
        # #plt.plot(t*1000,Te_forward_model_1-Te1_m_uncertainty,'b',linewidth=2,label ='AXUV_Te minus uncertainty ')
        plt.fill_between(t*1000,1.0*(Te_forward_model_1-Te1_m_uncertainty),1.0*(Te_forward_model_1+Te1_p_uncertainty),facecolor='red',alpha=0.6) 
        
        plt.plot(t*1000,1.0*Te_forward_model_2,'k',linewidth=2,label ='AXUV_Te_mylar21&6.2um')
        # #plt.plot(t*1000,Te_forward_model_2+Te2_p_uncertainty,'k',linewidth=2,label ='AXUV_Te plus uncertainty')
        # #plt.plot(t*1000,Te_forward_model_2-Te2_m_uncertainty,'b',linewidth=2,label ='AXUV_Te minus uncertainty ')
        plt.fill_between(t*1000,1.0*(Te_forward_model_2-Te2_m_uncertainty),1.0*(Te_forward_model_2+Te2_p_uncertainty),facecolor='black',alpha=0.6) 
        
        plt.plot(t*1000,1.0*Te_forward_model_3,'b',linewidth=2,label ='AXUV_Te_mylar21&12.4um')
        # #plt.plot(t*1000,Te_forward_model_3+Te3_p_uncertainty,'k',linewidth=2,label ='AXUV_Te plus uncertainty')
        # #plt.plot(t*1000,Te_forward_model_3-Te3_m_uncertainty,'b',linewidth=2,label ='AXUV_Te minus uncertainty ')
        plt.fill_between(t*1000,1.0*(Te_forward_model_3-Te3_m_uncertainty),1.0*(Te_forward_model_3+Te3_p_uncertainty),facecolor='blue',alpha=0.6) 
        
        # if shot_n == 20659 :
        #     plt.errorbar(5,420.00723,20.6955,None,'s','r',linewidth=2,label ='TS(600mm) Te@5ms')
        #     plt.errorbar(5,399.96,10.77,None,'s','r',linewidth=2,label ='TS(730mm) Te@5ms')
        #     #plt.errorbar(5,200,7,None,'s','r',linewidth=2,label ='TS(900mm) Te@5ms')
        
                           
        plt.ylim([0,500])
        plt.xlim([0, 30])
   
        plt.xlabel('time(ms)',fontsize= 14 )
        plt.ylabel('Te(eV) ',fontsize= 14 )
        plt.xticks(fontsize= 14 )
        plt.yticks(fontsize= 14 )
        plt.title('shot='+str(shot_n),fontsize=14)
        plt.legend(fontsize=14)
        
        

        
        plt.figure()
        plt.plot(plasma_current_time*1000,np.array(plasma_current_sig)/1000,'m',linewidth=2,label ='plasma current')
        plt.xlabel('time(ms)',fontsize= 14 )
        plt.ylabel('plasma current (kA) ',fontsize= 14 )
        plt.xticks(fontsize= 14 )
        plt.yticks(fontsize= 14 )
        plt.title('shot='+str(shot_n),fontsize=14)
        plt.legend(fontsize=14)
        
        plt.ylim([0,600])
        plt.xlim([0, 35])
        
        
        save_data={'shot number':shot_n,
                   'data' :data['waves'],
                    'axuv_Te_time (ms)':t*1000,
                    'axuv_mylar6um_current(nA)':axuv_my06_sig_filter*1e4,
                    'axuv_mylar13um_current(nA)':axuv_my13_sig_filter*1e4,
                    'axuv_mylar22um_current(nA)':axuv_my22_sig_filter*1e4,
                    'axuv_no_filter_current(nA)':axuv_no_filter_sig_filter*1e4,
                    'Ratio of 12.4um to 6.2um':Ratio1,
                    'Ratio of 21um to 6.2um':Ratio2,
                    'Ratio of 21um to 12.4um':Ratio3,
                    'Mylar12.4&6.2_axuv_Te (eV)':Te_forward_model_1,
                   'AXUV_Te1 plus uncertainty':Te1_p_uncertainty,
                   'AXUV_Te1 minus uncertainty':Te1_m_uncertainty,
                    'Mylar21&6.2_axuv_Te (eV)':Te_forward_model_2,
                   'AXUV_Te2 plus uncertainty':Te2_p_uncertainty,
                   'AXUV_Te2 minus uncertainty':Te2_m_uncertainty,
                    'Mylar21&12.4_axuv_Te (eV)':Te_forward_model_3,
                  'AXUV_Te3 plus uncertainty':Te3_p_uncertainty,
                  'AXUV_Te3 minus uncertainty':Te3_m_uncertainty,
                    }

    elif  shot_n >= 22472:
        # get the ratio of the two channel from the 3 filters system (have 3 ratios as we have three filters after 19004)
        #12.4um axuv channel goes to veto system form shot 22472
        

        # #21um mylar and 6.2um
        Ratio2=axuv_my22_sig_filter/axuv_my06_sig_filter


        

        
        #-----new coefficient after fixing the error in the code
        
        
        # fitcoe2_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty\mean values\fit_coe2_tan.npy')
        
        # fitcoe2_p_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty\Te1&Te2 error-plus\fit_coe2_tan.npy')
         
        # fitcoe2_m_load = np.load(r'C:\Users\xiande.feng\Downloads\PI3 plasma data analysis\PI3 AXUV_Te analysis\20240806\mean value and uncertainty\Te1&T2 error-minus\fit_coe2_tan.npy')
        

        fitcoe2_load = np.load(r'mean value and uncertainty\mean values\fit_coe2_tan.npy')
        
        fitcoe2_p_load = np.load(r'mean value and uncertainty\Te1&Te2 error-plus\fit_coe2_tan.npy')
         
        fitcoe2_m_load = np.load(r'mean value and uncertainty\Te1&T2 error-minus\fit_coe2_tan.npy')
        


        
        
        # AXUV_Te from mylar filter 21/6.2
        #tebeta=0.2
        Te_forward_model_betaTe2_1=fitcoe2_load[1][0]+fitcoe2_load[1][1]*Ratio2**1+fitcoe2_load[1][2]*Ratio2**2+fitcoe2_load[1][3]*Ratio2**3+fitcoe2_load[1][4]*Ratio2**4+fitcoe2_load[1][5]*Ratio2**5
        #tebeta=2
        Te_forward_model_betaTe2_2=fitcoe2_load[5][0]+fitcoe2_load[5][1]*Ratio2**1+fitcoe2_load[5][2]*Ratio2**2+fitcoe2_load[5][3]*Ratio2**3+fitcoe2_load[5][4]*Ratio2**4+fitcoe2_load[5][5]*Ratio2**5
        Te_forward_model_2=(Te_forward_model_betaTe2_1+Te_forward_model_betaTe2_2)/2.0
        
        
        # #uncertainty
        # #tebeta=0.2
        Te_forward_model_betaTe2_p_1=fitcoe2_p_load[1][0]+fitcoe2_p_load[1][1]*Ratio2**1+fitcoe2_p_load[1][2]*Ratio2**2+fitcoe2_p_load[1][3]*Ratio2**3+fitcoe2_p_load[1][4]*Ratio2**4+fitcoe2_p_load[1][5]*Ratio2**5
        # #tebeta=2
        Te_forward_model_betaTe2_p_2=fitcoe2_p_load[5][0]+fitcoe2_p_load[5][1]*Ratio2**1+fitcoe2_p_load[5][2]*Ratio2**2+fitcoe2_p_load[5][3]*Ratio2**3+fitcoe2_p_load[5][4]*Ratio2**4+fitcoe2_p_load[5][5]*Ratio2**5

        # #tebeta=0.2
        Te_forward_model_betaTe2_m_1=fitcoe2_m_load[1][0]+fitcoe2_m_load[1][1]*Ratio2**1+fitcoe2_m_load[1][2]*Ratio2**2+fitcoe2_m_load[1][3]*Ratio2**3+fitcoe2_m_load[1][4]*Ratio2**4+fitcoe2_m_load[1][5]*Ratio2**5
        # #tebeta=2
        Te_forward_model_betaTe2_m_2=fitcoe2_m_load[5][0]+fitcoe2_m_load[5][1]*Ratio2**1+fitcoe2_m_load[5][2]*Ratio2**2+fitcoe2_m_load[5][3]*Ratio2**3+fitcoe2_m_load[5][4]*Ratio2**4+fitcoe2_m_load[5][5]*Ratio2**5


        Te2_p_uncer_filter_1=Te_forward_model_betaTe2_p_1-Te_forward_model_betaTe2_1
        Te2_m_uncer_filter_1=Te_forward_model_betaTe2_1-Te_forward_model_betaTe2_m_1
        Te2_p_uncer_filter_2=Te_forward_model_betaTe2_p_2-Te_forward_model_betaTe2_2
        Te2_m_uncer_filter_2=Te_forward_model_betaTe2_2-Te_forward_model_betaTe2_m_2

        Te2_std_profile=np.array([Te_forward_model_betaTe2_1,Te_forward_model_betaTe2_2]).std(axis=0)

        Te2_p_uncer_filter = (Te2_p_uncer_filter_1+Te2_p_uncer_filter_2)/2
        Te2_m_uncer_filter = (Te2_m_uncer_filter_1+Te2_m_uncer_filter_2)/2


        Te2_p_uncertainty=np.sqrt(Te2_p_uncer_filter**2+Te2_std_profile**2)
        Te2_m_uncertainty=np.sqrt(Te2_m_uncer_filter**2+Te2_std_profile**2)
        

       
        plt.figure()
       

        plt.plot(t*1000,1.0*Te_forward_model_2,'k',linewidth=2,label ='AXUV_Te_mylar21&6.2um')
        # #plt.plot(t*1000,Te_forward_model_2+Te2_p_uncertainty,'k',linewidth=2,label ='AXUV_Te plus uncertainty')
        # #plt.plot(t*1000,Te_forward_model_2-Te2_m_uncertainty,'b',linewidth=2,label ='AXUV_Te minus uncertainty ')
        plt.fill_between(t*1000,1.0*(Te_forward_model_2-Te2_m_uncertainty),1.0*(Te_forward_model_2+Te2_p_uncertainty),facecolor='black',alpha=0.6) 
        

                           
        plt.ylim([0,500])
        plt.xlim([0, 30])
   
        plt.xlabel('time(ms)',fontsize= 14 )
        plt.ylabel('Te(eV) ',fontsize= 14 )
        plt.xticks(fontsize= 14 )
        plt.yticks(fontsize= 14 )
        plt.title('shot='+str(shot_n),fontsize=14)
        plt.legend(fontsize=14)
        
        

        
        plt.figure()
        plt.plot(plasma_current_time*1000,np.array(plasma_current_sig)/1000,'m',linewidth=2,label ='plasma current')
        plt.xlabel('time(ms)',fontsize= 14 )
        plt.ylabel('plasma current (kA) ',fontsize= 14 )
        plt.xticks(fontsize= 14 )
        plt.yticks(fontsize= 14 )
        plt.title('shot='+str(shot_n),fontsize=14)
        plt.legend(fontsize=14)
        
        plt.ylim([0,600])
        plt.xlim([0, 35])
        
        
        save_data={'shot number':shot_n,
                   'data' :data['waves'],
                    'axuv_Te_time (ms)':t*1000,
                    'axuv_mylar6um_current(nA)':axuv_my06_sig_filter*1e4,
                    'axuv_mylar13um_current(nA)':axuv_my13_sig_filter*1e4,
                    'axuv_mylar22um_current(nA)':axuv_my22_sig_filter*1e4,
                    'axuv_no_filter_current(nA)':axuv_no_filter_sig_filter*1e4,

                    'Ratio of 21um to 6.2um':Ratio2,


                    'Mylar21&6.2_axuv_Te (eV)':Te_forward_model_2,
                   'AXUV_Te2 plus uncertainty':Te2_p_uncertainty,
                   'AXUV_Te2 minus uncertainty':Te2_m_uncertainty,

                    }

    return save_data

if __name__=='__main__':
    shot_number = 20663
    
    axuv_Te_data=axuv_Te_with_error(shot_number)# input the shot number

# 2023/06/14    19074-82
# 2023/06/15    19086-92
# 2023/06/19    19104-113
# 2023/06/20   19117,119,121,123-126  good shot in 2023 campaign
#2023/06/27    19176-193
#2023/06/28    19195-199, from 200 swap ch2 and ch4 to check the 3khz noise source
#2023/06/29    19208   replace the old DB9 connector inside the AXUV Te box
#19546 good shot
#19790-793
#19825- 19840        19826 very good reference
#19848 - 19872   
#19906-19910
#compare 19914 and 19915, when the axvu te soft x-ray signal decays, survey spectrometer shows different behavior, why soft x ray crash ? loss of confinement? what's the time evolution of magnetic field?

#20081-85  fresh lithium coating show very high Te

# 20659- 20678 very good plasma current ,flat and high TS Te
# 20659 sawtooth, MHD 
# 20660 sawtooth, MHD 
# 20662 MHD-free shot
# 20663 sawtooth, MHD 
# 20664 MHD-free shot
# 20665 sawtooth, MHD 
# 20666 MHD at the end of shot
# 20667 sawtooth
# 20668 MHD at the end of shot
# 20670 MHD at the end of shot
# 20671 MHD free, but not flat
# 20672 MHD at the end of shot
# 20673 MHD at the end of shot
# 20674 sawtooth-like
# 20675 MHD-free shot
# 20676 MHD-free shot

 # saturation : 20081,83,84, 20297, 20346，20955
 
#from shot 20740, the calibration factor coefficient is not right, looks some fibers are loose, need to check it.
# the current on ch1 is decreasing with shot and at shot 20780 the current on ch1 with 5um mylar is lower than ch1 with 11um mylar
#The problem is the current on ch1 continues decreasing.....
# fixed from shot 20882, the gate valve is half closed....

# shot 21482- 21522, axuv Te raw data is bad , suspect the valve closed.
# SHOT 21527 data is good 527-545 helium shots

# shot 21548, 21549 multi time TS te comparison
# shot 22603-605 high Te after Lithium coating
# from shot 22472, one mylar channel move to veto board.
#22677  TS Te 500eV

# 20487， 20488，20743,20746, 20755,20757,20581,20584,21284 large difference, 20660,663 discrepacy after sawteeth
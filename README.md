# NMSU_IE_Big_Data

This repository contains the code relevant to my doctoral research program at NMSU with faculty advisor Talayeh Razzaghi. The Python coding is intended to explore algorithmic methods related to the research program.  All code written in Python with Tensorflow/Keras backend.  

 # CCPR Classification with Imbalance

The CCPR (control chart pattern recognition) coding explores algorithmic methods of improving classification metrics for generated and imported data using deep convolutional neural networks.  The code generates normal (~N(0,1)) and abonormal (~N(0,1) with variable downshift, upshift, downtrend, uptrend, systematic, and cyclic adjustment).  Window length, trend type, and trend size are variables within the coding.  The CCPR study also utilizes the Wafer time series data from the University of California - Riverside (UCR Time Series Database).  The extensive UCR-TS database can be access through https://www.cs.ucr.edu/~eamonn/time_series_data/. 

For the CCPR study, this respository contains the following code:
Code for two-class (abnormal/normal) classification using specified settings for window length, trend type, and trend size.
Code for two-class (abnormal/normal) classification to generate a heatmap of G-mean values to determine separable, partially separable, and nonseparable regions (window length and trend size) for each trend type.
Code for analysis of UCR Wafter Time Series data.
Code for multiclass classification of generated data.

# Humanitarian Fuel Demand Prediction

The humanitarian fuel demand coding explores algorithmic methods for improved sequence prediction regression modeling of Defense Logistics Agency (DLA) data from the Japan Tsunmai (2011) and Hurricane Katrina (2005) relief efforts.  This coding uses LSTM (long-short term memory) recurrent neural networks where data is resampled using blocked cross-validation

For the fuel demand study, the repository contains the following code:
Non-resampled Japan sequence data LSTM
Non-resampled Katrina sequence data LSTM

Important Note: Data access to the Humanitarian Fuel dataset must be requested directly from the repository owner (dfuqua@nmsu.edu) 

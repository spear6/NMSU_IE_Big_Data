# NMSU_IE_Big_Data

This repository contains the code relevant to my doctoral research program at NMSU with faculty advisor Talayeh Razzaghi. The Python coding is intended to explore algorithmic methods related to the research program.  All code written in Python with Tensorflow/Keras backend.  

 # CCPR Classification with Imbalance

The CCPR (control chart pattern recognition) coding explores algorithmic methods of improving classification metrics for generated and imported data using deep convolutional neural networks.  The code generates normal (~N(0,1)) and abonormal (~N(0,1) with variable downshift, upshift, downtrend, uptrend, systematic, and cyclic adjustment).  Window length, trend type, and trend size are variables within the coding.  The CCPR study also utilizes the Wafer time series data from the University of California - Riverside (UCR Time Series Database).  The extensive UCR-TS database can be access through https://www.cs.ucr.edu/~eamonn/time_series_data/. 

For the CCPR study, this respository contains the following code:

**CNN_Single_Run.py** Code for two-class (abnormal/normal) classification using specified settings for window length, trend type, and trend size.

User defined method: weighted_mse

Creates cost weighting in loss layer of convolutional neural network

**CNN_Looped_for_Gmean_Specificity.py** Code for two-class (abnormal/normal) classification to generate a heatmap of G-mean values to determine separable, partially separable, and nonseparable regions (window length and trend size) for each trend type.

User defined method: weighted_mse

Creates cost weighting in loss layer of convolutional neural network

**UCR-CNN.py** Code for analysis of UCR Wafter Time Series data.

User defined method: weighted_mse

Creates cost weighting in loss layer of convolutional neural network

# Short Sequence LSTM with modified RPCA preprocessing

This block of coding is built specifically to address short sequence prediction using a novel modified RPCA preprocessing and walk-forward validation teamed with an LSTM.  I use databases on humanitarian supply chain data (fuel) collected from the Defense Logistics Agency (DLA) from Hurricane Katrina (2005) and the Japan Tsunami (2011).  We also utilize two UCI databases (GNFUV and ISE).

**Katrina_RPCA.py** (code for implementing the preprocessing with LSTM using walk forward validation.

User defined method: R_pca (modified from https://github.com/dganguli/robust-pca)

User defined method: tanh_mod (buffers sparse set input during initial training)

User defined method: mahalnobis_dis (searches for sparse set outliers using Mahalanobis distance calculations)

User defined method: var_model (searches for sparse set outliers using Mahalanobis distance calculations)

User defined method: max_eigenvalues (searches for sparse set outliers using comparison of maximum eigenvalues)

User defined method: one_class (preprocessing of sparse set using unsupervised one-class SVM)




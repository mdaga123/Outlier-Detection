# Outlier-Detection
Detects outliers in one-dimensional data using pyod package

This program predicts the outliers in one-dimensional data using a specific machine learning algorithm in the 'pyod' Python package. The user inputs to the program are: - \

1) Outliers Fraction in the data (For eg. 0.05 - 5% data points are outliers) \
2) Machine Learning Algorithm to be used for detecting outliers: - Any one of the below mentioned keywords: -
              Keyword         Machine Learning Algorithm \
              
              hbos             Histogram-based  \       
              cblof            Clustering-based Local Outlier Factor \ 
              iforest          Isolation Forest-based \
              knn              K-Nearest Neighbour based  \ 
              aknn             Average K-Nearest Neighbour based \
              
3) Histogram Binwidth - for plotting of histogram of the data \
4) File Name - File from which data needs to be taken (Excel/CSV/Text file) \ 
               It should be noted that column of data should be named 'stat_value'.
               
 
From my experience, the best Machine Learning Algorithm for outlier detection in one-dimensional data has been Histogram-based for me. You can try other algorithms too. \

Both Jupyter Notebook and Python files contain the same code. Python file can be used for running in CLI: - \

Syntax for running in CLI: - \
finding_outliers.py -o <outliers_fraction> -a <algo_type> -b <histogram_binwidth> -f <file_name> 

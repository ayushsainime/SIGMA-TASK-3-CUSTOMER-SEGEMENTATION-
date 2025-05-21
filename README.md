APP LINK - 
https://btjbejlrzrstrgqkpihhty.streamlit.app/


# Customer Segmentation Dashboard

An interactive Streamlit web application for customer segmentation using unsupervised machine learning (KMeans clustering). The dashboard enables exploration of customer data, visualization of key trends, and dynamic adjustment of the number of customer segments.

![Screenshot 2025-05-21 212731](https://github.com/user-attachments/assets/a2129170-985e-46f9-acab-ece8e2ec5272)

![Screenshot 2025-05-21 212749](https://github.com/user-attachments/assets/0c8732ed-8869-423c-be6b-5265fc84c507)


![Screenshot 2025-05-21 212920](https://github.com/user-attachments/assets/b4f493bd-661b-496e-bf81-8eaf44e938cd)



## Dataset

- **Source:** `Customer_Segmentation_Dataset.csv`  provided by the SIGMA  team ( the bsiness club of NIT TRICHY ) 
- **Description:** Contains customer demographic, behavioral, and spending data for a retail business.
- **Key Features:** Age, Income, Loyalty (days as customer), Number of children, Product spendings, Education, Marital status, and more.

## Features

- **Exploratory Data Analysis:**  
  - Income vs. Total Spend scatter plot
  - Age distribution histogram
  - Loyalty vs. Spend scatter plot
  - Spend by number of children (strip plot)
  - Correlation heatmap

- **Clustering:**  
  - Adjustable KMeans clustering (user selects number of clusters)
  - Cluster size table
  - Cluster medians for key features
  - Radar chart visualizing cluster profiles
  - Elbow and silhouette plots for cluster diagnostics

## Technical Details

- **Language:** Python 3.x
- **App Framework:** [Streamlit](https://streamlit.io/)
- **Key Libraries:**  
  - pandas (data handling)
  - numpy (numerical operations)
  - matplotlib & seaborn (visualization)
  - scikit-learn (scaling, clustering, metrics)

- **Preprocessing Steps:**  
  - Missing value imputation (mean for income)
  - Feature engineering (age, loyalty, total spend, children)
  - One-hot encoding for categorical variables
  - Feature scaling (standardization)




**Start the app:**
   ```bash
   streamlit run streamlit_app.py
   

**Author:** Ayush Saini  
**License:** MIT  

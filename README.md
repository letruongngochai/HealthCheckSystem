#    Features of Health Check System
##    📊 1. Data Overview
* Table of data definitions
* Heart disease dataset summary
![image](https://github.com/letruongngochai/HealthCheckSystem/blob/main/main/image/figure%20(1).png)

##    📉 2. Statistics for Numerical Variables
* Displays summary statistics like mean, std, min, max
* Interpretations for numerical features (age, cholesterol, etc.)
![image](https://github.com/letruongngochai/HealthCheckSystem/blob/main/main/image/figure%20(2).png)

##    🧮 3. Statistics for Categorical Variables
* Frequency distribution of features like sex, cp, thal, etc.
* Insightful analysis written in markdown
![image](https://github.com/letruongngochai/HealthCheckSystem/blob/main/main/image/figure%20(3).png)

##    🔎 4. Data Analysis
* Interactive histograms for numerical variables with KDE plots
* Bar charts for categorical variables with percentage annotations
* Insights per feature based on selection


| Univariate Analysis | Bivariate Analysis |
| -------- | -------- |
| ![image](https://github.com/letruongngochai/HealthCheckSystem/blob/main/main/image/figure%20(4).png)     | ![image](https://github.com/letruongngochai/HealthCheckSystem/blob/main/main/image/figure%20(6).png)    |
| ![image](https://github.com/letruongngochai/HealthCheckSystem/blob/main/main/image/figure%20(5).png)     | ![image](https://github.com/letruongngochai/HealthCheckSystem/blob/main/main/image/figure%20(7).png)     |



##    ❤️ 5. Heart Disease Prediction
* Form-based prediction system
* Takes input parameters like age, chol, thalach, etc.
* Returns whether your heart is fine or not
![image](https://github.com/letruongngochai/HealthCheckSystem/blob/main/main/image/figure%20(8).png)
![image](https://github.com/letruongngochai/HealthCheckSystem/blob/main/main/image/figure%20(9).png)


##    🧠 6. Brain Tumor Segmentation
* Upload MRI images
* Backend UNet model segments and returns tumor overlay
* Uploaded images are stored securely on Google Cloud Storage (GCS)

![image](https://github.com/letruongngochai/HealthCheckSystem/blob/main/main/image/figure%20(10).png)
![image](https://github.com/letruongngochai/HealthCheckSystem/blob/main/main/image/figure%20(11).png)


#    ☁️ Cloud Integration
* Google Cloud Storage is used for storing uploaded medical images.
* Prediction models are hosted on an API backend (FastAPI).

#    🧠 Technologies Used
* Streamlit for UI

* Pandas, Seaborn, Matplotlib for analytics

* Plotly for interactive visualizations

* Google Cloud Storage for image handling

* FastAPI for backend model inference

* scikit-learn, PyTorch for model inference

#    🤖 Model Training
##    1. Heart Disease Prediction
* Data: [heart.csv](https://github.com/letruongngochai/HealthCheckSystem/blob/main/main/src/data/heart.csv)
* Method: Support Vector Machine
##    2.  Brain Tumor Segmentation
* Data: Figshare dataset
* Method: UNet
* [Training Notebook](https://colab.research.google.com/drive/1uLLaTYgcjHuy25ShSNTHF1irsXIGFMhF?usp=sharing)

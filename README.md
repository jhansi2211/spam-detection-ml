# spam-detection-ml

Company: Codtech IT Solutions

Name: Tottaramudi Jhansi Victoriya

Intern ID: CTIS1491

Domain: Python Programming

Duration: 4Weeks

Mentor: Neela Santhosh Kumar

Spam detection using Machine Learning and Scikit-learn:

descriptin:
# Spam Detection System using Machine Learning

This project is a Machine Learning based **Spam Detection System** developed using **Python** and **Scikit-learn**. The main goal of this project is to build a predictive model that can automatically classify SMS messages as **Spam** or **Not Spam (Ham)**. Spam detection is a very important real-world application of machine learning, used in email filtering, messaging apps, and online security systems.

# Project Objective

The objective of this project is to understand how machine learning can be used for **text classification** problems. Here, the task is to train a model that can read a message and decide whether it is spam (unwanted, promotional, or scam message) or ham (normal, genuine message). This project demonstrates the complete machine learning workflow including data preprocessing, feature extraction, model training, and performance evaluation.

# Dataset Used

For this project, I used the **SMS Spam Collection Dataset**, which contains thousands of labeled SMS messages. Each message is marked as either:

- **Ham** → Normal message  
- **Spam** → Unwanted or promotional message  

This dataset is widely used for learning and practicing text classification problems.

# Steps Involved in the Project

# Data Loading and Understanding  
The dataset was loaded using **Pandas**. Since the file did not have column names, I manually assigned the column names as **label** and **message**. This step helped in organizing the data for further processing.

# Data Preprocessing  
Machine learning models cannot understand text labels like "spam" or "ham". So, I converted them into numerical form:
- Spam = 1  
- Ham = 0  

This makes it easier for the model to perform calculations and predictions.

# Splitting the Dataset  
The dataset was divided into **training data (80%)** and **testing data (20%)**.  
The training data is used to teach the model, while the testing data is used to evaluate how well the model performs on unseen messages.

# Text Feature Extraction using TF-IDF  
Computers do not understand text directly, so I used **TF-IDF Vectorization** to convert text messages into numerical feature vectors. TF-IDF gives more importance to meaningful words like *"free"*, *"win"*, and *"offer"*, which are commonly found in spam messages. This step transforms text data into a format suitable for machine learning algorithms.

# Model Training  
I used the **Naive Bayes (MultinomialNB)** algorithm for classification. This algorithm works well for text data and is based on probability. It learns patterns in word usage and predicts whether a message is spam or not.

# Model Evaluation  
After training, the model was tested on unseen data. Its performance was evaluated using:
- **Accuracy Score** – Measures overall correctness  
- **Classification Report** – Shows precision, recall, and F1-score  
- **Confusion Matrix** – Visual representation of correct and incorrect predictions  

The model achieved high accuracy, showing that it can effectively detect spam messages.

# Testing with Custom Messages  
Finally, I tested the model using my own sample messages to see real-time predictions. The system successfully identified spam-like and normal messages.

# Technologies Used

- Python  
- Pandas & NumPy  
- Scikit-learn  
- Matplotlib & Seaborn  
- Jupyter Notebook / VS Code  

# Conclusion

This project helped me understand how machine learning can be applied to real-world problems like spam detection. It shows the full pipeline from raw text data to a trained predictive model. The combination of **TF-IDF** and **Naive Bayes** proved to be effective for text classification tasks. This project strengthened my understanding of Natural Language Processing (NLP) basics and machine learning model evaluation techniques.


⭐ This project is a practical demonstration of how intelligent systems can automatically filter unwanted messages and improve communication security.

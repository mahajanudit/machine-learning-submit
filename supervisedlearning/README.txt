Repository Link: https://github.com/mahajanudit/machine-learning-submit.git
How to obtain the code: git clone https://github.com/mahajanudit/machine-learning-submit.git
Data Set
1. Pollution:
   Link: https://www.kaggle.com/rupakroy/lstm-datasets-multivariate-univariate
   File path in repo: data/LSTM-Multivariate_pollution.csv
2. Credit Card: 
   Link: https://www.kaggle.com/xuandiluo/uci-credit-cardcsv
   File path in repo: data/UCI_Credit_Card.csv

Requirements to run the code:
	matplotlib==3.1.2
	python=3.7.7=h81c818b_4
	scikit-learn==0.23.2

Run Step
1. cd gatech-machine-learning1/supervisedlearning (project directory)
2. Decision Tree: python decisiontree.py
3. Neural Networks: python neuralnetworks.py
  3.1 To do grid search: uncomment line 191 and 197 and comments all other lines in the "__main__" functions. Rerun python neuralnetworks.py
4. Boosting: python boosting.py
5. KNN: python knn.py
6. SVM: python svm.py
  6.1 To do grid search: uncomment grid_search. Rerun python svm.py
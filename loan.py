import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

#Part 1 Load the Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
print("Loading dataset from URL")
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv(url, names=names, header=None, na_values = " ?")
print("Dataset loaded successfully")

print("First 5 rows of the dataset:")
print(df.head())

print("Shape of the dataset")
print(df.shape)

print("Attributes of the dataset")
print(df.columns)

print("Checking for missing values")
print(df.isnull().sum())
df.dropna(inplace=True)
print("Dropping all missing values")
print(df.isnull().sum())
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

#Part 2 Data Preprocessing
print("Encoding categorical variables")
y = df["income"]
x = df.drop("income", axis=1)
y = (y==">50K").astype(int)

catcols = x.select_dtypes(include=['object', 'str']).columns.tolist()
numcols = x.select_dtypes(include=[np.number]).columns.tolist()

le = preprocessing.LabelEncoder()
for col in catcols:
    x[col] = le.fit_transform(x[col])

scaler = preprocessing.StandardScaler()
x_scaled = scaler.fit_transform(x)

x_scaled = pd.DataFrame(x_scaled, columns=x.columns)

print("Shape of processed data")
print(x_scaled.shape)
print("Description of encoding")
print("Label encoding applied to all catecorical columns.")
print("Standardization applied to all numerical columns.")

#Part 3 Train Test Split
print("Splitting data into training and testing sets")
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42, stratify=y)
print("Data split completed")

print("Training size")
print(x_train.shape)

print("Testing size")
print(x_test.shape)

#Part 4 K-Nearest Neighbors
print("K-Nearest Neighbors")
k_values = [1, 3, 5, 7, 9, 11, 15, 20]
results = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    precision = classification_report(y_test, y_pred, output_dict=True)['1']['precision']
    recall = classification_report(y_test, y_pred, output_dict=True)['1']['recall']
    f1 = classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    results.append((k, precision, recall, f1, accuracy))
    print(f"K={k}: Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}, Accuracy={accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion)

results = pd.DataFrame(results, columns=['K', 'Precision', 'Recall', 'F1-Score', 'Accuracy'])

#Part 5 Choose Best K
best_k = results.loc[results['Accuracy'].idxmax()]['K']
print(f"Best K based on accuracy: {best_k}") 

plt.figure(figsize=(10, 6))
plt.plot(results['K'], results['Accuracy'], marker='o', label='Accuracy')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs K')
plt.xticks(results['K'])
plt.grid()
plt.legend()
plt.savefig('accuracy_vs_k.png')
plt.close()
print("Accuracy vs K plot was generated and saved as 'accuracy_vs_k.png'.")
print("The best K value was determined based on the highest accuracy score as shown in the plot.")

#Part 6 Final Model Evaluation
final_knn = KNeighborsClassifier(n_neighbors=int(best_k))
final_knn.fit(x_train, y_train)
final_y_pred = final_knn.predict(x_test)
final_precision = classification_report(y_test, final_y_pred, output_dict=True)['1']['precision']
final_recall = classification_report(y_test, final_y_pred, output_dict=True)['1']['recall']
final_f1 = classification_report(y_test, final_y_pred, output_dict=True)['1']['f1-score']
final_accuracy = accuracy_score(y_test, final_y_pred)
final_confusion = confusion_matrix(y_test, final_y_pred)

print("Final Model Evaluation with K=")
print(best_k)
print("Precision: ")
print(final_precision)
print("Recall: ")
print(final_recall) 
print("F1-Score: ")
print(final_f1)
print("Accuracy: ")
print(final_accuracy)
print("Confusion Matrix:")
print(final_confusion)

fig = plt.figure(figsize=(8, 6))
sns.heatmap(final_confusion, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix for K={best_k}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
display = ConfusionMatrixDisplay(confusion_matrix=final_confusion, display_labels=['<=50K', '>50K'])
display.plot(colorbar=True, cmap='Blues')
plt.savefig('confusion_matrix.png')
plt.close()
print("The confusion matrix heatmap was visualized and saved as 'confusion_matrix.png'.")

#Part 7 Visualization (already completed)
#Part 8 in report


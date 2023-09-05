import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('student_data.csv')

print(data.head())

study_time = data['StudyTime (hours)']
exam_scores = data['ExamScore']

correlation = study_time.corr(exam_scores)

plt.figure(figsize=(8, 6))
plt.scatter(study_time, exam_scores, alpha=0.5)
plt.title('Study Time vs. Exam Scores')
plt.xlabel('Study Time (hours)')
plt.ylabel('Exam Score')
plt.grid(True)
plt.show()

sns.set(style='whitegrid')
sns.pairplot(data, x_vars=['StudyTime (hours)'], y_vars=['ExamScore'], height=6)
plt.title('Pairplot: Study Time vs. Exam Scores')
plt.show()

print(f"Correlation Coefficient (Pearson): {correlation:.4f}")

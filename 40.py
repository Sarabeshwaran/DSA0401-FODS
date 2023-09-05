import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('soccer_players.csv')

top_goals_players = data.nlargest(5, 'Goals')

top_salary_players = data.nlargest(5, 'Salary')

average_age = data['Age'].mean()

above_average_age_players = data[data['Age'] > average_age]

position_counts = data['Position'].value_counts()

plt.figure(figsize=(8, 6))
position_counts.plot(kind='bar')
plt.title('Distribution of Players by Position')
plt.xlabel('Position')
plt.ylabel('Number of Players')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Top 5 Players with Highest Goals Scored:")
print(top_goals_players[['Name', 'Goals']])

print("\nTop 5 Players with Highest Salaries:")
print(top_salary_players[['Name', 'Salary']])

print(f"\nAverage Age of Players: {average_age:.2f}")

print("\nPlayers Above Average Age:")
print(above_average_age_players[['Name', 'Age']])

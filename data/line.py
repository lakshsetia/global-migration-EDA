import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
sns.set_style('whitegrid')
plt.figure(figsize=(12, 8))

# You'll need to replace this with your actual data
# Sample structure:
data = {
    'Year': [1990, 1995, 2000, 2005, 2010, 2015, 2020,2025],
    'China': [3.3,3.3,3.4,3.4,3.4,3.4,3.7,3.9],
    'India': [-1.1,-1.3,-1.5,-1.0,-1.1,-1.1,-1.2,-1.1],
    'UK': [2.6,2.6,4.6,5.1,3.2,3.0,3.0,3.0],
    'France': [0.2,0.7,2.2,1.4,1.7,1.5,1.6,1.4],
    'Germany': [6.5,1.5,1.5,0.8,1.6,3.4,2.7,1.6],
    'USA': [4.1,4.0,2.6,2.1,1.7,1.0,0.9,1.0],
    'Canada': [2.7,2.6,1.9,2.9,2.7,0.7,1.4,1.4],
    'Australia': [1.1,0.8,2.1,3.7,2.7,2.4,1.6,2.0],
    'New Zealand': [2.4,2.8,4.4,2.3,3.4,3.4,2.2,2.4]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Set Year as index to make plotting easier
df_plot = df.set_index('Year')

# Plot each country
for column in df_plot.columns:
    plt.plot(df_plot.index, df_plot[column], marker='o', linewidth=2, label=column)

# Add labels and title
plt.title('Annual Growth Rate by Country (1990-2025)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Annual Growth Rate (%)', fontsize=14)
plt.xticks(data['Year'], rotation=45)
plt.legend(title='Countries', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and display
plt.tight_layout()
plt.show()
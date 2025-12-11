# inserting neccesary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# setting the dataframe 
df = pd.read_csv("marketing_campaign.csv")

# printing the first 5 rows
print(df.head())

# printing the information of the dataframe
df.info()

# checking for missing values
print(df.isnull().sum())

# checking for duplicate values
print(df.duplicated().sum())

# describes the dataframe
print(df.describe())

# Remove dollar signs and commas from the 'Acquisition_Cost' column and convert it to float
df["Acquisition_Cost"] = df["Acquisition_Cost"].replace(r'[\$,]', '', regex=True).astype(float)
# List of numerical columns to check for outliers
numbers = ['ROI', 'Conversion_Rate', 'Clicks', 'Impressions', 'Engagement_Score', 'Acquisition_Cost']
# Display the median (50th percentile) of all numerical columns
print(df.select_dtypes(include=['number']).quantile())

# Plot for boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numbers])
plt.title("Boxplot for Outlier Detection")
plt.xticks(rotation=45)

# Display the plot
plt.show()

#Plot for histogram
df[numbers].hist(figsize=(12, 8), bins=30, color='skyblue', edgecolor='black')
plt.suptitle("Histogram of Numerical Columns")

#Display the plot
plt.show()

# plot for Pie Chart
plt.figure(figsize=(8, 8))
df['Channel_Used'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('Set2'), startangle=90)
plt.title("Distribution of Marketing Channels")
plt.ylabel('')  
# Display the plot
plt.show()

# Group by marketing channel and calculate the average ROI, then sort the values
channel_performance = df.groupby('Channel_Used')['ROI'].mean().sort_values()
# a bar plot for average ROI by marketing channel
plt.figure(figsize=(10, 5))
sns.barplot(x=channel_performance.index, y=channel_performance.values, hue=channel_performance.index, legend=False, palette='viridis')
# plot title and axis labels
plt.title("Average ROI by Marketing Channel")
plt.xlabel("Marketing Channel")
plt.ylabel("Average ROI")
plt.xticks(rotation=45)
# Display the plot
plt.show()

# Calculating Click-Through Rate (CTR) (%)
df['CTR'] = (df['Clicks'] / df['Impressions']) * 100
# Calculating Cost Per Click (CPC)
df["CPC"] = df["Acquisition_Cost"] / df["Clicks"]
# print 5 values of CTR & CPC 
print(df['CTR'].head())
print(df['CPC'].head())

# print performance first 5 rows & columns 
performance = df.groupby('Campaign_ID')[['CTR', 'CPC', 'Conversion_Rate', 'ROI']].mean().reset_index()
print(performance.head())

# Set figure size for better visibility
plt.figure(figsize=(10, 6))
# Creating a scatter plot to visualize the relationship between Engagement Score and ROI
sns.scatterplot(
    x=df['Engagement_Score'] + np.random.uniform(-0.1, 0.1, df.shape[0]), 
    y=df['ROI'], 
    hue=df['Channel_Used'], 
    palette='Set2', alpha=0.7
)
# Add plot title and axis labels
plt.title("ROI vs Engagement Score")
plt.xlabel("Engagement Score")
plt.ylabel("ROI")
# Move legend outside to avoid overlap
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  
plt.tight_layout()  
# Save without background
plt.savefig("ROI_vs_Engagement.png", transparent=True, dpi=300)
# Display the plot
plt.show()

# Set figure size for better readability
plt.figure(figsize=(10, 6))
# Creating a heatmap to visualize the correlation between numerical features
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
# Display the plot
plt.show()

# Convert the 'Date' column to datetime format, ensuring day comes first
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
# Group the dataset by 'Date' and calculate the mean ROI for each day
roi_trend = df.groupby('Date')['ROI'].mean()
# Set figure size for better visualization
plt.figure(figsize=(12, 5))
# Plot the ROI trend over time using a line plot
plt.plot(roi_trend.index, roi_trend.values, marker='o', linestyle='-', color='b')
# setting the title to the plot
plt.title("ROI Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Average ROI")
# Rotate x-axis labels for better visibility
plt.xticks(rotation=45)
# Display the plot
plt.show()


# Set figure size for better visualization
plt.figure(figsize=(8, 6))
# Creating a scatter plot to analyse the relationship between Cost Per Click (CPC) and ROI
sns.scatterplot(x=df['CPC'], y=df['ROI'], hue=df['Channel_Used'], palette='coolwarm')
# setting title and axis labels for better readability
plt.title("CPC vs ROI")
plt.xlabel("Cost Per Click (CPC)")
plt.ylabel("Return on Investment (ROI)")
# Display the plot
plt.show()

# Define a list of categorical columns to analyze
categorical_cols = ['Company', 'Campaign_Type', 'Target_Audience', 'Channel_Used', 'Location', 'Customer_Segment']
# Loop through each categorical column and display unique values with their counts
for col in categorical_cols:
    print(f"\nUnique values in {col}:")
    print(df[col].value_counts())


import requests
from bs4 import BeautifulSoup
import pandas as pd

a = 'a'
# Send a GET request to the website URL
url = 'https://www.slickcharts.com/sp500'
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table containing the company information
table = soup.find('table', {'class': 'table table-hover table-borderless table-sm'})
print(a)
print(table)

# Extract the table headers
headers = [th.text.strip() for th in table.find_all('th')]

# Extract the table rows
rows = []
for tr in table.find_all('tr')[1:]:
    rows.append([td.text.strip() for td in tr.find_all('td')])
    print (rows)

# Create a Pandas DataFrame from the headers and rows
df = pd.DataFrame(rows, columns=headers)

# Save the DataFrame to a CSV file
df.to_csv('sp500.csv', index=False)

print('Data saved to CSV file.')
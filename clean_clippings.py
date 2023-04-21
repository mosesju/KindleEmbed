filename = "kindle-clippings.txt"

with open(filename, 'r') as f:
    lines = f.readlines()

print(lines)

quotes = []
for i, line in enumerate(lines):
    if 'Your Highlight' in line:
        quote = lines[i+2].strip()
        quotes.append(quote)

import pandas as pd
df = pd.DataFrame(quotes, columns=["quote"])
df.to_csv('list.csv')
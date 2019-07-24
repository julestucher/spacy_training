import pandas as pd
import sys
import io

data = pd.read_csv('mturk-results-no-dups.csv')
new_data = {'Answer': [], 'TextTitle': [], 'Start': [], 'End': [], 'Titles': []}

with open('temp1.txt', 'r') as file:
    f = file.read().replace('\n', '')
i = 0
d = 872
while d < len(data['Answer']):
    s = f.find(']', i)
    if s == -1:
        break
    if f[s+1:s+2] == "'":
        new_data['Answer'].append(data['Answer'][d])
        new_data['TextTitle'].append(data['TextTitle'][d])
        new_data['Start'].append(data['Start'][d])
        new_data['End'].append(data['End'][d])
        new_data['Titles'].append(data['Titles'][d])
    i = s + 1
    d += 1


new = pd.DataFrame(data=new_data)
v3 = pd.read_csv('mturk-results-v4.csv')
print(new)
v3 = v3.append(new, sort=True)
v3 = v3.reset_index(drop=True)
v3.to_csv('mturk-results-v5.csv')
'''
d = 868
while True:
    print(data['Answer'][d])
    print(data['TextTitle'][d])
    print(data['Titles'][d])
    try:
        s=input()
        if s != '':
            new_data['Answer'].append(data['Answer'][d])
            new_data['TextTitle'].append(data['TextTitle'][d])
            new_data['Start'].append(data['Start'][d])
            new_data['End'].append(data['End'][d])
            new_data['Titles'].append(data['Titles'][d])
    except EOFError:
        break
    d += 1

new = pd.DataFrame(data=new_data)
v3 = pd.read_csv('mturk-results-v3.csv')
v3.append(new)
v3.to_csv('mturk-results-v4.csv')
'''

import pandas as pd

df = pd.read_csv('/mnt/data/test_file.csv')
print("Successful read of mounted data")
print(df)

df = pd.DataFrame({'name': ['Raphael', 'Donatello'],
                   'mask': ['red', 'purple'],
                   'weapon': ['sai', 'bo staff']})
df.to_csv('out.csv', index=False)
print("Successful save of output csv")


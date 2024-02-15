import pandas as pd
data={
    'Name':['John','Emma','Sant','Lisa','Tom'],
    'Age':[25,30,28,32,27],
    'Country':['USA','Canada','India','UK','Australia'],
    'Salary':[50000,60000,70000,80000,65000]
}

df=pd.DataFrame(data)
print("Original DataFrame")
print(df,"\n")

name_age=df[['Name','Age']]
print("Name and age colums")
print(name_age,"\n")


sorted_df=df.sort_values("Salary",ascending=False)
print("\n Sorted dataframe(by salary in descending order)")
print(sorted_df)

average_Salary=df['Salary'].mean()
print("\n Average salary",average_Salary)
df['Experience']=[3,6,4,6,5]
print("\n Dataframe with added exprience")
print(df)

df.loc[df['Name']=='Emma','Salary']=65000
print("\n Dataframe with updating emma's salary")
print(df)

df=df.drop("Experience",axis=1)
print("\n Dataframe after deleting the  experience col")
print(df)

import pandas as pd
data={
    'Name':['John','Emma','Sant','Lisa','Tom'],
    'Age':[25,30,28,32,27],
    'Country':['USA','Canada','India','UK','Australia'],
    'Salary':[50000,60000,55000,70000,52000]
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




output:
Original DataFrame
   Name  Age    Country  Salary
0  John   25        USA   50000
1  Emma   30     Canada   60000
2  Sant   28      India   55000
3  Lisa   32         UK   70000
4   Tom   27  Australia   52000 

Name and age colums
   Name  Age
0  John   25
1  Emma   30
2  Sant   28
3  Lisa   32
4   Tom   27 


 Sorted dataframe(by salary in descending order)
   Name  Age    Country  Salary
3  Lisa   32         UK   70000
1  Emma   30     Canada   60000
2  Sant   28      India   55000
4   Tom   27  Australia   52000
0  John   25        USA   50000

 Average salary 57400.0

 Dataframe with added exprience
   Name  Age    Country  Salary  Experience
0  John   25        USA   50000           3
1  Emma   30     Canada   60000           6
2  Sant   28      India   55000           4
3  Lisa   32         UK   70000           6
4   Tom   27  Australia   52000           5

 Dataframe with updating emma's salary
   Name  Age    Country  Salary  Experience
0  John   25        USA   50000           3
1  Emma   30     Canada   65000           6
2  Sant   28      India   55000           4
3  Lisa   32         UK   70000           6
4   Tom   27  Australia   52000           5

 Dataframe after deleting the  experience col
   Name  Age    Country  Salary
0  John   25        USA   50000
1  Emma   30     Canada   65000
2  Sant   28      India   55000
3  Lisa   32         UK   70000
4   Tom   27  Australia   52000
​
'

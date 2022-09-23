
import requests
from bs4 import BeautifulSoup
import pandas as pd
url =('https://www.reuters.com/markets/companies/AAPL.OQ','https://www.reuters.com/markets/companies/AMZN.OQ/')

members = []

person_name = []

for data in url:
    members = []
    response = requests.get(data).text
    soup = BeautifulSoup(response,'html5lib')

    search = soup.find('div',{"class":"about-company-card__company-leadership__1mNWX"})
    for dt,dd in zip(search.find_all('dt'),search.find_all('dd')):
        members.append((dt.text.strip(),dd.text.strip()))


    person_name.append(members)

c=1
for name in person_name:
    df = pd.DataFrame(name,columns=['Name','Job_Title'])
    print(df)
    df.to_csv(str(c)+'.csv',index=False)
    c += 1
print("Scrapping worked")

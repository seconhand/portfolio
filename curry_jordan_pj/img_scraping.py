from bs4 import BeautifulSoup
import requests
import urllib.request
import urllib


url = "https://www.sportslogos.net/teams/list_by_league/6/National_Basketball_Association/NBA/logos/"

r = requests.get(url)
soup = BeautifulSoup(r.content, "lxml")
#%%
#이미지 src 와 이름 태그 추출
img_src = soup.find('ul',{'class':'logoWall'}).find_all('img')
img_name = soup.find('ul',{'class':'logoWall'}).find_all('a')
#%%
#이미지src 리스트
img_lst = []
for img in img_src:
    src = img['src']
    img_lst.append(src)
#%%
#이름 리스트
name_lst = []
for name in img_name:
    n = name.get_text().strip()
    name_lst.append(n)      
#%%    
#이미지 저장
for i in range(len(img_lst)): 
    try:
        urllib.request.urlretrieve(img_lst[i],name_lst[i]+'.jpg')
    except Exception:
        pass


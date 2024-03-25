import requests

API_KEY = 'aFK4nHp8_Xs4RAUzrSxzxl119ulIWcom'

def get_news(ticker, day, month, year):
    limit = '10'
    api_url = f'https://api.polygon.io/v2/reference/news?limit={limit}&sort=published_utc&ticker={ticker}&published_utc.gte={year}-{month}-{day}&apiKey={API_KEY}'

    response = requests.get(api_url)

    #print(api_url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error accessing API: {response.status_code}")
        return None

def test_news():
    data = get_news('AAPL', '10', '02', '2021')
    if data:
        print(data['results'][0]['description'])
    else:
        print("No data retrieved")

if __name__ == "__main__":
    test_news()

import requests

url = 'https://api.threatbook.cn/v3/file/report'
params = {
    'apikey': '请替换apikey',
    'sandbox_type': 'ubuntu_1704_x64',
    'sha256': '请替换sha256'
}

if __name__ == "__main__":
    response = requests.get(url, params=params)
    print(response.json())
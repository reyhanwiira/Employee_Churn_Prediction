import requests
import json

class Generator:
    def __init__(self, url):
        self.url = url

    def get_recommendations(self, input_data):
        response = requests.post(self.url, json=input_data)
        return response
    
    def get_response(self, input_data):
        response = self.get_recommendations(input_data)
        return json.loads(response.content)
    
"""
    Wikihow class to download and preprocess Wikihow dataset
"""

import os
from bs4 import BeautifulSoup
import requests
import json

class Wikihow:
    def __init__(self, folder=None):
        self.folder=folder


    def download(self):
        # Get list of url categories
        print("Downloading Wikihow dataset ...")
        
        categories_url_list = []
        with open('./wikihow/categories_list', 'r') as file:
            line = file.readline()
            while line:
                categories_url_list.append(line)
                line = file.readline()

        # Create dataset folder
        if not os.path.exists(self.folder):
            try:
                os.makedirs(self.folder)
            except OSError:
                print("Error creating dataset directory {}".format(self.folder))

        # Get url links by parsing categories url
        url_list = []

        for category_url in categories_url_list[:2]:
            req = requests.get(category_url)
            soup = BeautifulSoup(req.content, 'html.parser')

            div_categories = soup.findAll('div', {'class': 'responsive_thumb'})

            for div in div_categories:
                link = div.find('a')
                url_list.append(link.get('href'))

        # Download url content
        for url in url_list[:2]:
            req = requests.get(url)
            soup = BeautifulSoup(req.content, 'html.parser')

            scripts = soup.findAll('script', {'type': 'application/ld+json'})
            js = None
            steps = None

            for script in scripts:
                js = json.loads(script.string)

                if 'headline' in js:
                    headline = js['headline']

                if 'step' in js:
                    steps = js['step']
                    break

            if steps == None:
                print("Ignoring file {} (wrong format) ...".format(file_name))
                continue

            # Write content to file
            file_name = url.split('/')[-1]

            with open(os.path.join(os.path.abspath(self.folder), file_name), 'w') as file:
                file.write("TITLE: {} ({})\n".format(str.upper(headline), url))
                idx = 1

                for step in steps:
                    if step['@type'] == 'HowToStep':
                        file.write("\nSTEP {}. {}\n".format(idx, step['text']))
                    elif step['@type'] == 'HowToSection':
                        file.write("\nSECTION: {}\n".format(str.upper(step['name'])))

                        for item in step['itemListElement']:
                            if item['@type'] == 'HowToStep':
                                file.write("\nSTEP {}. {}\n".format(idx, item['text']))
                                idx += 1
                    idx += 1



        return None

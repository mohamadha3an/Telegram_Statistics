import json
from collections import Counter
from pathlib import Path

import arabic_reshaper
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from loguru import logger
from parsivar import Normalizer, Tokenizer
from src.data import DATA_DIR
from wordcloud import WordCloud


class ChatStatistics:
    def __init__(self, chat_json):
        """_this class defines chat statistics and generates wordcloud from json file of telegram chat_

        Args:
            chat_json (_.json_): _telegram chat data in format of json file_
        """

        #load Chat data
        logger.info(f"Loading chat data from {chat_json}")
        with open(chat_json) as f:
            self.chat_data  = json.load(f)

        self.my_normalizer = Normalizer()

        #load stop_words
        logger.info(f"Loading stopwords from {DATA_DIR / 'stopwords.txt'}")
        stop_words = open(DATA_DIR/ 'stopwords.txt').readlines()
        stop_words = list(map(str.strip, stop_words))
        self.stop_words = list(map(self.my_normalizer.normalize, stop_words))

    def generate_word_cloud(self, output_dir):
        """_it generates the wordcloud of telegram chat_

        Args:
            output_dir (_path_): _path for saving the wordcloud picture in format of .png_
        """
        #tokenizing and normilizing the chat words by parsivar 
        logger.info("Loading text content...")
        text_content = ''
        my_tokenizer = Tokenizer()
        my_normalizer = Normalizer()

        for msg in self.chat_data['messages']:
            if type(msg['text']) is str:
                tokens = my_tokenizer.tokenize_words(my_normalizer.normalize(msg['text']))
                tokens = list(filter(lambda item: item not in self.stop_words, tokens))
                text_content += f" {' '.join(tokens)}"

        #arabic_reshaper and get_display is necessary for generating wordcloud
        text_content = self.my_normalizer.normalize(text_content)
        text_content = arabic_reshaper.reshape(text_content)
        text_content = get_display(text_content)

        #generating wordcloud
        wordcloud = WordCloud(
            font_path=str(DATA_DIR / 'Vazirmatn-Medium.ttf'),
            width=1200, height=800,
            background_color='white',
        ).generate(text_content)

        wordcloud.to_file(str(Path(output_dir) / 'WordCloud.png'))


if __name__ == "__main__":
    Chat_stats = ChatStatistics(chat_json=DATA_DIR / 'result.json')
    Chat_stats.generate_word_cloud(output_dir=DATA_DIR)

    print("Done...!")

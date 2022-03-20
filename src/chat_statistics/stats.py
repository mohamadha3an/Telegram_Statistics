import json
from collections import Counter
from pathlib import Path

import arabic_reshaper
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from parsivar import Normalizer, Tokenizer
from src.data import DATA_DIR
from wordcloud import WordCloud


class ChatStatistics:
    def __init__(self, chat_json):

        #load Chat data
        with open(chat_json) as f:
            self.chat_data  = json.load(f)

        self.my_normalizer = Normalizer()

        #load stop_words
        stop_words = open(DATA_DIR/ 'stopwords.txt').readlines()
        stop_words = list(map(str.strip, stop_words))
        self.stop_words = list(map(self.my_normalizer.normalize, stop_words))

    def generate_word_cloud(self, output_dir):
        text_content = ''
        my_tokenizer = Tokenizer()
        my_normalizer = Normalizer()

        for msg in self.chat_data['messages']:
            if type(msg['text']) is str:
                tokens = my_tokenizer.tokenize_words(my_normalizer.normalize(msg['text']))
                tokens = list(filter(lambda item: item not in self.stop_words, tokens))
                text_content += f" {' '.join(tokens)}"

        text_content = self.my_normalizer.normalize(text_content)
        text_content = arabic_reshaper.reshape(text_content)
        text_content = get_display(text_content)

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

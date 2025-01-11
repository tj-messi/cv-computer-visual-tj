from typing import Optional, Dict

from .config import Config
from .cost import Edit, EditOp
from .result import Result, ResultItem
from .state import State


class Trie:
    """
    Trie data structure for fast search of words in a dictionary with Levenshtein distance.

    Parameters
    ----------
    word : str
        The word that is represented by the current node

    """

    children: Dict[str, 'Trie']
    word: str
    is_word: bool

    def __init__(self, word: str = ''):
        self.children = {}
        self.word = word
        self.is_word = False

    def insert(self, word: str) -> None:
        """
        Insert a word into the trie.
        :param word: word to insert
        :return: None
        """
        word = word.lower()
        if word == '':
            self.is_word = True
            return

        if word[0] not in self.children:
            self.children[word[0]] = Trie(self.word + word[0] if self.word else word[0])

        self.children[word[0]].insert(word[1:])

    def find(self, word: str) -> bool:
        """
        Find an exact word in the trie (Levenshtein distance is zero).

        :param word:
        :return:
            True if the word is in the trie, False otherwise
        """
        if word == '':
            return self.is_word

        if word[0] not in self.children:
            return False

        return self.children[word[0]].find(word[1:])

    def find_dist(self, word: str, state: State, config: Config) -> Optional[Result]:
        """
        Find all words in the trie with Levenshtein distance to the given word less than or equal to the given distance.

        :param word: Word for which we are looking for similar words
        :param state: Current state of the search
        :param config: Configuration of the search
        :return: Result object with all words with Levenshtein distance less than or equal to the given distance
        """
        if state.distance < 0:
            return None

        words = Result(config)
        if word == '':
            if self.is_word:
                item = ResultItem(self.word, config.max_distance - state.distance, state.updates)
                words.add(item)
            if state.distance > 0:
                for k, v in self.children.items():
                    t = Edit(EditOp.DELETE, k)
                    words.update(v.find_dist('', state.step(t), config))
        else:
            c = word[0]
            for k, v in self.children.items():
                if k == c:
                    words.update(v.find_dist(word[1:], state.step(), config))
                else:
                    if state.distance > 0:
                        t = Edit(c, k)
                        words.update(v.find_dist(word[1:], state.step(t), config))

                        t = Edit(EditOp.ADD, k)
                        words.update(v.find_dist(word, state.step(t), config))
            if state.distance > 0:
                t = Edit(EditOp.DELETE, c)
                words.update(self.find_dist(word[1:], state.step(t), config))
        return words

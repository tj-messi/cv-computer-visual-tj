from dataclasses import dataclass
from typing import List, Optional, Dict

from .cost import Edit


@dataclass
class ResultItem:
    word: str
    dist: int
    updates: List[Edit]


class Result:
    words: Dict[str, ResultItem]

    def __init__(self, config):
        self.words = {}
        self.config = config

    def add(self, item: ResultItem):
        if item.word in self.words:
            if self.words[item.word].dist > item.dist:
                self.words[item.word] = item
        else:
            self.words[item.word] = item

    def update(self, other_result: Optional['Result']):
        if other_result is None:
            return
        if self.words == {}:
            self.words = other_result.words
            return
        for item in other_result.words.values():
            self.add(item)

    def __repr__(self):
        lines = ["Result:"]
        for word, item in self.words.items():
            lines.append(f"\t{word}: {item}")
        return "\n".join(lines)

    def is_in(self, word):
        return word.lower() in self.words

    def get_distance(self, word):
        data = self.words.get(word.lower(), None)
        return data.dist if data else None

    def get_result(self, word):
        return self.words.get(word.lower(), None)

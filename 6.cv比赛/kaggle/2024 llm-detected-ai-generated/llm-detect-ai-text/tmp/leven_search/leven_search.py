from typing import List, Dict

from .config import Config
from .cost import GranularEditCostConfig, EditCostConfig, EditCost
from .state import State
from .trie import Trie


class LevenSearch:
    """
    API class for searching words in a dictionary with Levenshtein distance.

    Before using this class, you need to insert all words from the dictionary using the method `insert`.

    Once all words are inserted, you can search for words with Levenshtein distance to the given word less than or equal
    to the given distance using the method `find_dist`.

    Also, there is a method `find` for searching exact words in the dictionary.
    """
    children: Dict[str, Trie]
    root_trie: Trie

    def __init__(self):
        self.children = {}
        self.root_trie = Trie()

    def insert(self, word: str) -> None:
        """
        Before
        :param word: word to insert
        :return: None
        """
        self.root_trie.insert(word.lower())

    def find(self, word: str) -> bool:
        """
        Find an exact word in the trie (Levenshtein distance is zero).
        :param word: word to find
        :return: True if the word is in the index, False otherwise
        """
        return self.root_trie.find(word.lower())

    def find_dist(self,
                  word: str,
                  max_distance: int = 0,
                  edit_cost_config: EditCostConfig | List[EditCost] | int | None = None):
        """
        Find all words in the index with Levenshtein distance to the given word less than or equal
        to the given distance.
        :param word: word to find
        :param max_distance: maximum Levenshtein distance
        :param edit_cost_config: granular edit cost.
            - If this is int, then the cost of all edits is equal to this value.
            - If this is a list of EditCost objects, then the cost of edits is taken from this list.
            - If this is EditCostConfig object, then the cost of edits is taken from this object.
        :return: Result object with all words with Levenshtein distance to the given word less than or equal
        """
        word = word.lower()
        config = Config(max_distance)
        if isinstance(edit_cost_config, EditCostConfig):
            edit_cost_config = edit_cost_config
        elif type(edit_cost_config) is int:
            edit_cost_config = EditCostConfig(edit_cost_config)
        elif type(edit_cost_config) is list:
            edit_cost_config = GranularEditCostConfig(edit_costs=edit_cost_config)
        elif edit_cost_config is not None:
            raise ValueError(f"edit_cost must be a list or EditCost object, got {type(edit_cost_config)}")
        else:
            edit_cost_config = EditCostConfig()
        state = State(word, max_distance, 0, [], edit_cost_config)
        return self.root_trie.find_dist(word, state, config)

"""Dictionary encoding utility.

Author: David
"""


class DictEncoding:
    """Build character-index mapping from a text dictionary file."""

    def __init__(self, dict_file: str) -> None:
        self.dict_mapping: dict[str, int] = {}
        self.dicts = ""

        index = 1
        with open(dict_file, "r", encoding="utf-8") as file:
            for line in file:
                char = line.strip() or " "
                self.dict_mapping[char] = index
                index += 1

        self.dicts = "".join(self.dict_mapping.keys())

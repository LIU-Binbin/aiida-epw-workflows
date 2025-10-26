from collections.abc import Mapping, Iterable
from rich import print as rprint


class Printer:
    prefix_item = '├── '
    prefix_last_item = '└── '
    prefix_indent = '    '
    prefix_parent = '│   '

    _print = print
    def __init__(self, data: dict | list):
        """
        Args:
            data (dict | list):
        """

        if not isinstance(data, (Mapping, Iterable)) or isinstance(data, str):
            raise TypeError("Input data must be a map or iterable.")
        self.data = data

    def print(self):
        if not self.data:
            self._print(self.data)
            return
        self._print_recursive(self.data)

    def _print_recursive(self, data, prefix: str = ""):
        # if data is a dict, convert to list of items
        is_mapping = isinstance(data, Mapping)
        # convert to list of items
        items = list(data.items()) if is_mapping else list(enumerate(data))

        for i, item in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = self.prefix_last_item if is_last else self.prefix_item

            if is_mapping:
                key, value = item
                self._print(f"{prefix}{connector}{key}")
            else:  # if data is an iterable
                index, value = item
                self._print(f"{prefix}{connector}[{index}]")

            # prepare prefix for next level
            new_prefix = prefix + (self.prefix_indent if is_last else self.prefix_parent)

            # recursive condition: value is a map or non-string iterable
            if isinstance(value, (Mapping, Iterable)) and not isinstance(value, str):
                self._print_recursive(value, new_prefix)
            else:
                # if value is a leaf node (basic type), print it directly
                self._print(f"{new_prefix}{self.prefix_last_item}{repr(value)}")
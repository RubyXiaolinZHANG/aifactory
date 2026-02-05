
class YAMLTreePrinter:
    """树形结构打印 YAML"""

    def __init__(self, yaml_data, log=None):
        self.data = yaml_data
        self._log = log

    def print_tree(self):
        """Print tree structure"""
        if self._log is None:
            print(f"YAML Tree Structure:")
            print("=" * 50)
        else:
            self._log.info(f"YAML Tree Structure:\n", raw=True)
            self._log.info("=" * 50 + "\n", raw=True)
        self._print_node(self.data)

    def _print_node(self, node, prefix="", is_last=True):
        """Recursively print node"""
        # Determine the prefix for the current line
        connector = "└── " if is_last else "├── "
        if self._log is None:
            print(prefix + connector, end="")
        else:
            self._log.info(prefix + connector, raw=True)

        if isinstance(node, dict):
            # If it's a dictionary, first print the number of keys
            if self._log is None:
                print(f"{{Total {len(node)} items}}")
            else:
                self._log.info(f"{{Total {len(node)} items}}\n", raw=True)

            items = list(node.items())
            for i, (key, value) in enumerate(items):
                # Determine if this is the last item
                last = (i == len(items) - 1)
                new_prefix = prefix + ("    " if is_last else "│   ")

                # Print key
                if self._log is None:
                    print(new_prefix + ("└── " if last else "├── ") + str(key) + ": ", end="")
                else:
                    self._log.info(new_prefix + ("└── " if last else "├── ") + str(key) + ": ", raw=True)

                if isinstance(value, (dict, list)):
                    if self._log is None:
                        print()
                    else:
                        self._log.info("\n", raw=True)
                    self._print_node(value, new_prefix + ("    " if last else "│   "), last)
                else:
                    if self._log is None:
                        print(str(value))
                    else:
                        self._log.info(str(value) + "\n", raw=True)

        elif isinstance(node, list):
            # If it's a list, first print the length
            if self._log is None:
                print(f"[List, total {len(node)} items]")
            else:
                self._log.info(f"[List, total {len(node)} items]\n", raw=True)
            for i, item in enumerate(node):
                last = (i == len(node) - 1)
                new_prefix = prefix + ("    " if is_last else "│   ")
                if self._log is None:
                    print(new_prefix + ("└── " if last else "├── ") + f"[{i}]: ", end="")
                else:
                    self._log.info(new_prefix + ("└── " if last else "├── ") + f"[{i}]: ", raw=True)

                if isinstance(item, (dict, list)):
                    if self._log is None:
                        print()
                    else:
                        self._log.info("\n", raw=True)
                    self._print_node(item, new_prefix + ("    " if last else "│   "), last)
                else:
                    if self._log is None:
                        print(str(item))
                    else:
                        self._log.info(str(item) + "\n", raw=True)
        else:
            # Basic type
            if self._log is None:
                print(str(node))
            else:
                self._log.info(str(node) + "\n", raw=True)

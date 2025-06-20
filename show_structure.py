import os


def print_tree(start_path=".", prefix=""):
    items = sorted(os.listdir(start_path))
    items = [item for item in items if not item.startswith(".")]  # Skip hidden files
    entries = [e for e in items if os.path.isfile(os.path.join(start_path, e))]
    folders = [e for e in items if os.path.isdir(os.path.join(start_path, e))]

    for index, name in enumerate(entries + folders):
        path = os.path.join(start_path, name)
        connector = "└── " if index == len(entries + folders) - 1 else "├── "
        print(prefix + connector + name)

        if os.path.isdir(path):
            extension = "    " if index == len(entries + folders) - 1 else "│   "
            print_tree(path, prefix + extension)


if __name__ == "__main__":
    print(".")
    print_tree()

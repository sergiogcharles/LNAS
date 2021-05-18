import os

cwd = os.getcwd()

filename = cwd + "/foo/bar/baz.txt"
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, "w") as f:
    f.write(str(0.33))
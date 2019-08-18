import os

path = '../LogosClean/voc_format/' # remember backslash
classes = next(os.walk(path))[1]
for name in classes:
    os.rename(path + name, path + name.replace(' ', ''))
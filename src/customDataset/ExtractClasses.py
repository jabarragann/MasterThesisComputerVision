import json
from pathlib import Path

def hex_to_rgb(hexColor):
    hexColor = hexColor.lstrip('#')
    hlen = len(hexColor)
    return tuple(int(hexColor[i:i + hlen // 3], 16) for i in range(0, hlen, hlen // 3))

if __name__ == '__main__':

    classesFile = "./classes/classes.json"

    f = open(classesFile,'r')
    data = json.load(f)

    simplifiedFile = []
    for cl in data:
        simplifiedFile.append({"name": cl['name'], "color": hex_to_rgb(cl['color'])})
        print(cl['color'], hex_to_rgb(cl['color']), cl["name"])

    #Add background class
    simplifiedFile.append({"name": "Background", "color": tuple([0,0,0])})

    simplifiedFile = json.dumps(simplifiedFile, indent=4)
    # Writing to colors to json
    with open("label_colors.json", "w") as outfile:
        outfile.write(simplifiedFile)

    f.close()
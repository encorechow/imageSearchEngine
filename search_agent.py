from searchEngine.colordescriptor import ColorDescriptor
from searchEngine.searcher import Searcher
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True,
    help = "Path to where the computed index will be stored")
ap.add_argument("-q", "--query", required = True,
    help = "Path to the query image")
ap.add_argument("-d", "--dataset", required = True,
    help = "Path to the result path")
args = vars(ap.parse_args())


cd = ColorDescriptor((8, 12, 3))

# Read query image and compute features
query = cv2.imread(args["query"])
features = cd.describe(query)

# Perform actual search
searcher = Searcher(args["index"])
results = searcher.search(features)

# Show result
cv2.imshow("Query", query)
for (score, imageId) in results:
    result = cv2.imread(args["dataset"] + "/" + imageId)
    cv2.imshow("Result", result)
    cv2.waitKey(0)

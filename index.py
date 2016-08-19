from colordescriptor import ColorDescriptor
import argparse
import glob
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
args = vars(ap.parse_args())

cd = ColorDescriptor((8, 12, 3))


# Open index file for writing output
output = open(args["index"], "w")

for ipath in glob.glob(args["dataset"] + "/*.jpg"):
    # Extract the image ID (filename) from the image path and load image
    imageId = ipath[ipath.rfind("/") + 1:]
    image = cv2.imread(ipath)

    # Describe the image
    features = cd.describe(image)

    # Write features to file
    features = [str(f) for f in features]
    output.write("%s,%s\n" % (imageId, ",".join(features)))

output.close()

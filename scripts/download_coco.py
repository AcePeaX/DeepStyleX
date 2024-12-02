import fiftyone as fo
import fiftyone.zoo as foz

# List available zoo datasets
print(foz.list_zoo_datasets())

# Download the COCO-2017 validation split and load it into FiftyOne
dataset = foz.load_zoo_dataset("coco-2017", split="validation")

# Give the dataset a new name, and make it persistent
dataset.name = "coco-2017-validation-example"
dataset.persistent = True

# Visualize it in the App
session = fo.launch_app(dataset)
#Vizualize the images and their boxes 

import matplotlib.pyplot as plt
import cv2
import json

# Specify the path to the image files
image_folder = ''

# Specify the path to the JSON file containing bounding box annotations
json_path = 'annotations/instancesonly_filtered_gtFine_train.json'

# Load the bounding box annotations from the JSON file
with open(json_path, 'r') as f:
    annotations = json.load(f)

# Get the images list from the JSON file
images = annotations['images']

# Specify the number of images to display
num_images_to_display = 5

# Initialize a counter variable
counter = 0
# Iterate over the images list
for image_info in images:
    # Check if the counter exceeds the specified limit
    if counter >= num_images_to_display:
        break
    
    # Get the image file name and image ID
    file_name = image_info['file_name']
    image_id = image_info['id']
    
    # Construct the complete image path
    image_path = image_folder + file_name
    
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    if image is not None:
        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract the bounding box coordinates and class labels
        for annotation in annotations['annotations']:
            bbox = annotation['bbox']
            class_label = annotation['category_id']
            annotation_image_id = annotation['image_id']

            # Check if the image ID matches the current image
            if annotation_image_id == image_id:
                # Find the category name for the class label
                for category in annotations['categories']:
                    if category['id'] == class_label:
                        label = category['name']
                        break

                # Draw the bounding box on the image
                x, y, w, h = bbox
                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
                cv2.putText(image, str(label), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the image with bounding box annotations
        plt.imshow(image)
        plt.axis('off')  # Optional: Turn off axis labels
        plt.show()
        
        # Increment the counter
        counter += 1
    else:
        print("Failed to load the image:", image_path)

#Check if the desired number of images is not reached
if counter < num_images_to_display:
    print("Only", counter, "images found. Less than the specified number of images to display.")

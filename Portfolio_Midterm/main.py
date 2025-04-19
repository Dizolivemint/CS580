import PIL.Image
import PIL.ImageDraw
import os
import face_recognition
import glob
import numpy as np

def find_first_image(directory):
    """
    Find the first JPG or PNG image in the specified directory.
    
    Args:
        directory (str): Directory to search in
    
    Returns:
        str: Path to the first image found, or None if no images found
    """
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return None
        
    image_list = glob.glob(os.path.join(directory, '*.[jp][pn]g'))  # Matches .jpg and .png
    
    if len(image_list) == 0:
        print(f"No images found in directory: {directory}")
        return None
        
    return image_list[0]

def load_image_with_warning(image_path):
    """Load an image file with error handling."""
    try:
        image = face_recognition.load_image_file(image_path)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def detect_faces(image_path):
    """
    Detect faces in an image and return the processed image with rectangles.
    
    Args:
        image_path (str): Path to the image
    
    Returns:
        tuple: (PIL Image with rectangles, list of face locations)
    """
    # Load the image
    image = load_image_with_warning(image_path)
    if image is None:
        return None, []
    
    # Find all faces in the image
    face_locations = face_recognition.face_locations(image)
    number_of_faces = len(face_locations)
    print(f"Found {number_of_faces} face(s) in this picture.")
    
    # Create PIL image for drawing
    pil_image = PIL.Image.fromarray(image)
    draw_handle = PIL.ImageDraw.Draw(pil_image)
    
    # Draw rectangles around faces
    for face_location in face_locations:
        top, right, bottom, left = face_location
        print(f"A face is located at pixel location Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")
        draw_handle.rectangle([left, top, right, bottom], outline="red", width=3)
    
    return pil_image, face_locations

def compare_faces(individual_image_path, group_image_path):
    """
    Determine if an individual face is present in a group of faces.
    
    Args:
        individual_image_path (str): Path to the image containing the individual face
        group_image_path (str): Path to the image containing a group of faces
    
    Returns:
        tuple: (bool, float, tuple) - (is_match, distance, face_location)
    """
    # Load the images
    individual_image = load_image_with_warning(individual_image_path)
    group_image = load_image_with_warning(group_image_path)
    
    if individual_image is None or group_image is None:
        return False, None, None
    
    # Get face encodings for the individual
    individual_face_locations = face_recognition.face_locations(individual_image)
    if len(individual_face_locations) == 0:
        print("No face found in the individual image.")
        return False, None, None
    
    individual_face_encoding = face_recognition.face_encodings(individual_image, individual_face_locations)[0]
    
    # Get face encodings for the group
    group_face_locations = face_recognition.face_locations(group_image)
    if len(group_face_locations) == 0:
        print("No faces found in the group image.")
        return False, None, None
    
    group_face_encodings = face_recognition.face_encodings(group_image, group_face_locations)
    
    print(f"Found {len(group_face_locations)} faces in the group image.")
    
    # Compare the individual face to each face in the group
    face_distances = face_recognition.face_distance(group_face_encodings, individual_face_encoding)
    
    # Find the closest match
    min_distance_idx = np.argmin(face_distances)
    min_distance = face_distances[min_distance_idx]
    
    # A threshold to determine if faces match (lower is more strict)
    threshold = 0.6
    
    if min_distance <= threshold:
        matched_face_location = group_face_locations[min_distance_idx]
        return True, min_distance, matched_face_location
    else:
        return False, min_distance, None

def visualize_comparison(individual_path, group_path, output_dir, is_match, face_location=None):
    """
    Visualize the comparison results with highlighted matches.
    
    Args:
        individual_path (str): Path to the individual image
        group_path (str): Path to the group image
        is_match (bool): Whether a match was found
        face_location (tuple): Location of the matched face (top, right, bottom, left)
    """
    # Load and process individual image
    individual_pil, _ = detect_faces(individual_path)
    
    # Load and process group image
    group_image = load_image_with_warning(group_path)
    group_pil = PIL.Image.fromarray(group_image)
    draw = PIL.ImageDraw.Draw(group_pil)
    
    # Get all face locations in the group
    group_face_locations = face_recognition.face_locations(group_image)
    
    # Draw red rectangles around all faces
    for face_loc in group_face_locations:
        top, right, bottom, left = face_loc
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
    
    # If a match was found, highlight that face in green
    if is_match and face_location is not None:
        top, right, bottom, left = face_location
        draw.rectangle([left, top, right, bottom], outline="green", width=5)
    
    # Save the images
    individual_output = os.path.join(os.path.dirname(output_dir), 'individual_processed.jpg')
    group_output = os.path.join(os.path.dirname(output_dir), 'group_processed.jpg')
    
    individual_pil.save(individual_output)
    group_pil.save(group_output)
    
    # Display the images
    individual_pil.show()
    group_pil.show()
    
    print("Processed images saved as:")
    print(individual_output)
    print(group_output)

def main():
    """Main function to run the face recognition system."""
    print("Face Recognition System")
    print("1. Detect faces in a single image")
    print("2. Compare an individual face to a group")
    choice = input("Enter your choice (1/2): ")
    
    output_dir = os.path.join(os.path.abspath('.'), 'output')
    source_dir = os.path.join(os.path.abspath('.'), 'source')
    source_image_path = find_first_image(source_dir)
    individual_dir = os.path.join(os.path.abspath('.'), 'individual')
    group_dir = os.path.join(os.path.abspath('.'), 'group')
    
    if choice == '1':
        if not source_image_path:
            return
        
        print(f"Using image: {source_image_path}")
        
        pil_image, _ = detect_faces(source_image_path)
        if pil_image:
            output_image_path = os.path.join(output_dir, 'faces_processed.jpg')
            pil_image.save(output_image_path)
            pil_image.show()
            print(f"Processed image saved as {output_image_path}")
    
    elif choice == '2':
        # Ensure directories exist
        for dir_path in [individual_dir, group_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Created directory: {dir_path}")
                
        individual_path = find_first_image(individual_dir)
        group_path = find_first_image(group_dir)
        
        if not individual_path:
            print("Please place an individual image in the 'individual' directory.")
            return
        
        if not group_path:
            print("Please place a group image in the 'group' directory.")
            return
        
        print(f"Using individual image: {individual_path}")
        print(f"Using group image: {group_path}")
        
        is_match, distance, face_location = compare_faces(individual_path, group_path)
        
        if distance is not None:
            if is_match:
                print(f"Match found! Distance: {distance:.4f}")
            else:
                print(f"No match found. Minimum distance: {distance:.4f}")
            
            visualize_comparison(individual_path, group_path, output_dir, is_match, face_location)
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
from PIL import Image, ImageDraw

def generate_image_with_black_box(width, height, box_width, box_height, output_file):
    """
    Generates a PNG image of specified width and height with a black box centered.

    :param width: Width of the image
    :param height: Height of the image
    :param box_width: Width of the black box
    :param box_height: Height of the black box
    :param output_file: Path to save the output PNG file
    """
    # Create a white background image
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Calculate coordinates of the black box
    top_left_x = (width - box_width) // 2
    top_left_y = (height - box_height) // 2
    bottom_right_x = top_left_x + box_width
    bottom_right_y = top_left_y + box_height

    # Draw the black box
    draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], fill="black")

    # Save the image
    image.save(output_file, "PNG")
    print(f"Image saved as {output_file}")

# Example usage
if __name__ == "__main__":
    generate_image_with_black_box(
        width=500,       # Image width
        height=400,      # Image height
        box_width=200,   # Black box width
        box_height=100,  # Black box height
        output_file="square.png"  # Output file name
    )

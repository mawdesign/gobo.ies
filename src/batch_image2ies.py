import os
import argparse
import sys
import glob

# Import image2ies.py.  We assume it's in the same directory
# or that the user has added the current directory to their Python path.
import image2ies

_IMAGE_FILETYPES_ = (".png", ".jpg", ".jpeg")


def process_images_in_directory(directory, beam_angle, max_candela, **kwargs):
    """
    Processes all images in a directory (non-recursively) using image2ies.py
    to generate IES files.

    Args:
        directory (str): Path to the directory containing the images.
        beam_angle (float): Beam angle for the IES generation.
        max_candela (float): Maximum candela value for the IES generation.
        **kwargs: other values from command line to be passed through to ies generation function
    """
    # Ensure the directory path is valid
    if os.path.isdir(directory):
        directory = os.path.join(directory, "*")
    elif "*" in directory or "?" not in directory:
        print(f"Error: Directory not found: {directory}")
        return

    # Iterate through all files in the directory
    for file in glob.glob(directory):
        # Construct the full file path
        filename = os.path.basename(file)
        filepath = os.path.dirname(file)

        # Check if it's a file and an image

        if (
            os.path.isfile(file)
            and os.path.splitext(filename)[1].lower() in _IMAGE_FILETYPES_
        ):
            # Construct the output IES file path
            ies_filename = os.path.splitext(filename)[0] + ".ies"
            ies_filepath = os.path.join(filepath, ies_filename)

            print(f"Processing image: {file}")
            print(f"      Output IES: {ies_filepath}")

            try:
                # Call the image_to_ies function directly
                image2ies.image_to_ies(
                    file,
                    beam_angle,
                    max_candela,
                    **kwargs,
                )
            except Exception as e:
                print(f"  Error generating IES file for {filename}:")
                print(f"  Exception: {e}")
        elif not (os.path.isfile(file) or os.path.isdir(file)):
            print(f"  File not found: {filename}")


if __name__ == "__main__":
    """
    Main function to parse command line arguments and call the processing function.
    """
    args = vars(image2ies.get_command_line(wildcards=True))
    kwargs = {
        k: v
        for (k, v) in args.items()
        if k not in ("directory", "beam_angle", "max_candela")
    }

    # Call the function to process the images
    process_images_in_directory(
        args["directory"],
        args["beam_angle"],
        args["max_candela"],
        **kwargs,
    )

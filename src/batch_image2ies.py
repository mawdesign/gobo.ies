import os
import argparse
import sys

def process_images_in_directory(directory, beam_angle, max_candela, theta_step, phi_step):
    """
    Processes all images in a directory (non-recursively) using image2ies.py
    to generate IES files.

    Args:
        directory (str): Path to the directory containing the images.
        beam_angle (float): Beam angle for the IES generation.
        max_candela (float): Maximum candela value for the IES generation.
        theta_step (float): Theta step for IES generation
        phi_step (float): Phi step for IES generation
    """
    # Ensure the directory path is valid
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        return

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Construct the full file path
        filepath = os.path.join(directory, filename)

        # Check if it's a file and an image (you might want to expand the image type check)
        if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Construct the output IES file path
            ies_filename = os.path.splitext(filename)[0] + ".ies"
            ies_filepath = os.path.join(directory, ies_filename)

            print(f"Processing image: {filename}")
            print(f"      Output IES: {ies_filepath}")

            try:
                # Import image2ies.py.  We assume it's in the same directory
                # or that the user has added the current directory to their Python path.
                import image2ies
                # Call the image_to_ies function directly
                image2ies.image_to_ies(
                    filepath,
                    beam_angle,
                    max_candela,
                    ies_filepath,
                    theta_step,
                    phi_step
                )

            except Exception as e:
                print(f"  Error generating IES file for {filename}:")
                print(f"  Exception: {e}")

def main():
    """
    Main function to parse command line arguments and call the processing function.
    """
    parser = argparse.ArgumentParser(description="Generate IES files from images in a directory.")
    parser.add_argument("directory", help="Path to the directory containing the images.")
    parser.add_argument("beam_angle", type=float, help="Beam angle for the IES generation (degrees).")
    parser.add_argument("max_candela", type=float, help="Maximum candela value for the IES generation.")
    parser.add_argument("--theta_step", type=float, default=1.0, help="Step size for vertical angles (theta) in degrees. Defaults to 1.0.")
    parser.add_argument("--phi_step", type=float, default=5.0, help="Step size for horizontal angles (phi) in degrees. Defaults to 5.0.")

    args = parser.parse_args()

    # Call the function to process the images
    process_images_in_directory(args.directory, args.beam_angle, args.max_candela, args.theta_step, args.phi_step)

if __name__ == "__main__":
    main()


import numpy as np
from PIL import Image, ImageOps
import polarTransform  # https://github.com/addisonElliott/polarTransform
import io
import datetime
import sys
import textwrap


def generate_ies_string(header_data, lamp_data, v_angles, h_angles, candelas):
    """
    Generates an IES file string from the provided photometric data.

    Args:
        lamp_data (dict): Dictionary containing lamp metadata.
        v_angles (list): List of vertical angles (theta) in degrees.
        h_angles (list): List of horizontal angles (phi) in degrees.
        candelas (list of list): 2D list of candela values.
        is_absolute (bool, optional): Flag indicating if candelas are absolute.
            Defaults to True.

    Returns:
        str: The IES file content as a string.
    """
    # IES file header
    ies_header = "IES:LM-63-2019\n"
    if "TEST" not in header_data:
        header_data["TEST"] = f" "
    if "TESTLAB" not in header_data:
        header_data["TESTLAB"] = "Image2ies"
    if "MANUFAC" not in header_data:
        header_data["MANUFAC"] = "Generic"
    if "ISSUEDATE" not in header_data:
        header_data["ISSUEDATE"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "FILEGENINFO" not in header_data:
        header_data["FILEGENINFO"] = f"Generated from {image_file}"
    ies_header += "\n".join(
        [f"[{k}] {v}".replace("\n", "\n[MORE] ") for (k, v) in header_data.items()]
    )
    ies_header += "\nTILT=NONE\n"

    # Lamp data string
    lamp_data["v_angle_count"] = len(v_angles)
    lamp_data["h_angle_count"] = len(h_angles) + 1
    lamp_data_str = "{lamp_count} {lumens} {candela_multiplier} {v_angle_count} {h_angle_count} {photometric_type} {units_type} {lamp_width} {lamp_length} {lamp_height}\n".format(
        **lamp_data
    )
    lamp_data_str += "{ballast_factor} {file_gen_type} {input_watts}\n".format(
        **lamp_data
    )

    # Angle and candela data
    v_angle_str = (
        "\n".join(
            textwrap.wrap(
                " ".join(f"{v:.1f}".rstrip("0").rstrip(".") for v in v_angles), 256
            )
        )
        + "\n"
    )
    h_angle_str = (
        "\n".join(
            textwrap.wrap(
                " ".join(f"{h:.1f}".rstrip("0").rstrip(".") for h in h_angles) + " 360",
                256,
            )
        )
        + "\n"
    )

    candela_str = ""
    for row in candelas:
        candela_str += (
            "\n".join(
                textwrap.wrap(
                    " ".join(f"{c:.1f}".rstrip("0").rstrip(".") for c in row), 256
                )
            )
            + "\n"
        )
    candela_str += (
        "\n".join(
            textwrap.wrap(
                " ".join(f"{c:.1f}".rstrip("0").rstrip(".") for c in candelas[0]), 256
            )
        )
        + "\n"
    )

    # Combine all parts
    ies_string = ies_header + lamp_data_str + v_angle_str + h_angle_str + candela_str
    lnum = 0
    for ln in ies_string.splitlines():
        lnum += 1
        if len(ln) > 256:
            print(f"Line {lnum} exceeds 256 characters ({len(ln)} chars)")
    return ies_string


def image_to_ies(
    image_path,
    beam_angle,
    max_candela,
    filename="output.ies",
    theta_step=5.0,
    phi_step=5.0,
    sample_type=3,
):
    """
    Generates an IES file from an image, using the image's grayscale values
    to determine the luminous intensity distribution.

    Args:
        image_path (str): Path to the input image file (e.g., JPEG, PNG).
        beam_angle_deg (float): The beam angle of the light source in degrees.
        max_candela (float): The maximum luminous intensity (candela) of the
                           light source.
        filename (str, optional): The name of the output IES file.
            Defaults to "output.ies".
        theta_step (float, optional): Step size for vertical angles (theta) in degrees.
            Defaults to 1.0 degree.
        phi_step (float, optional): Step size for horizontal angles (phi) in degrees.
            Defaults to 5.0 degrees.
        sample_type (int, optional): The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
            The following orders have special names:
            0 - nearest neighbor
            1 - bilinear
            3 - bicubic
    """
    try:
        # Open the image using PIL (Pillow)
        img = (
            Image.open(image_path).convert("L").rotate(90)
        )  # Convert to grayscale and rotate
        img = ImageOps.mirror(img)  # Mirror
        pixels = np.array(img)

        # Calculate the radius of the largest centered circle
        width, height = img.size
        radius = min(width, height) // 2
        beam_angle_deg = beam_angle / 2 + theta_step

        # Convert to a polar image
        polarImage, ptSettings = polarTransform.convertToPolarImage(
            pixels,
            finalRadius=radius,
            radiusSize=int(beam_angle_deg / theta_step),
            angleSize=int(360 / phi_step),
            order=sample_type,
        )

        # Handle the case where the maximum pixel value is 0.
        max_pixel_value = max(np.max(polarImage), 1)

        # Unique theta and phi values, with step
        unique_theta = np.arange(0, beam_angle_deg, theta_step)
        unique_phi = np.arange(0, 360, phi_step)

        # Ensure unique_phi is sorted for IES file compatibility
        unique_phi = np.sort(unique_phi)

        # Calculate luminous intensity
        intensity_data = polarImage * max_candela / max_pixel_value

        # Create the IES file
        # Metadata for the IES file
        lamp_data = {
            "lamp_count": 1,
            "lumens": -1,  # for absolute = -1, otherwise use max_candela,
            "candela_multiplier": 1,
            "photometric_type": 1,
            "units_type": 2,
            "lamp_width": -0.01,
            "lamp_length": -0.01,
            "lamp_height": 0.01,
            "ballast_factor": 1,
            "file_gen_type": 1.00010,
            "input_watts": 10,
        }

        header_data = {}

        ies_string = generate_ies_string(
            header_data, lamp_data, unique_theta, unique_phi, intensity_data
        )

        # Save the IES string to a file
        with open(filename, "w") as f:
            f.write(ies_string)

        print(f"IES file successfully generated: {filename}")

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(f"An error occurred: {e} in line {exc_tb.tb_lineno}")


if __name__ == "__main__":
    image_file = "hi.jpg"
    try:
        img = Image.open(image_file)
    except FileNotFoundError:
        img = Image.new("L", (100, 100), color=127)
        img.save(image_file)

    beam_angle = 60.0
    max_candela_value = 1000.0
    output_filename = "my_light.ies"
    theta_step_size = 1
    phi_step_size = 2.5

    image_to_ies(
        image_file,
        beam_angle,
        max_candela_value,
        output_filename,
        theta_step_size,
        phi_step_size,
    )

import numpy as np
from PIL import Image, ImageOps
import polarTransform  # https://github.com/addisonElliott/polarTransform
import argparse
import os
import datetime
import sys
import textwrap


def generate_ies_string(header_data, lamp_data, v_angles, h_angles, candelas, image_file = "image"):
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
    # Make IES file compliant with IES LM-63-2019 format spcification
    #
    # Add required headers if not supplied
    if "TEST" not in header_data:
        header_data["TEST"] = ""
    if "TESTLAB" not in header_data:
        header_data["TESTLAB"] = "Image2ies"
    if "MANUFAC" not in header_data:
        header_data["MANUFAC"] = "Generic Gobo"
    if "ISSUEDATE" not in header_data:
        header_data["ISSUEDATE"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "FILEGENINFO" not in header_data:
        header_data["FILEGENINFO"] = f"Generated from {image_file}"

    # Ensure vertical angles end at 90 or 180 degrees
    if v_angles[-1] not in (90, 180) and v_angles[-1] < 180:
        curr_angle = v_angles[-1]
        theta_step = v_angles[-1] - v_angles[-2]
        zero_col = np.zeros((len(candelas),1))
        while curr_angle not in (90, 180) and curr_angle < 180:
            curr_angle += theta_step
            candelas = np.hstack((candelas, zero_col))
            v_angles = np.append(v_angles, curr_angle)
    if v_angles[-1] > 180:
        if v_angles[-2] < 180:
            v_angles[-1] = 180
        else:
            raise Exception(f"vertical angles (theta values) exceed 180° (..., {v_angles[-3]}, {v_angles[-2]}, {v_angles[-1]})")

    # IES file header
    ies_header = "IES:LM-63-2019\n"
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
                " ".join(f"{v:.1f}".rstrip("0").rstrip(".") for v in v_angles), 255
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
            " \n".join(
                textwrap.wrap(
                    " ".join(f"{c:.1f}".rstrip("0").rstrip(".") for c in row), 255
                )
            )
            + "\n"
        )
    candela_str += (
        " \n".join(
            textwrap.wrap(
                " ".join(f"{c:.1f}".rstrip("0").rstrip(".") for c in candelas[0]), 255
            )
        )
        + "\n"
    )

    # Combine all parts
    ies_string = ies_header + lamp_data_str + v_angle_str + h_angle_str + candela_str

    # Check for lines that exceed recommended max of 256 characters
    lnum = 0
    long_lines = []
    for ln in ies_string.splitlines():
        lnum += 1
        if len(ln) > 256:
            long_lines.append((lnum,len(ln)))
    if long_lines:
        print(f"{len(long_lines)} line{'s' if len(long_lines) != 1 else ''} exceed{'s' if len(long_lines) == 1 else ''} 256 characters.")
        print(f"(e.g. line {long_lines[0][0]} is {long_lines[0][1]} characters)")
    return ies_string


def s_curve(x, median=0.5, slope=0.5, start=0, end=1, minimum=0, maximum=1):
    """
    Returns a value on an s-curve where:
      start   is the x axis curve start
      end     is the x axis curve end
      minimum is the y axis curve low point
      maximum is the y axis curve high point
      slope   is a value between 0 and 1 for the steepness of the curve
        0 gives a straight line at 45°
        1 gives a vertical line
    """

    # The curve alogrthym
    def f(x, n, c):
        match (x, n, c):
            case (x, n, c) if x <= 0:
                y = 0
            case (x, n, c) if x <= n:
                y = x**c / n ** (c - 1)
            case (x, n, c) if x <= 1:
                y = 1 - (1 - x) ** c / (1 - n) ** (c - 1)
            case _:
                y = 1
        return y

    # set parameters
    p = (median - start) / (end - start)
    h = maximum - minimum
    c = 2 / (1 - slope) - 1

    # find 'n' for f(p) = 0.5
    if c == 1:
        n = 0.5
    elif p > 0.5:
        n = (2 * p**c) ** (1 / (c - 1))
    else:
        n = 1 - (2 * (1 - p) ** c) ** (1 / (c - 1))

    # return array if array sent
    if isinstance(x, np.ndarray):
        y_list = [minimum + h * f((x1 - start) / (end - start), n, c) for x1 in x]
        y = np.fromiter(y_list, float)
    # ... or list if other collection sent
    elif isinstance(x, (list, tuple, set, dict)):
        y = [minimum + h * f((x1 - start) / (end - start), n, c) for x1 in x]
    else:
        y = f(minimum + h * (x1 - start) / (end - start), n, c)
    return y


def image_to_ies(
    image_path,
    beam_angle,
    max_candela,
    filename="output.ies",
    source_ies=False,
    theta_step=5.0,
    phi_step=5.0,
    sample_type=3,
    edge_fade=False,
    allow_spill=False,
):
    """
    Generates an IES file from an image, using the image's grayscale values
    to determine the luminous intensity distribution.

    Args:
        image_path (str): Path to the input image file (e.g., JPEG, PNG).
        beam_angle (float): The beam angle of the light source in degrees.
        max_candela (float): The maximum luminous intensity (candela) of the
                           light source.
        filename (str, optional): The name of the output IES file.
            Defaults to "output.ies".
        theta_step (float, optional): Step size for vertical angles (theta) in degrees.
            Defaults to 5.0 degrees.
        phi_step (float, optional): Step size for horizontal angles (phi) in degrees.
            Defaults to 5.0 degrees.
        sample_type (int, optional): The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
            The following orders have special names:
            0 - nearest neighbor
            1 - bilinear
            3 - bicubic
    """
    try:
        if source_ies:
            # Read source .ies file
            if not os.path.isfile(source_ies):
                raise FileNotFoundError(source_ies)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(image_path)
            from luxpy import iolidfiles as iolid
            LID = iolid.read_lamp_data(source_ies, normalize=None)
            # Overwrite values to fit with the source
            theta_step = LID['map']['thetas'][-1] / (len(LID['map']['thetas']) - 1)
            phi_step = LID['map']['phis'][-1] / (len(LID['map']['phis']) - 1)
            source_theta = LID['map']['thetas']
            source_phi = LID['map']['phis'][:-1]
            source_distribution = LID['map']['values'][:-1]
            edge_fade = False

        if image_path == "white":
            image_file = "user defined shape"
            img = Image.new("L", (100, 100), color=255)
            pixels = np.array(img)
            width, height = img.size
        elif os.path.isfile(image_path):
            image_file = os.path.basename(image_path)
            # Open the image using PIL (Pillow)
            with Image.open(image_path) as img:
                img = img.convert("L") #.rotate(90) # Convert to grayscale and rotate
                img = ImageOps.mirror(img)  # Mirror
                pixels = np.array(img)
                width, height = img.size
        else:
            raise FileNotFoundError(image_path)


        # Calculate the radius of the largest centered circle
        radius = min(width, height) // 2
        beam_angle_deg = beam_angle / 2 + theta_step
        if source_ies:
            # Ensure beam angle we are using is valid by finding closest theta angle
            beam_angle_deg = min(LID['map']['thetas'][::-1], key=lambda u:abs(u - beam_angle_deg))
        if edge_fade:
            match edge_fade:
                case [a, b, c, *_]: # at least 3 values
                    if c > 0.9:
                        c = 0.9
                    elif c < 0.0:
                        c = 0.0
                    edge_fade = [a, b, c]
                    beam_angle_deg += b
                case [a, b]:
                    edge_fade = [a, b, 0.5]
                    beam_angle_deg += b
                case [a]:
                    edge_fade = [a, a, 0.5]
                    beam_angle_deg += a
                case [] | True:
                    edge_fade = [theta_step * 2, theta_step * 2, 0.5]
                    beam_angle_deg += theta_step * 2
                case _:
                    edge_fade = False

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
        if source_ies:
            # Extract affected area from source, then modify with gobo pattern
            gobo_area = source_distribution[0:len(unique_phi), 0:len(unique_theta)] #int(beam_angle_deg/theta_step)]
            source_distribution[0:len(unique_phi), 0:len(unique_theta)] = gobo_area * (polarImage / max_pixel_value)
            # to add image to source .ies file use this:
            # source_distribution[0:len(unique_phi), 0:len(unique_theta)] = np.add(gobo_area, intensity_data)
            if not allow_spill:
                source_distribution[0:len(unique_phi), len(unique_theta):len(source_theta)] = 0.0
            intensity_data = source_distribution
            unique_theta = source_theta
            unique_phi = source_phi
        if edge_fade:
            fade_values = s_curve(
                unique_theta,
                start=beam_angle / 2 + edge_fade[0],
                end=beam_angle / 2 - edge_fade[1],
                median=beam_angle / 2,
                slope=edge_fade[2]
            )
            intensity_data = np.multiply(intensity_data, fade_values)

        # Create the IES file
        # Metadata for the IES file
        header_data = {}
        header_data["FILEGENINFO"] = f"Generated from {image_file}"

        lamp_data = {
            "lamp_count": 1,
            "lumens": -1,  # for absolute = -1, otherwise use max_candela,
            "candela_multiplier": 1,
            "photometric_type": 1,
            "units_type": 2,
            "lamp_width": -0.01,
            "lamp_length": -0.01,
            "lamp_height": 0,
            "ballast_factor": 1,
            "file_gen_type": 1.00010,
            "input_watts": 10,
        }

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


def get_command_line():
    '''
    Get command line arguments
    '''
    parser = argparse.ArgumentParser(description="Generate IES files from an image.")
    parser.add_argument("image_file", metavar="<image.jpg>", help="Path to the image.")
    parser.add_argument("-o", "--output", metavar="<output.ies>", help="Output filename, defaults to same name as image with .ies extension.")
    parser.add_argument("-s", "--source", metavar="<source.ies>", help="Filename of an .ies file to use as the source distribution.")
    parser.add_argument("-a", "--allow_spill", action="store_true", help="If using <source.ies>, allow spill light from beyond beam angle.")
    parser.add_argument("-b", "--beam_angle", type=float, default=30, help="Beam angle for the IES generation (degrees).")
    parser.add_argument("-e", "--edge_fade", nargs="*", type=float, default=False, help="Fade edge optionally followed by up to 3 values to specify shape: [WIDTH_INSIDE_BEAM], [WIDTH_OUTSIDE_BEAM] (degrees), and [SLOPE] (ratio).")
    parser.add_argument("-c", "--max_candela", type=float, default=1000, help="Maximum candela value for the IES generation.")
    parser.add_argument("-t", "--theta_step", type=float, default=1.0, help="Step size for vertical angles (theta) in degrees. Defaults to %(default)s°.")
    parser.add_argument("-p", "--phi_step", type=float, default=2.5, help="Step size for horizontal angles (phi) in degrees. Defaults to %(default)s°.")
    args = parser.parse_args()

    # sanity and rule checks
    if args.image_file.lower() != "white" and not os.path.isfile(args.image_file):
        raise FileNotFoundError(image_path)
    if args.source:
        args.edge_fade = False
        if not os.path.isfile(args.source):
            raise FileNotFoundError(args.source)
        if not os.path.isfile(args.image_file):
            raise FileNotFoundError(image_path)

    return args

if __name__ == "__main__":
    """
    Main function to parse command line arguments and call the processing function.
    """
    args = get_command_line()

    # verify input image file
    try:
        img = Image.open(args.image_file)
        img.close()
        image_file = args.image_file
    except FileNotFoundError:
        if args.image_file.lower() == "white":
            image_file = "white"
        else:
            raise

    # verify output .ies file
    if args.output is None:
        output_filename = os.path.splitext(os.path.basename(args.image_file))[0] + ".ies"
    elif os.path.splitext(args.output)[1].lower() != ".ies":
        output_filename = os.path.splitext(args.output)[0] + ".ies"
    else:
        output_filename = str(args.output)

    match args.edge_fade:
        case [a, b, c, *_]: # at least 3 values
            edge_fade = [a, b, c]
        case [a, b]:
            edge_fade = [a, b, 0.5]
        case [a]:
            edge_fade = [a, a, 0.5]
        case [] | True:
            edge_fade = [args.theta_step, args.theta_step, 0.5]
        case _:
            edge_fade = False

    # Call the function to process the image
    image_to_ies(
        image_file,
        args.beam_angle,
        args.max_candela,
        filename=output_filename,
        source_ies=args.source,
        theta_step=args.theta_step,
        phi_step=args.phi_step,
        edge_fade=edge_fade,
        allow_spill=args.allow_spill
    )

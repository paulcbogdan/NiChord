from typing import Union


def combine_imgs(fp_glass: str, fp_chord: str, fp_combined: str,
                 chord_mult: Union[float, int] = 1.05) -> None:
    """
    Combine the glass brain connectome and chord diagram into a single picture.
        It works by loading the images with PIL then cropping and pasting
        them into a single image.

    :param fp_glass: filepath of the glass brain connectome
    :param fp_chord: filepath of the chord diagram
    :param fp_combined: filepath of the resulting image that combines the
        glass brain and chord diagrams
    :param chord_mult: specifies the size of the chord diagram relative to
        the glass brain. Probably shouldn't change.
    """
    from PIL import Image

    image_glass = Image.open(fp_glass)
    image_chord = Image.open(fp_chord)

    w_glass, h_glass = image_glass.size
    w_chord, h_chord = image_chord.size

    w_chord_new = int(h_glass / h_chord * w_chord * chord_mult)
    h_chord_new = int(h_glass * chord_mult)
    image_chord = image_chord.resize((w_chord_new, h_chord_new))
    image_glass = image_glass.crop((w_glass * .303, 0, w_glass,
                                    h_glass))  # drop the rightmost (axial)
                                        # brain. The chord diagram will go here

    new_im = Image.new('RGB', (image_glass.size[0] + w_chord_new, int(
        (h_chord_new - (chord_mult - 1) / 2 * h_chord_new))),
                       color='white')
    new_im.paste(image_chord, (image_glass.size[0], 0))
    new_im.paste(image_glass, (0, int((chord_mult - 1) / 2 * h_chord_new)))

    new_fp = fp_combined
    new_im.save(new_fp)

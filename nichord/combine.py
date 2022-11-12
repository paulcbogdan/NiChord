import os
from typing import Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

from nichord import plot_chord, plot_glassbrain


def combine_imgs(fp_glass: str, fp_chord: str, fp_combined: str,
                 chord_mult: Union[float, int] = 1.05,
                 title: Union[None, str] = None,
                 ) -> None:
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
    image_glass = image_glass.crop((int(w_glass * .303), 0, w_glass,
                                    h_glass))  # drop the rightmost (axial)
                                        # brain. The chord diagram will go here

    new_im = Image.new('RGB', (image_glass.size[0] + w_chord_new, int(
        (h_chord_new - (chord_mult - 1) / 2 * h_chord_new))),
                       color='white')
    new_im.paste(image_chord, (image_glass.size[0], 0))
    new_im.paste(image_glass, (0, int((chord_mult - 1) / 2 * h_chord_new)))

    new_fp = fp_combined

    if title is not None: new_im = add_title(title, new_im)
    new_im.save(new_fp)


def add_title(title: str, im, fontsize=82):
    font = ImageFont.truetype("arial.ttf", fontsize)
    im_w_title = Image.new(im.mode, (im.size[0], im.size[1] + 100), 'white')
    W, H = im.size[0], 100
    im_w_title.paste(im, (0, 100))
    draw = ImageDraw.Draw(im_w_title)
    _, _, w, h = draw.textbbox((0, 0), title, font=font)
    draw.text(((W-w)/2, (H-h)/2), title, font=font, fill='black', align='right')
    return im_w_title


def plot_and_combine(dir_out: str,
                     fn: str,
                     idx_to_label: dict,
                     edges: Union[list, np.ndarray],
                     edge_weights: Union[list, np.ndarray, None] = None,
                     coords: Union[list, np.ndarray] = None,
                     cmap: Union[None, str, matplotlib.colors.Colormap] = None,
                     network_order: Union[list, None] = None,
                     network_colors: Union[dict, None] = None,
                     chord_kwargs: Union[None, dict] = None,
                     glass_kwargs: Union[None, dict] = None,
                     ) -> None:
    name, file_ext = os.path.splitext(fn)
    if file_ext == '':
        file_ext = '.png'

    dir_chord = os.path.join(dir_out, 'chord')
    if not os.path.isdir(dir_chord): os.mkdir(dir_chord)
    dir_glass = os.path.join(dir_out, 'glass')
    if not os.path.isdir(dir_glass): os.mkdir(dir_glass)

    fp_chord = os.path.join(dir_chord, f'{name}_chord{file_ext}')
    fp_glass = os.path.join(dir_glass, f'{name}_glass{file_ext}')
    fp_combined = os.path.join(dir_out, f'{name}{file_ext}')

    if chord_kwargs is None: chord_kwargs = {}
    if 'network_order' not in chord_kwargs:
        chord_kwargs['network_order'] = network_order
    if 'network_colors' not in chord_kwargs:
        chord_kwargs['network_colors'] = network_colors
    if 'cmap' not in chord_kwargs:
        chord_kwargs['cmap'] = cmap

    plot_chord(idx_to_label, edges, edge_weights=edge_weights,
               coords=coords, fp_chord=fp_chord,
               **chord_kwargs)

    if glass_kwargs is None: glass_kwargs = {}
    if 'network_order' not in glass_kwargs:
        glass_kwargs['network_order'] = network_order
    if 'network_colors' not in glass_kwargs:
        glass_kwargs['network_colors'] = network_colors
    if 'cmap' not in glass_kwargs:
        glass_kwargs['cmap'] = cmap

    plot_glassbrain(idx_to_label, edges, edge_weights, fp_glass,
                    coords, **glass_kwargs)

    combine_imgs(fp_glass, fp_chord, fp_combined)

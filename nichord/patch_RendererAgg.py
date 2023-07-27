import matplotlib as mpl
from  matplotlib.backends import backend_agg

class NeuoChordRenderAgg(backend_agg.RendererAgg):
    def __init__(self, width, height, dpi):
        super().__init__(width, height, dpi)

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # There may be a bug in matplotlib.backend_agg.RenderAgg, which makes
        #   rotated text not look ideal when plotting character by character.
        #   This issue is seemingly fixed by commenting out the code below.
        #   I submitted a bug report to matplotlib, along with this solution.
        #   It has not been accepted yet, so for now I am "monkey patching" the
        #   function with this.

        if ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)


        flags = backend_agg.get_hinting_flag()

        # matplotlib changed the name of one of its functions in version 3.6
        matplotlib_v36 = callable(getattr(self, '_prepare_font', None))
        if matplotlib_v36:
            font = self._prepare_font(prop)
        else:
            font = self._get_agg_font(prop)

        if font is None:
            return None
        # [comment preserved from the original version of this function]
        # We pass '0' for angle here, since it will be rotated (in raster
        # space) in the following call to draw_text_image).

        font.set_text(s, 0, flags=flags)
        font.draw_glyphs_to_bitmap(antialiased=mpl.rcParams['text.antialiased'])

        # [comment preserved from the original version of this function]
        # d = font.get_descent() / 64.0
        # # "The descent needs to be adjusted for the angle."
        # xo, yo = font.get_bitmap_offset()
        # xo /= 64.0
        # yo /= 64.0
        # xd = d * sin(radians(angle))
        # yd = d * cos(radians(angle))
        # x = round(x  + xd)
        # y = round(y  + yd)
        self._renderer.draw_text_image(font, x, y + 1, angle, gc)


def do_monkey_patch() -> None:
    backend_agg.RendererAgg = NeuoChordRenderAgg



if __name__ == '__main__':
    do_monkey_patch()
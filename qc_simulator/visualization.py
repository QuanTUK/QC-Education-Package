from simulator import simulator
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from io import BytesIO
import base64
import numpy as np

# TODO: Docstrings
# TODO: Updating the Plot inside of draw instead of creating new figure
# TODO: (Animating DCN with camera view)

class Visualization:
    """Superclass for all visualizations
    """
    def __init__(self, simulator: simulator):
        self._sim = simulator
        self._fig = None
        self._colors = None
        self._widths = None


    def export_base64(self, formatStr='png'):
        return base64.b64encode(self._export_buffer(formatStr)).decode("ascii")
        

    def export_png(self, fname:str):
        self._export(fname)

    def export_pdf(self, fname:str):
        self._export(fname, 'svg')
    
    def export_svg(self, fname:str):
        self._export(fname, 'pdf')


    def _export_buffer(self, formatStr):
        buf = BytesIO()
        self._export(buf, formatStr)
        return buf.getbuffer()


    def _export(self, target, formatStr='png') :
        self.draw()  # always redraw
        self._fig.savefig(target, format=formatStr, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)


    def show(self):
        """Show fig
        """ 
        # NOTE: Assert Gui backend
        # TODO: More than one show possible
        self.draw()  # always redraw
        plt.show()


    def draw(self):
        pass


class CircleNotation(Visualization):
    """Circle Notation
    """

    def __init__(self, simulator: simulator, cols=None):
        """_summary_

        Args:
            simulator (qc_simulator.simulator): _description_
            cols (_type_, optional): _description_. Defaults to None.
        """
        self._sim = simulator
        self._colors = {'edge': 'black', 'fill': '#77b6ba', 'phase': 'black'}
        self._widths = {'edge': 1, 'phase': 1}
        self._circleDist = 3

        self._fig = None
        self._cols = cols if cols != None else 2**self._sim._n      
        
        
    def draw(self):
        bits = 2**self._sim._n
        x_max = self._circleDist * self._cols
        y_max = self._circleDist * bits/self._cols
        xpos = self._circleDist/2
        ypos = y_max - self._circleDist/2

        self._fig = plt.figure(layout='compressed', dpi=300)
        ax = self._fig.gca()
       
        val = np.abs(self._sim._register)
        phi = -np.angle(self._sim._register, deg=False).flatten()
        lx, ly = np.sin(phi), np.cos(phi)

        ax.set_xlim([0, x_max])
        ax.set_ylim([0, y_max])
        ax.set_axis_off()
        ax.set_aspect('equal')

        for i in range(2**self._sim._n):
            fill = mpatches.Circle((xpos, ypos), radius=val[i], color=self._colors['fill'], edgecolor=None)
            ring = mpatches.Circle((xpos, ypos), radius=1, fill=False, edgecolor=self._colors['edge'], linewidth=self._widths['edge'])
            phase = mlines.Line2D([xpos, xpos+lx[i]], [ypos, ypos+ly[i]], color=self._colors['phase'], linewidth=self._widths['phase'])
            ax.add_artist(fill)
            ax.add_artist(ring)
            ax.add_artist(phase)
            label = np.binary_repr(i, width=self._sim._n) # width is deprecated since numpy 1.12.0
            ax.text(xpos, ypos - 1.35, f'|{label:s}>', horizontalalignment='center', verticalalignment='center')
            # NOTE text vs TextPath: text can easily be centered, textpath size is fixed when zooming
            # tp = TextPath((xpos-0.2*len(label), ypos - 1.35), f'|{label:s}>', size=0.4)
            # ax.add_patch(PathPatch(tp, color="black"))
            # NOTE Area/prop as text inside circle?
            xpos += self._circleDist
            if (i+1) % self._cols == 0:
                xpos = self._circleDist / 2
                ypos -= self._circleDist


# TODO: Textgröße fest -> TextPatch
class DimensionalCircleNotation(Visualization):
    """Circle Notation
    """

    def __init__(self, simulator: simulator):
        """_summary_

        Args:
            simulator (qc_simulator.simulator): _description_
        """
        self._sim = simulator

        # Style of circles
        self._colors = {'edge': 'black', 'bg': 'white', 'fill': '#77b6baff', 'phase': 'black', 'cube': '#5a5a5a'}
        self._widths = {'edge': .5, 'phase': .5, 'cube': .5, 'textsize': 10, 'textwidth': .1}
        self._arrowStyle = {'width':.03, 'head_width':.3, 'head_length':.5, 'edgecolor':None, 'facecolor':'black'}

        # Placement
        self._c = 5  # circle distance
        self._o = self._c / 2  # offset for 3rd dim qubits

        self._coords = np.array([   [0, 1],                 # |000>
                                    [1, 1],                 # |001>
                                    [0, 0],                 # |010>
                                    [1, 0],                 # |011>
                                    [0, 0],                 # |100>
                                    [0, 0],                 # |101>
                                    [0, 0],                 # |110>
                                    [0, 0]], dtype=float)   # |111>
        # Set distance
        self._coords *= self._c   
        # offset 3rd dim qubits 
        self._coords[4:] = self._coords[:4] + self._o
        # center around origin
        # self._coords -= self._c/2
        
        self._fig = None
        self._ax = None
        self._val, self._phi = None, None
        self._lx, self._ly = None, None
         
        
    def draw(self):
        """_summary_
        """
        self._fig = plt.figure(layout='compressed', dpi=300)
        self._ax = self._fig.gca()
        self._ax.set_axis_off()
        self._ax.set_aspect('equal')

        self._val = np.abs(self._sim._register)
        self._phi = -np.angle(self._sim._register, deg=False).flatten()
        self._lx, self._ly = np.sin(self._phi), np.cos(self._phi)

        bits = 2**self._sim._n
        if bits > 4:
            self._drawLine([0,4,5])
            self._drawLine([1,5,7,3])
            self._drawDottedLine([2,6,7])
            self._drawDottedLine([4,6])
            self._drawCircle(7)
            self._drawCircle(6)
            self._drawCircle(5)
            self._drawCircle(4)
        if bits > 2:
            self._drawLine([0,2,3,1])
            self._drawCircle(3)
            self._drawCircle(2)
        self._drawLine([0,1])
        self._drawCircle(1)
        self._drawCircle(0)

        # Basisvectors
        # NOTE: Array/Liste für Positionen -> kwargs
        if self._sim._n == 1:
            self._drawArrows(-1, self._c + 2)  
            self._ax.set_xlim([-1.2, 6.2])
            self._ax.set_ylim([3.5, 7.5])
        elif self._sim._n == 2:
            self._drawArrows(-2.5, self._c + 2.5)  
            self._ax.set_xlim([-4, 6.2])
            self._ax.set_ylim([-2,8])
        elif self._sim._n == 3:
            self._ax.set_xlim([-5, 8.7])
            self._ax.set_ylim([-2, 10.35])
            self._drawArrows(-self._c+self._o*2/3, self._c + 2.5)  


    def _drawArrows(self, x0, y0):
        alen = self._c*2/3 # NOTE: -> kwargs
        if self._sim._n > 2:
            di = alen / np.sqrt(2)
            self._ax.text(x0+di/2-.15, y0+di/2+.15, 'Qubit #3', size=self._widths['textsize'], usetex=False, horizontalalignment='right', verticalalignment='center')
            self._ax.arrow(x0, y0, di, di, **self._arrowStyle)
        if self._sim._n > 1:
            self._ax.text(x0-.3, y0-alen/2, 'Qubit #2', size=self._widths['textsize'], usetex=False, horizontalalignment='right', verticalalignment='center')
            self._ax.arrow(x0, y0, 0, -alen, **self._arrowStyle)
        self._ax.text(x0+alen/2, y0+.3, 'Qubit #1', size=self._widths['textsize'], usetex=False, horizontalalignment='center', verticalalignment='center')    
        self._ax.arrow(x0, y0, alen, 0, **self._arrowStyle)


    def _drawDottedLine(self, points:list):
        self._ax.plot(self._coords[points,0], self._coords[points,1], color=self._colors['cube'], linewidth=self._widths['cube'], linestyle='dotted', zorder=1)


    def _drawLine(self, points:list):
        self._ax.plot(self._coords[points,0], self._coords[points,1], color=self._colors['cube'], linewidth=self._widths['cube'], linestyle='solid', zorder=1)
            

    def _drawCircle(self, index:int):
        xpos, ypos = self._coords[index]
        # White bg circle area of unit circle
        bg = mpatches.Circle((xpos, ypos), radius=1, color=self._colors['bg'], edgecolor=None)
        self._ax.add_artist(bg)
        # Fill area of unit circle
        if self._val[index] > 0:
            fill = mpatches.Circle((xpos, ypos), radius=self._val[index], color=self._colors['fill'], edgecolor=None)
            self._ax.add_artist(fill)
        # Black margin for circles
        ring = mpatches.Circle((xpos, ypos), radius=1, fill=False, edgecolor=self._colors['edge'], linewidth=self._widths['edge'])
        self._ax.add_artist(ring)
        # Indicator for phase
        if self._val[index] > 0:
            phase = mlines.Line2D([xpos, xpos+self._lx[index]], [ypos, ypos+self._ly[index]], color=self._colors['phase'], linewidth=self._widths['phase'])
            self._ax.add_artist(phase)
        # Add label to circle
        label = np.binary_repr(index, width=self._sim._n) # width is deprecated since numpy 1.12.0
        # print(index, label)
        if self._sim._n == 3:
            off = -1.3 if int(label[1]) else 1.3
        elif self._sim._n == 2:
            off = -1.3 if int(label[0]) else 1.3
        else: 
            off = -1.3
        self._ax.text(xpos, ypos + off, fr'$|{label:s}\rangle$', size=self._widths['textsize'], usetex=False, horizontalalignment='center', verticalalignment='center')
            
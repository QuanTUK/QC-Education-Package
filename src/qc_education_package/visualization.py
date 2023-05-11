#----------------------------------------------------------------------------
    # Created By: Nikolas Longen, nlongen@rptu.de
    # Reviewed By: Maximilian Kiefer-Emmanouilidis, maximilian.kiefer@rptu.de
    # Created: March 2023
    # Project: DCN QuanTUK
#----------------------------------------------------------------------------
from .simulator import Simulator
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from io import BytesIO
from base64 import b64encode
import numpy as np


class Visualization:
    """Superclass for all visualizations of quantum computer states. 
    This way all visualizations inherit export methods. 
    Alle subclasses must implement/overwrite a draw method and should also overwrite the __init__ method.
    """
    def __init__(self, simulator, parse_math=True):
        """Constructor for Visualization superclass.

        Args:
            simulator (qc_simulator.simulator): Simulator object to be visualized.
        """
        self._sim = simulator
        self._bitOrder = simulator._bitOrder
        self._fig = None
        self._colors = None
        self._widths = None
        self._lastState = None
        mpl.rcParams['text.usetex'] = False
        mpl.rcParams['text.parse_math'] = parse_math
      

    def export_png(self, fname:str, transparent=True):
        """Export the current visualization as PNG image to given path.

        Args:
            fname (str): fname or path to export image to.
        """
        self._export(fname, 'png', transparent)


    def export_pdf(self, fname:str, transparent=True):
        """Export the current visualization as PDF file to given path.

        Args:
            fname (str): fname or path to export file to.
        """
        self._export(fname, 'pdf', transparent)
    

    def export_svg(self, fname:str, transparent=True):
        """Export the current visualization as SVG image to given path.

        Args:
            fname (str): fname or path to export image to.
        """
        mpl.rcParams['svg.fonttype'] = 'none'
        self._export(fname, 'svg', transparent)


    def export_base64(self, formatStr='png', transparent=True):
        """Export given format as base64 string. Mostly to handover images for flask website.

        Args:
            formatStr (str, optional): Format for image. Defaults to 'png'.

        Returns:
            str: base64 string representation of the generated image.
        """
        return b64encode(self._export_buffer(formatStr, transparent)).decode("ascii")


    def _export_buffer(self, formatStr, transparent=True):
        """Export current visualization in format into IO buffer. 

        Args:
            formatStr (str, optional): Format for image. Defaults to 'png'.

        Returns:
            BytesIO: returns view of buffer containing image using BytesIO.getbuffer()
        """
        buf = BytesIO()
        self._export(buf, formatStr, transparent)
        return buf.getbuffer()


    def _export(self, target, formatStr='png', transparent=True):
        """General export method to save current pyplot figure, so all all exports will share same form factor, res etc.  

        Args:
            target (plt.savefig compatible object): Target to save pyplot figure to.
            formatStr (str, optional): Format for image. Defaults to 'png'.
        """
        self._redraw()
        self._fig.savefig(target, format=formatStr, bbox_inches='tight', pad_inches=0, dpi=300, transparent=transparent)


    def show(self):
        """Method to show current figure using plt.show but making sure the visualization is always redrawn.
        """ 
        self._redraw()
        plt.show()


    def _redraw(self):
        """Checks if simulator state is changed and redraws the image if so.
        """
        if self._lastState != self._sim:
            self.draw()


    def draw(self):
        # TODO: Add style guide for draw method
        pass


class CircleNotation(Visualization):
    """A Visualization subclass for the well known Circle Notation representation.
    """

    def __init__(self, simulator):
        """Constructor for the Circle Notation representation.

        Args:
            simulator (qc_simulator.simulator): Simulator object to be visualized.
            cols (int, optional): Arrange Circle Notation into a set of columns. Defaults to None.
        """
        super().__init__(simulator)  # Execute constructor of superclass
        
        self._colors = {'edge': 'black', 'fill': '#77b6ba', 'phase': 'black'}
        self._widths = {'edge': 1, 'phase': 1}
        self._circleDist = 3
        self._fig = None
              
        
    def draw(self, cols=None):
        """Draw Circle Notation representation of current simulator state.
        """
        self._cols = cols if cols != None else 2**self._sim._n
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
            ax.text(xpos, ypos - 1.35, fr'$|{label:s}\rangle$', horizontalalignment='center', verticalalignment='center')
            # NOTE text vs TextPath: text can easily be centered, textpath size is fixed when zooming
            # tp = TextPath((xpos-0.2*len(label), ypos - 1.35), f'|{label:s}>', size=0.4)
            # ax.add_patch(PathPatch(tp, color="black"))
            xpos += self._circleDist
            if (i+1) % self._cols == 0:
                xpos = self._circleDist / 2
                ypos -= self._circleDist


class DimensionalCircleNotation(Visualization, ):
    """A Visualization subclass for the newly introduced Dimensional Circle Notation (DCN) representation.
    """

    def __init__(self, simulator):
        """Constructor for the Dimensional Circle Notation representation.
        This representation can be used for up to 3 qubits.
        

        Args:
            simulator (qc_simulator.simulator): Simulator object to be visualized.
            show_values (bool): Show magnitude and phase for each state, defaults to False.
        """
        super().__init__(simulator)  # Execute constructor of superclass
        assert(simulator._n <= 3)  # DCN is made for up to 3 qubits
        # TODO: 4-5 Qubits 

        # Style of circles
        self._colors = {'edge': 'black', 'bg': 'white', 'fill': '#77b6baff', 'phase': 'black', 'cube': '#8a8a8a'}
        self._widths = {'edge': .7, 'phase': .7, 'cube': .5, 'textsize': 10, 'textwidth': .1}
        self._textsize = {'register': 10, 'magphase':10, 'axislbl':10}
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

        self._axis_labels = np.arange(1,self._sim._n+1)[::-self._bitOrder]

        self._fig = None
        self._ax = None
        self._val, self._phi = None, None
        self._lx, self._ly = None, None
        self._showMagnPhase = False
         
        
    def showMagnPhase(self, show_values:bool):
        print(show_values)
        self._showMagnPhase = show_values

    
    def draw(self):
        """Draw Dimensional Circle Notation representation of current simulator state.
        """
        self._lastState = self._sim
        self._fig = plt.figure(layout='compressed')
        self._ax = self._fig.gca()
        self._ax.set_axis_off()
        self._ax.set_aspect('equal')

        self._val = np.abs(self._sim._register)
        self._phi = -np.angle(self._sim._register, deg=False).flatten()
        self._lx, self._ly = np.sin(self._phi), np.cos(self._phi)

        # limits for coordinate axis [[[xmin, xmax] [ymin, ymax]], [...]]
        limits = np.array(  [[[-1.2, 6.2], [3.5, 7.5]],    # 1 Qubit
                            [[-4, 6.2], [-2,8]],          # 2 Qubits
                            [[-5, 8.7],[-2, 10.35]]])      # 3 Qubits
        
        # Scale textsizes such that ratio circles to textsize constant
        # automatic relative to length of y axis
        # ylen = limits[:,1,1] - limits[:,1,0]
        # scale = ylen[2] / ylen
        
        # hard coded 
        scale = [2, 1.25, 1]
        
        factor = scale[self._sim._n-1]
        print(f"{self._sim._n} qubit - Scaling text by {factor:2.2f}")
        for k in self._textsize.keys():
            self._textsize[k] *= factor
            
        self._ax.set_xlim(limits[self._sim._n-1, 0])
        self._ax.set_ylim(limits[self._sim._n-1, 1])

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

        if self._sim._n == 1:
            self._drawArrows(-1, self._c + 2)  
        elif self._sim._n == 2:
            self._drawArrows(-2.5, self._c + 2.5)  
        elif self._sim._n == 3:
            self._drawArrows(-self._c+self._o*2/3, self._c + 2.5)  


    def _drawArrows(self, x0, y0):
        """Helper method to draw arrows for coordinate axis at given position.

        Args:
            x0 (float): Origin x direction
            y0 (float): Origin y direction
        """
        alen = self._c*2/3 
        if self._sim._n > 2:
            di = alen / np.sqrt(2)
            self._ax.text(x0+di/2.15, y0+di/2+.15, f'Qubit #{self._axis_labels[2]:1d}', size=self._textsize['axislbl'], horizontalalignment='right', verticalalignment='center')
            self._ax.arrow(x0, y0, di, di, **self._arrowStyle)
        if self._sim._n > 1:
            self._ax.text(x0-.3, y0-alen/2, f'Qubit #{self._axis_labels[1]:1d}', size=self._textsize['axislbl'], horizontalalignment='right', verticalalignment='center')
            self._ax.arrow(x0, y0, 0, -alen, **self._arrowStyle)
        self._ax.text(x0+alen/2+.2, y0+.3, f'Qubit #{self._axis_labels[0]:1d}', size=self._textsize['axislbl'], horizontalalignment='center', verticalalignment='center')    
        self._ax.arrow(x0, y0, alen, 0, **self._arrowStyle)


    def _drawDottedLine(self, points):
        """Helper method.
        Draw dotted lines connecting the given points in the xy-plane.

        Args:
            points (nested list([float, float]): List of xy-coordinates
        """
        self._ax.plot(self._coords[points,0], self._coords[points,1], color=self._colors['cube'], linewidth=self._widths['cube'], linestyle='dotted', zorder=1)


    def _drawLine(self, points):
        """Helper method.
        Draw lines connecting the given points in the xy-plane.

        Args:
            points (nested list([float, float]): List of xy-coordinates
        """
        self._ax.plot(self._coords[points,0], self._coords[points,1], color=self._colors['cube'], linewidth=self._widths['cube'], linestyle='solid', zorder=1)
            

    def _drawCircle(self, index):
        """Helper method. 
        Draw single circle for DCN. 

        Args:
            index (int): Index of the circle to be drawn.
        """
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
        off = 1.3
        off_magn_phase = .4

        if self._sim._n == 3:
            place = -1 if int(label[1]) else 1
        elif self._sim._n == 2:
            place = -1 if int(label[0]) else 1
        else: 
            place = -1

        self._ax.text(xpos, ypos + place*off, fr'$|{label:s}\rangle$', size=self._textsize['register'], horizontalalignment='center', verticalalignment='center')
        if self._showMagnPhase: 
            self._ax.text(xpos, ypos + place*(off+off_magn_phase), fr'{self._val[index]:+2.3f} | {np.rad2deg(self._phi[index]):+2.0f}°', size=self._textsize['magphase'], horizontalalignment='center', verticalalignment='center')

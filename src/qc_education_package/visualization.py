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

# TODO: Skalieren der Bilder (Text und Linienbreite) in den Griff bekommen. -> fixed figsize?


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
        self.fig = None
        self._lastState = None
        mpl.rcParams['text.usetex'] = False
        mpl.rcParams['text.parse_math'] = parse_math
        # common settings
        self._params = {'dpi': 300, 'transparent': True, 'showValues': False, 'bitOrder': simulator._bitOrder}

      

    def exportPNG(self, fname:str, title=''):
        """Export the current visualization as PNG image to given path.

        Args:
            fname (str): fname or path to export image to.
        """
        self._export(fname, 'png', title)


    def exportPDF(self, fname:str, title=''):
        """Export the current visualization as PDF file to given path.

        Args:
            fname (str): fname or path to export file to.
        """
        self._export(fname, 'pdf', title)
    

    def exportSVG(self, fname:str, title=''):
        """Export the current visualization as SVG image to given path.

        Args:
            fname (str): fname or path to export image to.
        """
        mpl.rcParams['svg.fonttype'] = 'none'  # Export as text and not paths
        self._export(fname, 'svg', title)


    def exportBase64(self, formatStr='png', title=''):
        """Export given format as base64 string. Mostly to handover images for flask website.

        Args:
            formatStr (str, optional): Format for image. Defaults to 'png'.

        Returns:
            str: base64 string representation of the generated image.
        """
        return b64encode(self._exportBuffer(formatStr, title)).decode('ascii') 


    def _exportBuffer(self, formatStr, title=''):
        """Export current visualization in format into IO buffer. 

        Args:
            formatStr (str, optional): Format for image. Defaults to 'png'.

        Returns:
            BytesIO: returns view of buffer containing image using BytesIO.getbuffer()
        """
        buf = BytesIO()
        self._export(buf, formatStr, title)
        return buf.getbuffer()


    def _export(self, target:str, formatStr:str, title:str):
        """General export method to save current pyplot figure, so all all exports will share same form factor, res etc.  

        Args:
            target (plt.savefig compatible object): Target to save pyplot figure to.
            formatStr (str, optional): Format for image. Defaults to 'png'.
        """
        self._redraw()
        self.fig.suptitle(title)
        self.fig.savefig(target, format=formatStr, bbox_inches='tight', pad_inches=0, dpi=self._params['dpi'], transparent=self._params['transparent'])


    def show(self):
        """Method to show current figure using plt.show but making sure the visualization is always redrawn.
        """ 
        self._redraw()
        plt.show()


    def _redraw(self):
        """Checks if simulator state is changed and redraws the image if so.
        """
        if self._lastState != self._sim:
            self._lastState = self._sim.copy()
            self.draw()


    def showMagnPhase(self, show_values:bool):
        """Switch showing magnitude and phase of each product state in a register

        Args:
            show_values (bool): Show value fir true, else do not show
        """
        self._params['showValues'] = show_values


    def draw(self):
        # TODO: Add style guide for draw method
        pass


    def hist(self, qubit=None, size=100) -> tuple[np.array, mpl.figure, mpl.axes.Axes]:
        """Create a histogram plot for repeated measurements of the simulator state. Here the state of the simulator will not collaps after a measurement.
        Arguments are passed to simulator.read(). If no qubit is given (qubit=None) all qubit are measured.

        Args:
            qubit (int or list(int), optional): qubit to read. Defaults to None.
            size (int), optional): Repeat the measurement size times. Default 1 Measurement.

        Returns:
            (np.array, mpl.figure, mpl.axes.Axes): Measurement results and pyplot figure and axes of the histogram plot to further manipulate if needed
        """
        _, result = self._sim.read(qubit, size)
        histFig = plt.figure(0)
        plt.get_current_fig_manager().set_window_title("Histogram plot")
        ax = histFig.subplots()
        ax.hist(result, density=True)
        ax.set_xlabel("Measured state")
        ax.set_ylabel("N")
        ax.set_title(f"Measured all qubits {size} times." if qubit is None else f"Measured qubit {qubit} {size} times.")
        return result, histFig, ax





class CircleNotation(Visualization):
    """A Visualization subclass for the well known Circle Notation representation.
    """

    def __init__(self, simulator, **kwargs):
        """Constructor for the Circle Notation representation.

        Args:
            simulator (qc_simulator.simulator): Simulator object to be visualized.
            cols (int, optional): Arrange Circle Notation into a set of columns. Defaults to None.
        """
        super().__init__(simulator)  # Execute constructor of superclass
        
        self._params.update({'color_edge': 'black', 'color_fill': '#77b6ba', 'color_phase': 'black',
        'width_edge': .7, 'width_phase': .7,
        'textsize_register': 10, 'textsize_magphase':8, 'dist_circles': 3,
        'offset_registerLabel': -1.35, 'offset_registerValues': -2.3})
        
        for key, val in kwargs:
            self._params[key] = val

        self.fig = None
              
        
    def draw(self, cols=None):
        """Draw Circle Notation representation of current simulator state.
        """
        self._cols = cols if cols != None else 2**self._sim._n
        bits = 2**self._sim._n
        self._c = self._params['dist_circles'] 
        x_max = self._c * self._cols
        y_max = self._c * bits/self._cols
        xpos = self._c/2
        ypos = y_max - self._c/2

        self.fig = plt.figure(layout='compressed', dpi=self._params['dpi'])
        plt.get_current_fig_manager().set_window_title("Circle Notation")
        ax = self.fig.gca()
       
        val = np.abs(self._sim._register)
        phi = -np.angle(self._sim._register, deg=False).flatten()
        lx, ly = np.sin(phi), np.cos(phi)

        ax.set_xlim([0, x_max])
        ax.set_ylim([0, y_max])
        ax.set_axis_off()
        ax.set_aspect('equal')

        # Scale textsizes such that ratio circles to textsize constant
        # automatic relative to length of y axis
        scale = [2, 1.25, .8]
        
        factor = scale[self._sim._n-1]
        # print(f"{self._sim._n} qubit - Scaling text by {factor:2.2f}")
        for k in ['textsize_register', 'textsize_magphase']:
            self._params[k] *= factor
        

        for i in range(2**self._sim._n):
            if val[i] > 1e-3:
                fill = mpatches.Circle((xpos, ypos), radius=val[i], color=self._params['color_fill'], edgecolor=None)
                phase = mlines.Line2D([xpos, xpos+lx[i]], [ypos, ypos+ly[i]], color=self._params['color_phase'], linewidth=self._params['width_phase'])
                ax.add_artist(fill)
                ax.add_artist(phase)
            ring = mpatches.Circle((xpos, ypos), radius=1, fill=False, edgecolor=self._params['color_edge'], linewidth=self._params['width_edge'])
            ax.add_artist(ring)
            label = np.binary_repr(i, width=self._sim._n) # width is deprecated since numpy 1.12.0
            ax.text(xpos, ypos + self._params['offset_registerLabel'], fr'$|{label:s}\rangle$', size=self._params['textsize_register'], horizontalalignment='center', verticalalignment='center')
            # NOTE text vs TextPath: text can easily be centered, textpath size is fixed when zooming
            # tp = TextPath((xpos-0.2*len(label), ypos - 1.35), f'|{label:s}>', size=0.4)
            # ax.add_patch(PathPatch(tp, color="black"))
            if self._params['showValues']: 
                ax.text(xpos, ypos + self._params['offset_registerValues'], f'{val[i]:+2.3f}\n{np.rad2deg(phi[i]):+2.0f}°', size=self._params['textsize_magphase'], horizontalalignment='center', verticalalignment='center')

            xpos += self._c
            if (i+1) % self._cols == 0:
                xpos = self._c / 2
                ypos -= self._c
            


class DimensionalCircleNotation(Visualization, ):
    """A Visualization subclass for the newly introduced Dimensional Circle Notation (DCN) representation.
    """

    def __init__(self, simulator, **kwargs):
        """Constructor for the Dimensional Circle Notation representation.
        This representation can be used for up to 3 qubits.
        

        Args:
            simulator (qc_simulator.simulator): Simulator object to be visualized.
        """
        super().__init__(simulator)  # Execute constructor of superclass
        assert(simulator._n <= 3)  # DCN is made for up to 3 qubits
        # TODO: 4-5 Qubits 

        # Style of circles NOTE: Still not sure if I like this approach for settings and params
        self._params.update({'color_edge': 'black', 'color_bg': 'white', 'color_fill': '#77b6baff', 'color_phase': 'black', 'color_cube': '#8a8a8a',
        'width_edge': .7, 'width_phase': .7, 'width_cube': .5, 'width_textsize': 10, 'width_textwidth': .1,
        'textsize_register': 10, 'textsize_magphase':8, 'textsize_axislbl':10,
        'width_arrow':.03, 'width_arrowhead':.3, 'length_arrow':.5, 'color_arrow_edge':None, 'color_arrow_face':'black',
        'dist_circles': 3.5, 
        'offset_3d': 1, 'offset_registerLabel': 1.3, 'offset_registerValues': .6})

        # Placement
        self._params['offset_3d'] = self._params['dist_circles'] / 2  # offset for 3rd dim qubits

        for key, val in kwargs:
            self._params[key] = val

        self._c = self._params['dist_circles']
        self._o = self._params['offset_3d']
        self._arrowStyle = {'width':self._params['width_arrow'], 'head_width':self._params['width_arrowhead'], 'head_length':self._params['length_arrow'], 'edgecolor':self._params['color_arrow_edge'], 'facecolor':self._params['color_arrow_face']}


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


        self.fig = None
        self._ax = None
        self._val, self._phi = None, None
        self._lx, self._ly = None, None
        self._scaleAxis()
         

    
    def draw(self):
        """Draw Dimensional Circle Notation representation of current simulator state.
        """
        self.fig = plt.figure(layout='compressed')
        plt.get_current_fig_manager().set_window_title("Dimensional Circle Notation")
        self._ax = self.fig.gca()
        self._ax.set_axis_off()
        self._ax.set_aspect('equal')

        self._val = np.abs(self._sim._register)
        self._phi = -np.angle(self._sim._register, deg=False).flatten()
        self._lx, self._ly = np.sin(self._phi), np.cos(self._phi)

        self._ax.set_xlim(self._limits[self._sim._n-1, 0])
        self._ax.set_ylim(self._limits[self._sim._n-1, 1])

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

        self._axis_labels = np.arange(1,self._sim._n+1)[::self._params['bitOrder']]
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
            self._ax.text(x0+di/2.15, y0+di/2+.15, f'Qubit #{self._axis_labels[2]:1d}', size=self._params['textsize_axislbl'], horizontalalignment='right', verticalalignment='center')
            self._ax.arrow(x0, y0, di, di, **self._arrowStyle)
        if self._sim._n > 1:
            self._ax.text(x0-.3, y0-alen/2, f'Qubit #{self._axis_labels[1]:1d}', size=self._params['textsize_axislbl'], horizontalalignment='right', verticalalignment='center')
            self._ax.arrow(x0, y0, 0, -alen, **self._arrowStyle)
        self._ax.text(x0+alen/2+.2, y0+.3, f'Qubit #{self._axis_labels[0]:1d}', size=self._params['textsize_axislbl'], horizontalalignment='center', verticalalignment='center')    
        self._ax.arrow(x0, y0, alen, 0, **self._arrowStyle)


    def _drawDottedLine(self, points):
        """Helper method.
        Draw dotted lines connecting the given points in the xy-plane.

        Args:
            points (nested list([float, float]): List of xy-coordinates
        """
        self._ax.plot(self._coords[points,0], self._coords[points,1], color=self._params['color_cube'], linewidth=self._params['width_cube'], linestyle='dotted', zorder=1)


    def _drawLine(self, points):
        """Helper method.
        Draw lines connecting the given points in the xy-plane.

        Args:
            points (nested list([float, float]): List of xy-coordinates
        """
        self._ax.plot(self._coords[points,0], self._coords[points,1], color=self._params['color_cube'], linewidth=self._params['width_cube'], linestyle='solid', zorder=1)
            

    def _drawCircle(self, index):
        """Helper method. 
        Draw single circle for DCN. 

        Args:
            index (int): Index of the circle to be drawn.
        """
        xpos, ypos = self._coords[index]
        # White bg circle area of unit circle
        bg = mpatches.Circle((xpos, ypos), radius=1, color=self._params['color_bg'], edgecolor=None)
        self._ax.add_artist(bg)
        # Fill area of unit circle
        if self._val[index] >= 1e-3:
            fill = mpatches.Circle((xpos, ypos), radius=self._val[index], color=self._params['color_fill'], edgecolor=None)
            self._ax.add_artist(fill)
        # Black margin for circles
        ring = mpatches.Circle((xpos, ypos), radius=1, fill=False, edgecolor=self._params['color_edge'], linewidth=self._params['width_edge'])
        self._ax.add_artist(ring)
        # Indicator for phase
        if self._val[index] >= 1e-3:
            phase = mlines.Line2D([xpos, xpos+self._lx[index]], [ypos, ypos+self._ly[index]], color=self._params['color_phase'], linewidth=self._params['width_phase'])
            self._ax.add_artist(phase)
        # Add label to circle
        label = np.binary_repr(index, width=self._sim._n) # width is deprecated since numpy 1.12.0
        # print(index, label)
        off = 1.3
        off_magn_phase = .6#.4

        if self._sim._n == 3:
            place = -1 if int(label[1]) else 1
        elif self._sim._n == 2:
            place = -1 if int(label[0]) else 1
        else: 
            place = -1

        self._ax.text(xpos, ypos + place*self._params['offset_registerLabel'], fr'$|{label:s}\rangle$', size=self._params['textsize_register'], horizontalalignment='center', verticalalignment='center')
        if self._params['showValues']:
            self._ax.text(xpos, ypos + place*(self._params['offset_registerLabel']+self._params['offset_registerValues']), f'{self._val[index]:+2.3f}\n{np.rad2deg(self._phi[index]):+2.0f}°', size=self._params['textsize_magphase'], horizontalalignment='center', verticalalignment='center')

    def _scaleAxis(self):
         # NOTE: Need to update witdth circle dist 
        # limits for coordinate axis [[[xmin, xmax] [ymin, ymax]], [...]]
        self._limits = np.array(  [[[-1.1, 4.6], [1.8, 6]],    # 1 Qubit
                            [[-4, 4.6], [-1.5, 6.5]],          # 2 Qubits
                            [[-4, 6.35],[-1, 8.35]]])      # 3 Qubits
        
        # Scale textsizes such that ratio circles to textsize constant
        # automatic relative to length of y axis
        scale = [2, 1.25, 1]
        
        factor = scale[self._sim._n-1]
        # print(f"{self._sim._n} qubit - Scaling text by {factor:2.2f}")
        for k in ['textsize_register', 'textsize_magphase', 'textsize_axislbl']:
            self._params[k] *= factor
            
        
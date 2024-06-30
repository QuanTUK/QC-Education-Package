# ----------------------------------------------------------------------------
# Created By: Nikolas Longen, nlongen@rptu.de
# Reviewed By: Maximilian Kiefer-Emmanouilidis, maximilian.kiefer@rptu.de
# Created: March 2023
# Project: DCN QuanTUK
# ----------------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
# from matplotlib.textpath import TextPath
# from matplotlib.patches import PathPatch
from io import BytesIO
from base64 import b64encode
import numpy as np

# TODO: Implement logging


class Visualization:
    """Superclass for all visualizations of quantum computer states.
    This way all visualizations inherit export methods.
    Alle subclasses must implement/overwrite a draw method and should also
    overwrite the __init__ method.
    """

    def __init__(self, simulator, parse_math=True):
        """Constructor for Visualization superclass.

        Args:
            simulator (qc_simulator.simulator): Simulator object to be
            visualized.
        """
        self._sim = simulator
        self.fig = None
        self._lastState = None
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["text.parse_math"] = parse_math
        # common settings
        self._params = {
            "dpi": 300,
            "transparent": True,
            "showValues": False,
            "bitOrder": simulator._bitOrder,
        }

    def exportPNG(self, fname: str, title=""):
        """Export the current visualization as PNG image to given path.

        Args:
            fname (str): fname or path to export image to.
        """
        self._export(fname, "png", title)

    def exportPDF(self, fname: str, title=""):
        """Export the current visualization as PDF file to given path.

        Args:
            fname (str): fname or path to export file to.
        """
        self._export(fname, "pdf", title)

    def exportSVG(self, fname: str, title=""):
        """Export the current visualization as SVG image to given path.

        Args:
            fname (str): fname or path to export image to.
        """
        mpl.rcParams["svg.fonttype"] = "none"  # Export as text and not paths
        self._export(fname, "svg", title)

    def exportBase64(self, formatStr="png", title=""):
        """Export given format as base64 string. Mostly to handover images for
        flask website.

        Args:
            formatStr (str, optional): Format for image. Defaults to 'png'.

        Returns:
            str: base64 string representation of the generated image.
        """
        return b64encode(self._exportBuffer(formatStr, title)).decode("ascii")

    def _exportBuffer(self, formatStr, title=""):
        """Export current visualization in format into IO buffer.

        Args:
            formatStr (str, optional): Format for image. Defaults to 'png'.

        Returns:
            BytesIO: returns view of buffer containing image using
            BytesIO.getbuffer()
        """
        buf = BytesIO()
        self._export(buf, formatStr, title)
        return buf.getbuffer()

    def _export(self, target: str, formatStr: str, title: str):
        """General export method to save current pyplot figure, so all all
        exports will share same form factor, res etc.

        Args:
            target (plt.savefig compatible object): Target to save pyplot
            figure to.
            formatStr (str, optional): Format for image. Defaults to 'png'.
        """
        self._redraw()
        self.fig.suptitle(title)
        self.fig.savefig(
            target,
            format=formatStr,
            bbox_inches="tight",
            pad_inches=0,
            dpi=self._params["dpi"],
            transparent=self._params["transparent"],
        )

    def show(self):
        """Method to show current figure using plt.show but making sure the
        visualization is always redrawn.
        """
        self._redraw()
        plt.show()

    def _redraw(self):
        """Checks if simulator state is changed and redraws the image if so."""
        if self._lastState != self._sim:
            self._lastState = self._sim.copy()
            self.draw()

    def showMagnPhase(self, show_values: bool):
        """Switch showing magnitude and phase of each product state in a
        register

        Args:
            show_values (bool): Show value fir true, else do not show
        """
        self._params.update({"showValues": show_values})

    def draw(self):
        # TODO: Add style guide for draw method
        pass

    def _createLabel(self, number: int):
        """Creates a binary label for a given index with zero padding fitting
        to the number of qubits.

        Args:
            number (int): Number to convert

        Returns:
            str: binary label fot the given number in braket
        """
        # NOTE: width is deprecated since numpy 1.12.0
        return np.binary_repr(number, width=self._sim._n)

    def hist(self, qubit=None, size=100
             ) -> tuple[np.array, mpl.figure, mpl.axes.Axes]:
        """Create a histogram plot for repeated measurements of the simulator
        state. Here the state of the simulator will not collaps after a
        measurement. Arguments are passed to simulator.read(). If no qubit is
        given (qubit=None) all qubit are measured.

        Args:
            qubit (int or list(int), optional): qubit to read.
            Defaults to None.
            size (int), optional): Repeat the measurement size times.
            Default 1 Measurement.

        Returns:
            (np.array, mpl.figure, mpl.axes.Axes): Measurement results and
            pyplot figure and axes of the histogram plot to further manipulate
            if needed
        """
        _, result = self._sim.read(qubit, size)
        histFig = plt.figure(0)
        plt.get_current_fig_manager().set_window_title("Histogram plot")
        ax = histFig.subplots()
        ax.hist(result, density=True)
        ax.set_xlabel("Measured state")
        ax.set_ylabel("N")
        ax.set_title(
            f"Measured all qubits {size} times."
            if qubit is None
            else f"Measured qubit {qubit} {size} times."
        )
        return result, histFig, ax


class CircleNotation(Visualization):
    """A Visualization subclass for the well known Circle Notation
    representation.
    """

    def __init__(self, simulator, **kwargs):
        """Constructor for the Circle Notation representation.

        Args:
            simulator (qc_simulator.simulator): Simulator object to be
            visualized.
            cols (int, optional): Arrange Circle Notation into a set of
            columns. Defaults to None.
        """
        super().__init__(simulator)  # Execute constructor of superclass

        self._params.update(
            {
                "color_edge": "black",
                "color_fill": "#77b6ba",
                "color_phase": "black",
                "width_edge": 0.7,
                "width_phase": 0.7,
                "textsize_register": 10,
                "textsize_magphase": 8,
                "dist_circles": 3,
                "offset_registerLabel": -1.35,
                "offset_registerValues": -2.3,
            }
        )

        self.fig = None

    def draw(self, cols=None):
        """Draw Circle Notation representation of current simulator state."""
        self._cols = cols if cols is not None else 2**self._sim._n
        bits = 2**self._sim._n
        self._c = self._params["dist_circles"]
        x_max = self._c * self._cols
        y_max = self._c * bits / self._cols
        xpos = self._c / 2
        ypos = y_max - self._c / 2

        self.fig = plt.figure(layout="compressed", dpi=self._params["dpi"])
        plt.get_current_fig_manager().set_window_title("Circle Notation")
        ax = self.fig.gca()

        val = np.abs(self._sim._register)
        phi = -np.angle(self._sim._register, deg=False).flatten()
        lx, ly = np.sin(phi), np.cos(phi)

        ax.set_xlim([0, x_max])
        ax.set_ylim([0, y_max])
        ax.set_axis_off()
        ax.set_aspect("equal")

        # Scale textsizes such that ratio circles to textsize constant
        # automatic relative to length of y axis
        scale = [2, 1.25, 0.8]

        factor = scale[self._sim._n - 1]
        for k in ["textsize_register", "textsize_magphase"]:
            self._params[k] *= factor

        for i in range(2**self._sim._n):
            if val[i] > 1e-3:
                fill = mpatches.Circle(
                    (xpos, ypos),
                    radius=val[i],
                    color=self._params["color_fill"],
                    edgecolor=None,
                )
                phase = mlines.Line2D(
                    [xpos, xpos + lx[i]],
                    [ypos, ypos + ly[i]],
                    color=self._params["color_phase"],
                    linewidth=self._params["width_phase"],
                )
                ax.add_artist(fill)
                ax.add_artist(phase)
            ring = mpatches.Circle(
                (xpos, ypos),
                radius=1,
                fill=False,
                edgecolor=self._params["color_edge"],
                linewidth=self._params["width_edge"],
            )
            ax.add_artist(ring)
            label = self._createLabel(i)
            ax.text(
                xpos,
                ypos + self._params["offset_registerLabel"],
                rf"$|{label:s}\rangle$",
                size=self._params["textsize_register"],
                horizontalalignment="center",
                verticalalignment="center",
            )
            # NOTE text vs TextPath:
            # text can easily be centered
            # textpath size is fixed when zooming
            # tp = TextPath((xpos-0.2*len(label),
            # ypos - 1.35),
            # f'|{label:s}>',
            # size=0.4)
            # ax.add_patch(PathPatch(tp, color="black"))
            if self._params["showValues"]:
                ax.text(
                    xpos,
                    ypos + self._params["offset_registerValues"],
                    f"{val[i]:+2.3f}\n{np.rad2deg(phi[i]):+2.0f}°",
                    size=self._params["textsize_magphase"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )

            xpos += self._c
            if (i + 1) % self._cols == 0:
                xpos = self._c / 2
                ypos -= self._c


class DimensionalCircleNotation(Visualization):
    """A Visualization subclass for the newly introduced Dimensional Circle
    Notation (DCN) representation.
    """

    def __init__(self, simulator, parse_math=True, version=2):
        """Constructor for the Dimensional Circle Notation
        representation.

        Args:
            simulator (qc_simulator.simulator): Simulator object to be
            visualized.
        """
        super().__init__(simulator)  # Execute constructor of superclass
        print(f"Setting up DCN Visualization in version {version}.")
        self._arrowStyle = {
            "width": .03,
            "head_width": .3,
            "head_length": .5,
            "edgecolor": None,
            "facecolor": 'black',
        }

        self._params.update({
            'version': version,
            'labels_dirac': True if version == 1 else False,
            'color_edge': 'black',
            'color_bg': 'white',
            'color_fill': '#77b6baff',
            'color_phase': 'black',
            'color_cube': '#8a8a8a',
            'width_edge': .7,
            'width_phase': .7,
            'width_cube': .5,
            'width_textwidth': .1,
            'offset_registerLabel': 1.3,
            'offset_registerValues': .6})

        # Create empty variables for later use
        self.fig = None
        self._ax = None
        self._val, self._phi = None, None
        self._lx, self._ly = None, None

    def draw(self):
        """Draw Dimensional Circle Notation representation of current
        simulator state.
        """
        # Setup pyplot figure
        self.fig = plt.figure(layout="compressed")
        plt.get_current_fig_manager().set_window_title(
            "Dimensional Circle Notation"
        )
        self._ax = self.fig.gca()
        self._ax.set_axis_off()
        self._ax.set_aspect("equal")
        # Get arrays with magnitude and phase of the register
        self._val = np.abs(self._sim._register)
        self._phi = -np.angle(self._sim._register, deg=False).flatten()
        # Get x, y components of the phase for drawing phase dial inside
        # circles
        self._lx, self._ly = np.sin(self._phi), np.cos(self._phi)

        # Explicit positions for the qubits do not specify the bit-order
        # Bitorder can be changed by flipping the value, phase, label arrays
        self._axis_labels = np.arange(
            1, self._sim._n + 1)[:: self._params["bitOrder"]]

        # Hard coded visualization for number of qubits
        # Match for python >= 3.10
        # match self._sim._n:
        # HowTo add visualization for more qubits:
        # 1. Define coords in self._coords.
        #    All draw methods afterwards work with indices of this array.
        #    Coords should have the index according to value, phase arrays
        # 2. Draw wireframe connecting points of certain indices first
        # 2. Draw circles at given position using the index

        # 1 Qubit:
        #case 1:
        if self._sim._n == 1:
            # Setup positions of the circles, so these can be accessed easy
            self._coords = np.array(
                [[0, 1],
                    [1, 1]],
                dtype=float
            )
            # Set distance
            self._coords *= 3.5

            # Set text sizes for this visualization
            self._params.update({
                'textsize_register': 20,
                'textsize_magphase': 16,
                'textsize_axislbl': 20
            })

            # In the following the index is used to draw the visualization
            # Draw cube wire frame, qubit 1
            self._drawLine([0, 1])
            self._drawCircle(1)
            self._drawCircle(0)

            # old style dcn coordinate axis
            # dirac labels are coonfigured in init already
            if self._params['version'] == 1:
                # Text for coordinate axis
                self._ax.text(
                    0.35,
                    5.5,
                    f"Qubit #{self._axis_labels[0]:1d}",
                    size=self._params['textsize_axislbl'],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                # Arrows for coordinate axis (x,y,dx,dy, **kwargs)
                self._ax.arrow(-1, 5, 2.3, 0, **self._arrowStyle)

                # Set axis limits according to plot size (grows with n)
                self._ax.set_xlim([-1.1, 4.6])
                self._ax.set_ylim([1.8, 6])
            
            # DCN V2: different coordinate axis
            else:
                # Arrows for coordinate axis (x,y,dx,dy, **kwargs)
                x, y, len_tick = -1, 5, .2
                self._ax.arrow(x, y, self._coords[1, 0] + 1.5, 0,
                                **self._arrowStyle)
                tick_y = [y-len_tick, y+len_tick]
                self._ax.plot(
                    [self._coords[0, 0], self._coords[0, 0]],
                    tick_y,  # y coord like arrow
                    color='black',
                    linewidth=1,
                    linestyle="solid",
                    zorder=1,
                )
                self._ax.text(
                    self._coords[0, 0],
                    y + 2*len_tick,
                    "0",
                    size=self._params["textsize_register"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                self._ax.plot(
                    [self._coords[0, 1], self._coords[0, 1]],
                    tick_y,  # y coord like arrow
                    color='black',
                    linewidth=1,
                    linestyle="solid",
                    zorder=1,
                )
                self._ax.text(
                    self._coords[0, 1],
                    y + 2*len_tick,
                    "1",
                    size=self._params["textsize_register"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                # Add Qubit Nr. Label to arrow
                self._ax.text(
                    self._coords[0, 1]/2,
                    y + 2*len_tick,
                    "Qubit #1",
                    size=self._params["textsize_register"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )

                # Set axis limits according to plot size (grows with n)
                self._ax.set_xlim([-1.1, 4.6])
                self._ax.set_ylim([1.8, 6])

        # 2 Qubits:
        # case 2:
        elif self._sim._n == 2:
            # Setup positions of the circles, so these can be accessed easy
            self._coords = np.array(
                [[0, 1],
                    [1, 1],
                    [0, 0],
                    [1, 0]],
                dtype=float
            )
            # Set distance
            self._coords *= 3.5

            # Set text sizes for this visualization
            self._params.update({
                'textsize_register': 17.5,
                'textsize_magphase': 14,
                'textsize_axislbl': 17.5
            })

            # In the following the index is used to draw the visualization
            # Draw cube wire frame
            self._drawLine([0, 2, 3, 1])
            self._drawCircle(3)
            self._drawCircle(2)
            # Qubit 1
            self._drawLine([0, 1])
            # Draw arrows of coordinate axis
            self._drawCircle(1)
            self._drawCircle(0)

            # old style dcn coordinate axis
            # dirac labels are coonfigured in init already
            if self._params['version'] == 1:
                # Text for coordinate axis
                self._ax.text(
                    -1,
                    5.5,
                    f"Qubit #{self._axis_labels[0]:1d}",
                    size=self._params['textsize_axislbl'],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                self._ax.text(
                    -2.8,
                    4.25,
                    f"Qubit #{self._axis_labels[1]:1d}",
                    size=self._params['textsize_axislbl'],
                    horizontalalignment="right",
                    verticalalignment="center",
                )
                # Arrows for coordinate axis (x,y,dx,dy, **kwargs)
                self._ax.arrow(-2.5, 6, 2.3, 0, **self._arrowStyle)
                self._ax.arrow(-2.5, 6, 0, -3.5, **self._arrowStyle)

                # Self axis limits according to plot size (grows with n)
                self._ax.set_xlim([-4, 4.6])
                self._ax.set_ylim([-1.5, 6.5])

            # DCN V2: different coordinate axis
            else:
                # Arrows for coordinate axis (x,y,dx,dy, **kwargs)
                x, y, len_tick = -1.5, 5, .2
                self._ax.arrow(x-.5, y, self._coords[1, 1] + 3, 0,
                                **self._arrowStyle)
                self._ax.arrow(x, y+.5, 0, - (self._coords[0, 1] + 3),
                                **self._arrowStyle)
                # Ticks on horizontal axis
                tick_y = [y-len_tick, y+len_tick]
                self._ax.plot(
                    [self._coords[0, 0], self._coords[0, 0]],
                    tick_y,  # y coord like arrow
                    color='black',
                    linewidth=1,
                    linestyle="solid",
                    zorder=1,
                )
                self._ax.text(
                    self._coords[0, 0],
                    y + 2*len_tick,
                    "0",
                    size=self._params["textsize_register"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                self._ax.plot(
                    [self._coords[0, 1], self._coords[0, 1]],
                    tick_y,  # y coord like arrow
                    color='black',
                    linewidth=1,
                    linestyle="solid",
                    zorder=1,
                )
                self._ax.text(
                    self._coords[0, 1],
                    y + 2*len_tick,
                    "1",
                    size=self._params["textsize_register"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                # Add Qubit Nr. Label to arrow
                self._ax.text(
                    self._coords[0, 1]/2,
                    y + 2*len_tick,
                    "Qubit #1",
                    size=self._params["textsize_register"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                # ticks on vertical axis
                tick_x = [x-len_tick, x+len_tick]
                self._ax.plot(
                    tick_x,  # x coord like arrow
                    [self._coords[0, 1], self._coords[0, 1]],
                    color='black',
                    linewidth=1,
                    linestyle="solid",
                    zorder=1,
                )
                self._ax.text(
                    x - 2*len_tick,
                    self._coords[0, 1],
                    "0",
                    size=self._params["textsize_register"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                self._ax.plot(
                    tick_x,  # y coord like arrow
                    [self._coords[3, 1], self._coords[3, 1]],
                    color='black',
                    linewidth=1,
                    linestyle="solid",
                    zorder=1,
                )
                self._ax.text(
                    x - 2*len_tick,
                    self._coords[3, 1],
                    "1",
                    size=self._params["textsize_register"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                # Add Qubit Nr. Label to arrow
                self._ax.text(
                    x - 1.2,
                    self._coords[0, 1]/2,
                    "Qubit #2",
                    size=self._params["textsize_register"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                
                # Self axis limits according to plot size (grows with n)
                self._ax.set_xlim([-4, 5])
                self._ax.set_ylim([-1.5, 6.5])

        # 3 Qubits:
        # case 3:
        elif self._sim._n == 3:
            # Setup positions of the circles, so these can be accessed easy
            self._coords = np.array(
                [[0, 1],
                    [1, 1],
                    [0, 0],
                    [1, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0]],
                dtype=float,
            )
            # Set distance
            self._coords *= 3.5
            # Offset 3rd dim circles to the rear from position of the
            # first 4 circles
            self._coords[4:] = self._coords[:4] + 1.75

            # Set text sizes for this visualization
            self._params.update({
                'textsize_register': 10,
                'textsize_magphase': 8,
                'textsize_axislbl': 10
            })

            # In the following the index is used to draw the visualization
            # Draw cube wire, Qubit 3
            self._drawLine([0, 4, 5])
            self._drawLine([1, 5, 7, 3])
            self._drawDottedLine([2, 6, 7])
            self._drawDottedLine([4, 6])
            self._drawCircle(7)
            self._drawCircle(6)
            self._drawCircle(5)
            self._drawCircle(4)
            # Qubit 2
            self._drawLine([0, 2, 3, 1])
            self._drawCircle(3)
            self._drawCircle(2)
            # Qubit 1
            self._drawLine([0, 1])
            self._drawCircle(1)
            self._drawCircle(0)

            # old style dcn coordinate axis
            # dirac labels are coonfigured in init already
            if self._params['version'] == 1:
                # Text for coordinate axis
                self._ax.text(
                    -1,
                    +5.5,
                    f"Qubit #{self._axis_labels[0]:1d}",
                    size=self._params['textsize_axislbl'],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                self._ax.text(
                    -2.8,
                    +4.25,
                    f"Qubit #{self._axis_labels[1]:1d}",
                    size=self._params['textsize_axislbl'],
                    horizontalalignment="right",
                    verticalalignment="center",
                )
                self._ax.text(
                    0,
                    +6.975,
                    f"Qubit #{self._axis_labels[2]:1d}",
                    size=self._params['textsize_axislbl'],
                    horizontalalignment="right",
                    verticalalignment="center",
                )
                # Arrows for coordinate axis (x,y,dx,dy, **kwargs)
                self._ax.arrow(-2.5, 6, 2.3, 0, **self._arrowStyle)
                self._ax.arrow(-2.5, 6, 0, -3.5, **self._arrowStyle)
                self._ax.arrow(-2.5, 6, 1.65, 1.65, **self._arrowStyle)

                # Self axis limits according to plot size (grows with n)
                self._ax.set_xlim([-4, 6.35])
                self._ax.set_ylim([-1, 8.35])
                
            # DCN V2: different coordinate axis
            else:
                # Arrows for coordinate axis (x,y,dx,dy, **kwargs)
                x, y, len_tick = -2, 7, .2
                # horizontal axis
                self._ax.arrow(x-.5, y, self._coords[1, 1] + 4.5, 0,
                                **self._arrowStyle)
                # vertical axis
                self._ax.arrow(x, y+.5, 0, - (self._coords[0, 1] + 5),
                                **self._arrowStyle)
                # diagonal axis
                self._ax.arrow(x-.35, y-.35, 3.35, 3.35,
                                **self._arrowStyle)
                # Ticks on horizontal axis
                tick_y = [y-len_tick, y+len_tick]
                self._ax.plot(
                    [self._coords[0, 0], self._coords[0, 0]],
                    tick_y,  # y coord like arrow
                    color='black',
                    linewidth=1,
                    linestyle="solid",
                    zorder=1,
                )
                self._ax.text(
                    self._coords[0, 0],
                    y + 2*len_tick,
                    "0",
                    size=self._params["textsize_register"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                self._ax.plot(
                    [self._coords[0, 1], self._coords[0, 1]],
                    tick_y,  # y coord like arrow
                    color='black',
                    linewidth=1,
                    linestyle="solid",
                    zorder=1,
                )
                self._ax.text(
                    self._coords[0, 1],
                    y + 2*len_tick,
                    "1",
                    size=self._params["textsize_register"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                # Add Qubit Nr. Label to arrow
                self._ax.text(
                    self._coords[0, 1]/2,
                    y + 3*len_tick,
                    "Qubit #1",
                    size=self._params["textsize_register"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                # ticks on vertical axis
                tick_x = [x-len_tick, x+len_tick]
                self._ax.plot(
                    tick_x,  # x coord like arrow
                    [self._coords[0, 1], self._coords[0, 1]],
                    color='black',
                    linewidth=1,
                    linestyle="solid",
                    zorder=1,
                )
                self._ax.text(
                    x - 2*len_tick,
                    self._coords[0, 1],
                    "0",
                    size=self._params["textsize_register"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                self._ax.plot(
                    tick_x,  # y coord like arrow
                    [self._coords[3, 1], self._coords[3, 1]],
                    color='black',
                    linewidth=1,
                    linestyle="solid",
                    zorder=1,
                )
                self._ax.text(
                    x - 2*len_tick,
                    self._coords[3, 1],
                    "1",
                    size=self._params["textsize_register"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                # Add Qubit Nr. Label to arrow
                self._ax.text(
                    x - 1.2,
                    self._coords[0, 1]/2,
                    "Qubit #2",
                    size=self._params["textsize_register"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                # ticks on diagonal axis
                len_tick /= np.sqrt(2)
                off1, off2 = 0.8, 2.2
                self._ax.plot(
                    # coords somewhat random so it lust just right
                    [x+off1+len_tick, x+off1-len_tick],
                    [y+off1-len_tick, y+off1+len_tick],
                    color='black',
                    linewidth=1,
                    linestyle="solid",
                    zorder=1,
                )
                self._ax.text(
                    x+off1-0.4,
                    y+off1+0.4,
                    "0",
                    size=self._params["textsize_register"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                self._ax.plot(
                    # coords somewhat random so it lust just right
                    [x+off2+len_tick, x+off2-len_tick],
                    [y+off2-len_tick, y+off2+len_tick],
                    color='black',
                    linewidth=1,
                    linestyle="solid",
                    zorder=1,
                )
                self._ax.text(
                    x+off2-.4,
                    y+off2+.4,
                    "1",
                    size=self._params["textsize_register"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                # Add Qubit Nr. Label to arrow
                self._ax.text(
                    -1.75,
                    y + 2.1,
                    "Qubit #3",
                    size=self._params["textsize_register"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                
                # Self axis limits according to plot size (grows with n)
                self._ax.set_xlim([-3, 7])
                self._ax.set_ylim([-3, 12])

        # case _:
        else:
            raise NotImplementedError(
                "DCN V2 is not implemented for so " + "many qubits."
            )

        # Flip axis labels if bitOrder is set to 1
        self._axis_labels = np.arange(1, self._sim._n + 1
                                      )[:: self._params["bitOrder"]]

    def _drawDottedLine(self, index):
        """Helper method:
        Draw dotted lines connecting points at given index. The coordinates of
        the points a defined internal in the _coords array

        Args:
            index (nested list([float, float]): List of indices
        """
        self._ax.plot(
            self._coords[index, 0],
            self._coords[index, 1],
            color=self._params["color_cube"],
            linewidth=self._params["width_cube"],
            linestyle="dotted",
            zorder=1,
        )

    def _drawLine(self, index):
        """Helper method:
        Draw lines connecting the given points at given index. The coordinates
        of the points a defined internal in the _coords array.

        Args:
            index (nested list([float, float]): List of indices
        """
        self._ax.plot(
            self._coords[index, 0],
            self._coords[index, 1],
            color=self._params["color_cube"],
            linewidth=self._params["width_cube"],
            linestyle="solid",
            zorder=1,
        )

    def _drawCircle(self, index):
        """Helper method:
        Draw single circle for DCN. Position and values of the circle are
        provided internal. Hand over the corect index

        Args:
            index (int): Index of the circle to be drawn.
        """
        xpos, ypos = self._coords[index]
        # White bg circle area of unit circle
        bg = mpatches.Circle(
            (xpos, ypos),
            radius=1,
            color=self._params["color_bg"],
            edgecolor=None
        )
        self._ax.add_artist(bg)
        # Fill area of unit circle
        if self._val[index] >= 1e-3:
            fill = mpatches.Circle(
                (xpos, ypos),
                radius=self._val[index],
                color=self._params["color_fill"],
                edgecolor=None,
            )
            self._ax.add_artist(fill)
        # Black margin for circles
        ring = mpatches.Circle(
            (xpos, ypos),
            radius=1,
            fill=False,
            edgecolor=self._params["color_edge"],
            linewidth=self._params["width_edge"],
        )
        self._ax.add_artist(ring)
        # Indicator for phase
        if self._val[index] >= 1e-3:
            phase = mlines.Line2D(
                [xpos, xpos + self._lx[index]],
                [ypos, ypos + self._ly[index]],
                color=self._params["color_phase"],
                linewidth=self._params["width_phase"],
            )
            self._ax.add_artist(phase)
        if self._params['labels_dirac']:
            # Add dirac label to circle
            label = self._createLabel(index)
            if self._sim._n == 3:
                place = -1 if int(label[1]) else 1
            elif self._sim._n == 2:
                place = -1 if int(label[0]) else 1
            else:
                place = -1

            self._ax.text(
                xpos,
                ypos + place * self._params["offset_registerLabel"],
                rf"$|{label:s}\rangle$",
                size=self._params["textsize_register"],
                horizontalalignment="center",
                verticalalignment="center",
            )
        if self._params["showValues"]:
            self._ax.text(
                xpos,
                ypos
                + place
                * (
                    self._params["offset_registerLabel"]
                    + self._params["offset_registerValues"]
                ),
                f"{self._val[index]:+2.3f}\n"
                + f"{np.rad2deg(self._phi[index]):+2.0f}°",
                size=self._params["textsize_magphase"],
                horizontalalignment="center",
                verticalalignment="center",
            )

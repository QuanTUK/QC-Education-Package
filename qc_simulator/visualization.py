from simulator import simulator
from pyplot3d_helper import pathpatch_2d_to_3d, pathpatch_translate
import matplotlib
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


    def export_base64(self):
        return base64.b64encode(self._export_buffer()).decode("ascii")
        

    def export_png(self, fname:str):
        self._export(fname)


    def _export_buffer(self):
        buf = BytesIO()
        self._export(buf)
        return buf.getbuffer()


    def _export(self, target):
        if self._fig is None:
            self.draw()
        # TODO: Pyplot Bug: https://github.com/matplotlib/matplotlib/issues/21688
        # Bugreport schreiben
        self._fig.savefig(target, format="png", bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)


    def show(self):
        """Show fig
        """ 
        # NOTE: Assert Gui backend
        if self._fig == None:
            self.draw()
        matplotlib.use('TkAgg')
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
        self._colors = {'edge': 'black', 'edge_bg': 'white', 'fill': '#77b6baff', 'phase': 'black', 'cube': '#5a5a5a'}
        self._widths = {'edge': .5, 'phase': .5, 'cube': .5, 'textsize': 5, 'textwidth': .1}
        self._circleDist = 3

        self._fig = None
        self._cols = cols if cols != None else 2**self._sim._n      
        
        
    def draw(self):
        bits = 2**self._sim._n
        x_max = self._circleDist * self._cols
        y_max = self._circleDist * bits/self._cols
        xpos = self._circleDist/2
        ypos = y_max - self._circleDist/2

        self._fig = plt.figure(layout='compressed', dpi=300)  # layout='compressed'
        ax = self._fig.gca()
       
        val = np.abs(self._sim._register)
        phi = np.angle(self._sim._register, deg=False).flatten()
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
            # fr'$|{label:s}\rangle$', size=self._widths['textsize'], usetex=False), color="black", linewidth=self._widths['textwidth'])
            ax.text(xpos, ypos - 1.35, fr'$|{label:s}\rangle$', size=self._widths['textsize'], usetex=False, horizontalalignment='center', verticalalignment='center')
            # NOTE text vs TextPath: text can easily be centered, textpath size is fixed when zooming
            # tp = TextPath((xpos-0.2*len(label), ypos - 1.35), f'|{label:s}>', size=0.4)
            # ax.add_patch(PathPatch(tp, color="black"))
            # NOTE Area/prop as text inside circle?
            xpos += self._circleDist
            if (i+1) % self._cols == 0:
                xpos = self._circleDist / 2
                ypos -= self._circleDist


class DimensionalCircleNotation(Visualization):
    """Dimensional Circle Notation
    """

    def __init__(self, simulator, azim=25, elev=25, roll= 0):
        self._sim = simulator
        self._colors = {'edge': 'black', 'edge_bg': 'white', 'fill': '#77b6baff', 'phase': 'black', 'cube': '#5a5a5a'}
        self._widths = {'edge': .5, 'phase': .5, 'cube': .5, 'textsize': .4, 'textwidth': .1}
        self._circleDist = 5

        # Tait–Bryan angles -> https://en.wikipedia.org/wiki/Euler_angles#Tait%E2%80%93Bryan_angles
        # yaw, pitch, roll
        self._azim = azim
        self._elev = elev
        self._roll = roll

        self._fig = None

    def draw(self):
        val = np.abs(self._sim._register).flatten()
        phase = np.angle(self._sim._register, deg=False).flatten()
        lx, ly = -np.sin(phase), np.cos(phase)

        self._fig = plt.figure(layout='compressed', dpi=300)  # layout='compressed'
        ax = self._fig.add_subplot(projection='3d', proj_type = 'ortho', computed_zorder=False)

        ax.set_xlim([-self._circleDist, self._circleDist])
        ax.set_ylim([-self._circleDist, self._circleDist])
        ax.set_zlim([-self._circleDist, self._circleDist])

        ax.set_axis_off()
        ax.set_aspect('equal')
        
        # Adjust camera view for DCN representation
        ax.view_init(azim=self._azim, elev=self._elev, roll=self._roll, vertical_axis='y')

        # Draw cube wireframe connecting circles
        c = self._circleDist/2
        ax.plot([c, -c, -c, c, c],[c, c, -c, -c, c],[c,c,c,c,c], color=self._colors['cube'], linewidth=self._widths['cube'], linestyle='solid', zorder=1)
        ax.plot([c, -c, -c, c, c],[c, c, c, c, c],[c,c,-c,-c,c], color=self._colors['cube'], linewidth=self._widths['cube'], linestyle='-', zorder=1)
        ax.plot([c, c, c, c, c],[c, -c, -c, c, c],[c,c,-c,-c,c], color=self._colors['cube'], linewidth=self._widths['cube'], linestyle='-', zorder=1)
        ax.plot([c, c, c, c, c],[c, -c, -c, c, c],[c,c,-c,-c,c], color=self._colors['cube'], linewidth=self._widths['cube'], linestyle='-', zorder=1)
        ax.plot([-c, -c, c],[-c, -c, -c],[c, -c, -c], color=self._colors['cube'], linewidth=self._widths['cube'], linestyle='dotted', zorder=1)
        ax.plot([-c, -c],[c, -c],[-c, -c], color=self._colors['cube'], linewidth=self._widths['cube'], linestyle='dotted', zorder=1)

        for i in range(2**self._sim._n):
            label = np.binary_repr(i, width=self._sim._n) # width is deprecated since numpy 1.12.0
            # Get coordinates on cube from binary repr
            x,y,z = -self._circleDist * (1-int(label[2]))+c, self._circleDist * (1-int(label[1]))-c, self._circleDist * (1-int(label[0]))-c

            bg = mpatches.Circle((0, 0), radius=1, fill=True, facecolor=self._colors['edge_bg'], zorder=5)  
            ax.add_patch(bg)
            # for vertical axis = z, order/role of angle changes, when vertical axis changes
            pathpatch_2d_to_3d(bg, azim=self._roll, elev=self._azim, roll=-self._elev, z = 0)
            pathpatch_translate(bg, (x,y,z))

            fill = mpatches.Circle((0, 0), radius=val[i], color=self._colors['fill'], edgecolor=None, zorder=10)
            ax.add_patch(fill)
            pathpatch_2d_to_3d(fill, azim=self._roll, elev=self._azim, roll=-self._elev, z = 0)
            pathpatch_translate(fill, (x,y,z))

            ring = mpatches.Circle((0, 0), radius=1, fill=False, edgecolor=self._colors['edge'], linewidth=self._widths['edge'], zorder=10) 
            ax.add_patch(ring)
            pathpatch_2d_to_3d(ring, azim=self._roll, elev=self._azim, roll=-self._elev, z = 0)
            pathpatch_translate(ring, (x,y,z))

            dial = mpatches.FancyArrowPatch((0, 0), (lx[i], ly[i]), linewidth=self._widths['phase'] ,zorder=100)  
            ax.add_patch(dial)
            pathpatch_2d_to_3d(dial, azim=self._roll, elev=self._azim, roll=-self._elev, z = 0)
            pathpatch_translate(dial, (x,y,z))

            off = -1.7 if int(label[1]) else 1.3
            tp = PathPatch(TextPath((0,0), fr'$|{label:s}\rangle$', size=self._widths['textsize'], usetex=False), color="black", linewidth=self._widths['textwidth'])
            ax.add_patch(tp)
            pathpatch_2d_to_3d(tp, azim=self._roll, elev=self._azim, roll=-self._elev, z = 0)
            pathpatch_translate(tp, (x-.6, y+off,z))
        
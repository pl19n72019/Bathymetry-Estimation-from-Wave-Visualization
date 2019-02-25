# -*- coding: utf-8 -*-
from time import time
import numpy as np
import inspect
from src.models import cnn
from src.networks.cnn import CNN
from src.networks.autoencoder import AutoEncoder
from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import QMainWindow, QSizePolicy, QPushButton, QAction, \
    QDesktopWidget, QFileDialog, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, FigureCanvasQT
from matplotlib.figure import Figure
import matplotlib.animation as animation


class ApplicationWindow(QMainWindow):
    """The bathymetry-prediction main application window and all its containers.

    The bathymetry-prediction main application window contains the Hovmöller
    diagram canvas, the dynamic or static bathymetry canvas and all the
    necessary information. The window is by default maximized and no responsive
    tools have been implemented (it should also not be modified).

    Attributes:
        title (str): Title of the bathymetry-prediction application window.

    """

    def __init__(self, title=''):
        """Initialization of the bathymetry-prediction main application window.

        When an application window is called, the header parameters and the
        containers are initialize and should not be modified.

        Args:
            title (str): the main application window's title

        Examples:
            This bathymetry-prediction main application window should be called
            in an application environment.

            >>> app = PyQt5.QtWidgets.QApplication(sys.argv)
            >>> ex = ApplicationWindow.ApplicationWindow(title="Title")
            >>> sys.exit(app.exec_())

        """
        super().__init__()

        self.title = title    # title of the window
        self.__b_width = 300  # width of the buttons
        self.__b_height = 50  # height of the buttons
        self.__b_shift = 50   # shift of the buttons
        self.__m_height = 30  # height of the menu bar
        self.__set_window()   # window containers

    def __set_window(self):
        """Establishment of the bathymetry-prediction window's canvas.

        This methods set up the front-end header parameters, the menu bar, all
        the canvas (either static and dynamic) and all the buttons.

        """
        geometry = QDesktopWidget().availableGeometry()  # screen's geometry

        # define the window titled self.title
        self.setWindowTitle(self.title)
        self.setFixedSize(geometry.width(), geometry.height())
        self.__menu_bar()  # Setting up the menu bar

        # buttons
        self.__buttons()

        # timestack Canvas
        self.ts_canvas = TSCanvas(self,
                                  width=int((self.width() - self.__b_width) / 3),
                                  height=int(self.height() - self.__m_height))
        self.ts_canvas.move(0, self.__m_height)
        self.ts = ''

        # bathymetry Canvas
        self.b_canvas = BCanvas(self,
                                width=int(2 * (self.width() - self.__b_width) / 3),
                                height=int(2 * self.height() / 3))
        self.b_canvas.move(self.ts_canvas.get_width_height()[0],
                           self.__m_height)
        self.bath = 'src/graphics/B_init.npy'

        self.weight = ''

        self.show()

    def __menu_bar(self):
        """Establishment of the front-end and back-end menu bar.

        Classically, the available menus are: File.
        Creation of keybord shortcut for each button.

        """
        # menu front-end
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu('File')

        # choose timestack file button action
        choose_ts_button = QAction('Choose timestack file', self)
        choose_ts_button.setShortcut('Ctrl+T')
        choose_ts_button.setStatusTip('Choose timestack file')
        choose_ts_button.triggered.connect(self.browse_ts)
        file_menu.addAction(choose_ts_button)

        # choose expected bathymetry file button action
        choose_b_button = QAction('Choose expected bathymetry file', self)
        choose_b_button.setShortcut('Ctrl+B')
        choose_b_button.setStatusTip('Choose expected bathymetry file')
        choose_b_button.triggered.connect(self.browse_b)
        file_menu.addAction(choose_b_button)

        # choose model for prediction
        choose_m_button = QAction('Choose model file', self)
        choose_m_button.setShortcut('Ctrl+M')
        choose_m_button.setStatusTip('Choose model file')
        choose_m_button.triggered.connect(self.browse_m)
        file_menu.addAction(choose_m_button)

        # start or stop the animation
        pp_button = QAction('Start/Stop the animation', self)
        #pp_button.setShortcut('Space')
        pp_button.setStatusTip('Start/Stop the animation')
        #pp_button.clicked.connect(self.start_stop)
        pp_button.triggered.connect(self.start_stop)
        pp_button.toggle()
        file_menu.addAction(pp_button)
        
        # exit button action
        exit_button = QAction('Exit', self)
        exit_button.setShortcut('Ctrl+Q')
        exit_button.setStatusTip('Exit application')
        exit_button.triggered.connect(self.close)
        file_menu.addAction(exit_button)

    def __buttons(self):
        """Establishment of the front-end and back-end buttons bar.

        Classically, the available buttons are:
                - Choose timestack file
                - Choose bathymetry file
                - Choose model file
                - Play/Pause
                - Exit

        """
        maxw = self.width() - self.__b_width + self.__b_shift  		# Max width
        maxh = self.height() - self.__m_height - self.__b_height  	# Max height
        right_b_width = self.__b_width - self.__b_shift  			# Button width

        # choose timestack file button
        choose_ts_btn = QPushButton('Choose timestack file', self)
        choose_ts_btn.setToolTip('Choose timestack file')
        choose_ts_btn.clicked.connect(self.browse_ts)
        choose_ts_btn.resize(right_b_width, self.__b_height)
        choose_ts_btn.move(maxw, self.menuBar().height())

        # choose expected bathymetry file button
        choose_b_btn = QPushButton('Choose expected bathymetry file', self)
        choose_b_btn.setToolTip('Choose expected bathymetry file')
        choose_b_btn.clicked.connect(self.browse_b)
        choose_b_btn.resize(right_b_width, self.__b_height)
        choose_b_btn.move(maxw, self.menuBar().height() + self.__b_height)

        # choose weight file button
        choose_m_btn = QPushButton('Choose model file', self)
        choose_m_btn.setToolTip('Choose model file')
        choose_m_btn.clicked.connect(self.browse_m)
        choose_m_btn.resize(right_b_width, self.__b_height)
        choose_m_btn.move(maxw, self.menuBar().height() + 2 * self.__b_height)

        # play/pause wawe animation from the timestack
        self.pp_pbtn = QPushButton('Start/Stop', self)
        self.pp_pbtn.setToolTip('Start/Stop')
        self.pp_pbtn.setCheckable(True)
        self.pp_pbtn.toggle()
        self.pp_pbtn.clicked.connect(self.start_stop)
        self.pp_pbtn.resize(right_b_width, self.__b_height)
        self.pp_pbtn.move(maxw, self.menuBar().height() + 4 * self.__b_height)
        self.pp_pbtn.setShortcut('Space')


        # close button
        close_btn = QPushButton('Exit', self)
        close_btn.setToolTip('Exit Application')
        close_btn.clicked.connect(self.close)
        close_btn.resize(right_b_width, self.__b_height)
        close_btn.move(maxw, maxh)

    def start_stop(self):
        """Action of the Start/Stop button.
		
		Description: start or stop the wawe animation depending on the state of the 
					 button and update the display in consequence.
		
		"""
        self.timer = self.b_canvas.fcqt.new_timer(1, [(self.b_canvas.update_ts, (), {})])
        if self.pp_pbtn.isChecked():
			# Stop the animation
            self.pp_pbtn.setText('Start')
            self.pp_pbtn.setCheckable(False)
            self.timer.stop()
            self.b_canvas.plot(bath_path=self.bath)
        else:
			# Start the animation
            self.pp_pbtn.setText('Stop')
            self.pp_pbtn.setCheckable(True)
            self.b_canvas.c_time = time()
            self.timer.start()

    def browse_ts(self):
        """Action of the Choose timestack button.
		
		Description: browse the selected timestack file and update the left canvas and wawes 
					 animation.
		
		"""
		# Get the timestack filename
        filename = QFileDialog.getOpenFileName(None, 'Find Timestack',
                                               '',
                                               '(*.npy)')[0]
        if filename:
			# Update the ts_canvas and b_canvas
            self.ts = filename
            self.ts_canvas.plot(n_ts=-1, ts_path=self.ts)
            self.b_canvas.ts = np.load(self.ts)

    def browse_b(self):
        """Action of the Choose bathymetry button.
		
		Description: browse the selected bathymetry file and update the middle canvas.
		
		"""
		# Get the bathymetry filename
        filename = QFileDialog.getOpenFileName(None, 'Find Bathymetry',
                                               '',
                                               '(*.npy)')[0]
        if filename:
			# Update the b_canvas
            self.bath = filename
            self.b_canvas.plot(bath_path=self.bath)

    def browse_m(self, ae=False):
        """Action of the Choose model button.
		
		Description: browse the selected encoder model and the selected cnn model and then
					 update the predicted bathymetry in consequence.
		
		"""
		# Get the encoder filename
        encoder_filename = QFileDialog.getOpenFileName(None, 'Find trained encoder',
                                               'src/saves/weights/',
                                               '(*.h5)')[0]
                                               
        # Get the encoder model name and version
        encoder_path = encoder_filename.split('.')[0]
        encoder_name = encoder_path.split('/')[-1]
        encoder_version = int(encoder_filename.split('.')[1])
        
        # Get the cnn model filename
        model_filename = QFileDialog.getOpenFileName(None, 'Find trained model',
                                               'src/saves/weights/',
                                               '(*.h5)')[0]
                                               
        # Get the cnn model name and version
        model_path = model_filename.split('.')[0]
        model_name = model_path.split('/')[-1]
        model_version = int(model_filename.split('.')[1])
        
        if encoder_filename and model_filename:
			# Create the encoder and the cnn
            ae1 = AutoEncoder(load_models=encoder_name, version=encoder_version)
            cnn1 = CNN(load_models=model_name, version=model_version)
			
			# Adjust the timestack shape
            ts_origi = self.b_canvas.ts[200:]	#croping
            width, height = ts_origi.shape
            ts_origi = np.array([ts_origi])
			
			# Predict the encoded the timestack (= encode the timestack)
            ts_enc = ae1.predict(ts_origi.reshape(len(ts_origi), 
												  width, 
												  height, 
												  1), 
								 batch_size=1)
            a ,width, height = ts_enc.shape
            ts_enc = np.array([ts_enc])
            
            # Predict the bathymetry
            self.b_canvas.bath_pred = cnn1.predict(ts_enc.reshape(len(ts_enc), 
																  width, 
																  height, 
																  1), 
												   batch_size=1, 
												   smooth=True).flatten()
            
            # Update the display
            self.b_canvas.plot(bath_path=self.bath)

            



class TSCanvas(FigureCanvasQTAgg):
    """The TimeStack canvas (only a part of the Hovmöller diagram).

    The canvas is static and the size of the image is adapted to show n_ts lines
    of the diagram.

    Attributes:
        fig (matplotlib.figure.Figure): TimeStack figure.
        ax (matplotlib.axes.Axes): Axes of the TimeStack figure.

    """

    def __init__(self, parent=None, width=800, height=600, dpi=100):
        """Initialization of the Time-Stack canvas.

        It makes a bridge between Matplotlib and the window, creates the figure
        to display and then show a part of the TimeStack.

        Args:
            parent (QWidget): Parent of the canvas (default: None).
            width (int): Width of the canvas (default: 800).
            height (int): Height of the canvas (default: 600).
            dpi (int): Number of pixel per inch of the canvas (default: 100).

        Examples:
            On a pre-defined window.

            >>> ts_canvas = TSCanvas()
            >>> ts_canvas.move(0, 0)

        """
        self.fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        self.ax = self.fig.add_subplot(111)

        FigureCanvasQTAgg.__init__(self, self.fig)
        FigureCanvasQTAgg.setSizePolicy(self,
                                        QSizePolicy.Expanding,
                                        QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
        self.setParent(parent)
        self.plot()  # matplotlib canvas

    def plot(self, n_ts=-1, ts_path='src/graphics/TS_init.npy'):
        """Drawing of the Hovmöller diagram.

        Args:
			n_ts (int): TimeStack time window to display (default: -1).
            ts_path (str): Path of the timestack to plot
				(default: 'src/graphics/TS_init.npy').

        """
        self.ax.clear()
        ts_fig = self.ax.imshow(np.load(ts_path)[:n_ts])
        # self.fig.colorbar(ts_fig)
        self.ax.set_title('TimeStack : ' + ts_path.split('/')[-1].split('.')[0])
        self.draw()


class BCanvas(FigureCanvasQTAgg):
    """The Bathymetry canvas (static and dynamic).

    The canvas can be either static of dynamic, depending on the Start/Stop
    button's state

    Attributes:
        fig (matplotlib.figure.Figure): TimeStack figure.
        fcqt (matplotlib.backends.backend_qt5agg.FigureCanvasQT): the PyQT
            canvas for Matplotlib.
        ax (matplotlib.axes.Axes): Axes of the TimeStack figure.

    """

    def __init__(self, parent=None, width=800, height=600, dpi=100,
                 bath_path='src/graphics/B_init.npy', ts_path='src/graphics/TS_init.npy'):
        """Initialization of the Bathymetry canvas.

        It makes a bridge between Matplotlib and the window, creates the figure
        to display and then show statically or dynamically the figure.

        Args:
            parent (QWidget): Parent of the canvas (default: None).
            width (int): Width of the canvas (default: 800).
            height (int): Height of the canvas (default: 600).
            dpi (int): Number of pixel per inch of the canvas (default: 100).
            bath_path (str): Path of the expected bathymetry to plot
				(default: 'src/graphics/B_init.npy').
            ts_path (str): Path of the timestack to plot
                (default: 'src/graphics/TS_init.npy').

        Examples:
            On a pre-defined window.

            >>> b_canvas = BCanvas()
            >>> b_canvas.move(0, 0)

        """
        self.fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        self.fcqt = FigureCanvasQT(self.fig)
        self.ax = self.fcqt.figure.subplots()

        FigureCanvasQTAgg.__init__(self, self.fcqt.figure)
        FigureCanvasQTAgg.setSizePolicy(self,
                                        QSizePolicy.Expanding,
                                        QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
        self.setParent(parent)
		
        self.bath_path = bath_path
        self.bath = np.load(bath_path)  		# bathymetry vector
        self.bath_pred = np.load(bath_path)  	# bathymetry vector
        self.ts = np.load(ts_path)  			# timestack matrix
        self.n_time = len(self.ts)  			# size of time window
        self.n_wave = len(self.ts[0])  			# distance to the horizon
        self.n_bath = len(self.bath)  			# size of the bathymetry grid
        
        self.plot()

    def update_ts(self):
        """Drawing of the expected and predict bathymetry and the wawes.
        
        """
        wave = self.ts[int(10 * (time()-self.c_time)) % self.n_time]

        self.ax.clear()
        self.ax.plot(self.bath, color='xkcd:chocolate', label='Exp. Bath.')
        self.ax.plot(self.bath_pred, '--', color='xkcd:chocolate',
                     label='Pred. Bath.')
        self.ax.fill_between(range(self.n_wave),
                             wave,
                             self.bath[:self.n_wave],
                             facecolor='xkcd:azure')
        self.ax.fill_between(range(self.n_bath),
                             min(self.bath),
                             self.bath,
                             facecolor='orange')
        self.__infos()

        self.ax.figure.canvas.draw()

    def plot(self, bath_path='src/graphics/B_init.npy'):
        """Drawing of the expected and predict bathymetry a flat water level.
        
        Args:
			n_ts (int): TimeStack time window to display (default: -1).
            bath_path (str): Path of the expected bathymetry
				(default: 'src/graphics/B_init.npy').

        """
        
        self.ax.clear()
        self.bath_path = bath_path
        self.bath = np.load(bath_path)
        self.n_time = len(self.ts)  	# size of time window
        self.n_wave = len(self.ts[0])  	# distance to the horizon
        self.n_bath = len(self.bath)  	# size of the bathymetry grid
        self.ax.plot(self.bath, color='xkcd:chocolate', label='Exp. Bath.')
        self.ax.plot(self.bath_pred, '--', color='xkcd:chocolate',
                     label='Pred. Bath.')
        self.ax.fill_between(range(self.n_wave),
                             0,
                             self.bath[:self.n_wave],
                             facecolor='xkcd:azure')
        self.ax.fill_between(range(self.n_bath),
                             min(self.bath),
                             self.bath,
                             facecolor='orange')
        self.__infos()

        self.draw()

    def __infos(self):
        """Creation of the information of the canvas.
        
        """
        self.ax.set_xlabel(r'Horizon distance $(m)$')
        self.ax.set_ylabel(r'Depth $(m)$')
        self.ax.set_xlim((0, self.n_bath - 1))
        self.ax.set_ylim((np.min(self.bath), np.max(self.ts)))
        self.ax.legend()
        self.ax.grid()
        self.ax.set_title(r'Expected and Predicted Bathymetry : ' + self.bath_path.split('/')[-1].split('.')[0])

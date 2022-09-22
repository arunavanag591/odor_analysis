# import matplotlib and numpy as usual
#plots
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#plots
import figurefirst
from figurefirst import FigureLayout,mpl_functions

# import matplotlib.ticker as mtick
# import pylab as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable # for colorbar
# import seaborn as sns
# from matplotlib.ticker import MaxNLocator


# import numpy as np
# import pandas as pd
# now import pylustrator
import pylustrator

# activate pylustrator
# pylustrator.start()

# METHODS FIGURE 01 _  Desert

# img1 = mpimg.imread('../../Figure/Paper/Setup.jpg')
# img2 = mpimg.imread('../../Figure/Paper/SetupDesert.png')
# img3 = mpimg.imread('../../Figure/Paper/DesertSensorSetup.jpeg')
# img4 = mpimg.imread('../../Figure/Paper/OCHist.jpeg')
# img5 = mpimg.imread('../../Figure/Paper/DesertWind.jpeg')
# img6 = mpimg.imread('../../Figure/Paper/OdorTs.jpeg')
# plt.figure(1)
# plt.subplot(161)
# plt.imshow(img1)
# plt.axis('off')

# plt.subplot(162)
# plt.imshow(img2)
# plt.axis('off')

# plt.subplot(163)
# plt.imshow(img3)
# plt.axis('off')

# plt.subplot(164)
# plt.imshow(img4)
# plt.axis('off')

# plt.subplot(165)
# plt.imshow(img5)
# plt.axis('off')

# plt.subplot(166)
# plt.imshow(img6)
# plt.axis('off')

# # f,(ax1,ax2)=plt.subplots(1,2)
# # ax1.imshow(img2)
# # ax2.imshow(img1)
# # ax1.axis('off')
# # ax2.axis('off')
# #% start: automatic generated code from pylustrator
# plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
# import matplotlib as mpl
# plt.figure(1).set_size_inches(20.320000/2.54, 10.160000/2.54, forward=True)
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)
# plt.figure(1).axes[0].set_position([0.035290, 0.052116, 0.222603, 0.445532])
# plt.figure(1).axes[0].annotate('New Annotation', (-0.5, 3152.5), (1576.0, 1576.0), arrowprops=dict(arrowstyle='->'))  # id=plt.figure(1).axes[0].texts[0].new
# plt.figure(1).axes[0].texts[0].set_fontsize(5)
# plt.figure(1).axes[0].texts[0].set_position([174.396955, 629.979846])
# plt.figure(1).axes[0].texts[0].set_text("Odor Sensor")
# plt.figure(1).axes[0].texts[0].xy = (554.023209, 960.328183)
# plt.figure(1).axes[0].annotate('New Annotation', (-0.5, 3152.5), (1576.0, 1576.0), arrowprops=dict(arrowstyle='->'))  # id=plt.figure(1).axes[0].texts[1].new
# plt.figure(1).axes[0].texts[1].set_fontsize(5)
# plt.figure(1).axes[0].texts[1].set_position([1283.527486, 629.979846])
# plt.figure(1).axes[0].texts[1].set_text("GPS Antenna")
# plt.figure(1).axes[0].texts[1].xy = (1685.250554, 1410.167556)
# plt.figure(1).axes[0].annotate('New Annotation', (-0.5, 3152.5), (1576.0, 1576.0), arrowprops=dict(arrowstyle='->'))  # id=plt.figure(1).axes[0].texts[2].new
# plt.figure(1).axes[0].texts[2].set_fontsize(5)
# plt.figure(1).axes[0].texts[2].set_position([2699.587220, 629.979846])
# plt.figure(1).axes[0].texts[2].set_text("IMU")
# plt.figure(1).axes[0].texts[2].xy = (2816.477900, 1437.961809)
# plt.figure(1).axes[1].set_position([0.040600, 0.458661, 0.214698, 0.468530])
# plt.figure(1).axes[1].annotate('New Annotation', (-0.5, 861.5), (394.5, 430.5), arrowprops=dict(arrowstyle='->'))  # id=plt.figure(1).axes[1].texts[0].new
# plt.figure(1).axes[1].texts[0].set_color("#ffffffff")
# plt.figure(1).axes[1].texts[0].set_fontsize(5)
# plt.figure(1).axes[1].texts[0].set_position([185.448637, 817.241547])
# plt.figure(1).axes[1].texts[0].set_text("Stationery Wind Sensor")
# plt.figure(1).axes[1].texts[0].set_weight("bold")
# plt.figure(1).axes[1].texts[0].xy = (116.208405, 640.128865)
# plt.figure(1).axes[1].annotate('New Annotation', (-0.5, 861.5), (394.5, 430.5), arrowprops=dict(arrowstyle='->'))  # id=plt.figure(1).axes[1].texts[1].new
# plt.figure(1).axes[1].texts[1].set_color("#ffffffff")
# plt.figure(1).axes[1].texts[1].set_fontsize(5)
# plt.figure(1).axes[1].texts[1].set_position([115.375010, 154.117813])
# plt.figure(1).axes[1].texts[1].set_text("Stationery GPS/RTK Station")
# plt.figure(1).axes[1].texts[1].set_weight("bold")
# plt.figure(1).axes[1].texts[1].xy = (363.519073, 330.205747)
# plt.figure(1).axes[1].annotate('New Annotation', (-0.5, 861.5), (394.5, 430.5), arrowprops=dict(arrowstyle='->'))  # id=plt.figure(1).axes[1].texts[2].new
# plt.figure(1).axes[1].texts[2].set_color("#ffffffff")
# plt.figure(1).axes[1].texts[2].set_fontsize(5)
# plt.figure(1).axes[1].texts[2].set_position([388.291677, 245.762211])
# plt.figure(1).axes[1].texts[2].set_text("Odor Outlet")
# plt.figure(1).axes[1].texts[2].set_weight("bold")
# plt.figure(1).axes[1].texts[2].xy = (494.125395, 361.162557)
# plt.figure(1).axes[2].set_xlim(-0.5, 1800.5)
# plt.figure(1).axes[2].set_xticks([0.0, 500.0, 1000.0, 1500.0])
# plt.figure(1).axes[2].set_xticklabels(["", "", "", ""], fontsize=10)
# plt.figure(1).axes[2].set_label("")
# plt.figure(1).axes[2].set_position([0.299963, 0.571348, 0.173120, 0.342011])
# plt.figure(1).axes[2].get_xaxis().get_label().set_text("Longitude,m")
# plt.figure(1).axes[2].get_yaxis().get_label().set_text("Latitude,m")
# plt.figure(1).axes[3].set_position([0.313713, 0.029571, 0.146885, 0.430137])
# plt.figure(1).axes[3].annotate('New Annotation', (-0.5, 1778.5), (607.0, 889.0), arrowprops=dict(arrowstyle='->'))  # id=plt.figure(1).axes[3].texts[0].new
# plt.figure(1).axes[3].texts[0].set_fontsize(5)
# plt.figure(1).axes[3].texts[0].set_position([944.286276, 1576.879105])
# plt.figure(1).axes[3].texts[0].set_rotation(90.0)
# plt.figure(1).axes[3].texts[0].set_text("Odor Threshold > 4.5V")
# plt.figure(1).axes[3].texts[0].xy = (585.165980, 1180.810006)
# plt.figure(1).axes[4].set_position([0.500587, 0.317968, 0.443658, 0.659476])
# plt.figure(1).axes[5].set_position([0.500587, 0.008884, 0.397656, 0.312825])
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[0].new
# plt.figure(1).texts[0].set_fontsize(5)
# plt.figure(1).texts[0].set_ha("center")
# plt.figure(1).texts[0].set_position([0.304849, 0.711676])
# plt.figure(1).texts[0].set_rotation(90.0)
# plt.figure(1).texts[0].set_text("Latitude, m")
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
# plt.figure(1).texts[1].set_fontsize(5)
# plt.figure(1).texts[1].set_ha("center")
# plt.figure(1).texts[1].set_position([0.386523, 0.558399])
# plt.figure(1).texts[1].set_text("Longitude, m")
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
# plt.figure(1).texts[2].set_fontsize(5)
# plt.figure(1).texts[2].set_ha("center")
# plt.figure(1).texts[2].set_position([0.387310, 0.475076])
# plt.figure(1).texts[2].set_text("Odor Historgram")
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
# plt.figure(1).texts[3].set_fontsize(5)
# plt.figure(1).texts[3].set_position([0.331387, 0.448799])
# plt.figure(1).texts[3].set_rotation(0.0)
# plt.figure(1).texts[3].set_text("WS > 3.5m/s")
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[4].new
# plt.figure(1).texts[4].set_fontsize(5)
# plt.figure(1).texts[4].set_position([0.331387, 0.247339])
# plt.figure(1).texts[4].set_text("WS < 3.5m/s")
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[5].new
# plt.figure(1).texts[5].set_fontsize(5)
# plt.figure(1).texts[5].set_ha("center")
# plt.figure(1).texts[5].set_position([0.304849, 0.205348])
# plt.figure(1).texts[5].set_rotation(90.0)
# plt.figure(1).texts[5].set_text("Frequency,  hz")
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[6].new
# plt.figure(1).texts[6].set_fontsize(5)
# plt.figure(1).texts[6].set_ha("center")
# plt.figure(1).texts[6].set_position([0.387156, 0.010845])
# plt.figure(1).texts[6].set_text("Odor Concentration, v")
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[7].new
# plt.figure(1).texts[7].set_fontsize(9)
# plt.figure(1).texts[7].set_position([0.014974, 0.920276])
# plt.figure(1).texts[7].set_text("A")
# plt.figure(1).texts[7].set_weight("bold")
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[8].new
# plt.figure(1).texts[8].set_fontsize(9)
# plt.figure(1).texts[8].set_position([0.014974, 0.408559])
# plt.figure(1).texts[8].set_text("B")
# plt.figure(1).texts[8].set_weight("bold")
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[9].new
# plt.figure(1).texts[9].set_fontsize(9)
# plt.figure(1).texts[9].set_ha("center")
# plt.figure(1).texts[9].set_position([0.293553, 0.920276])
# plt.figure(1).texts[9].set_text("C")
# plt.figure(1).texts[9].set_weight("bold")
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[10].new
# plt.figure(1).texts[10].set_fontsize(9)
# plt.figure(1).texts[10].set_position([0.501461, 0.920276])
# plt.figure(1).texts[10].set_text("D")
# plt.figure(1).texts[10].set_weight("bold")
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[11].new
# plt.figure(1).texts[11].set_fontsize(9)
# plt.figure(1).texts[11].set_position([0.293521, 0.408559])
# plt.figure(1).texts[11].set_text("E")
# plt.figure(1).texts[11].set_weight("bold")
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[12].new
# plt.figure(1).texts[12].set_fontsize(9)
# plt.figure(1).texts[12].set_position([0.501461, 0.305556])
# plt.figure(1).texts[12].set_text("F")
# plt.figure(1).texts[12].set_weight("bold")
# plt.figure(1).texts[13].set_visible(False)
#% end: automatic generated code from pylustrator
# plt.show()
# mpl_functions.set_fontsize(plt.figure(1), 5)
# plt.figure(1).savefig('../../Figure/Paper/MethodsFigureExported_.png', dpi=300, bbox_inches = "tight")
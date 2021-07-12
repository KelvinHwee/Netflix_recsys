# import pandas as pd
# import numpy as np
# import seaborn as sns
#
# x = list(np.arange(10))
# y = list(np.arange(10))
#
# sns.scatterplot(x,y)

import plotly_express as px
from plotly.offline import plot
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF

import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
fig.show()


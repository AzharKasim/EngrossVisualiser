import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import os, re
import datetime as dt

def findFile():
    currentFolder = os.listdir()

    pattern= r'sessions_lifetime_\d\d\d\d\d\d\d\d-\d\d\d\d\d\d.csv'
    fileFormat = re.compile(pattern)

    for items in currentFolder:
        if items.endswith('.csv'):
            m = fileFormat.match(items)
            if m != None:
                file = m.group()
                return file
    raise ValueError("File Not Found")

def get_currentMonth():
    today = dt.date.today()
    month = today.strftime('%Y-%m')
    return month


def set_dataFrame(file, choice):

    colns = ['Date','Time','Label','Duration(Minutes)']

    df = pd.read_csv(file, usecols=colns)

    df = df.dropna()
    df['date_time'] = df.loc[:,'Date'] + " " + df.loc[:,'Time']
    df['date_time'] = pd.to_datetime(df['date_time'])
    df.set_index(df['date_time'], inplace=True)
    df.drop(['Date','Time', 'date_time'], axis=1, inplace=True)
    df['Duration'] = df['Duration(Minutes)'].apply(lambda x: int(round(x/60,0)))

    df = set_dataFrameTime(df, choice)
    weekly_df = df.resample('W').sum() #['2019-12':]
    monthly_df = df.resample('M').sum() #['2019-12':]

    grouped_labels = df.groupby('Label')
    df_label = grouped_labels.sum()
    df_label.sort_values(by='Duration', ascending=True, inplace=True)
    return weekly_df, monthly_df, df_label



def set_dataFrameTime(df, choice):
    try:
        if choice == 1:     # lifetime
            return df
        elif choice == 2:   # 6 months
            return df['2019-12':'2020-05']
        elif choice == 3:   # This month
            month = get_currentMonth()
            return df[month]
        elif choice == 4:   # Last month
            return df['2020-04']
        elif choice == 5:   # Today
            return df['2020-05-15']
        elif choice == 6:   # yesterday
            return df['2020-05-14']
    except KeyError as error:
        print(f"KeyError: {error}")
        print(f"Did you download the latest Engross data?")
        print("Could not get time period of your choice.")
        print("Returning lifetime data.")
        return df


def plot_MonthlyDf(df, fig, gs):

    X = convert_monthlyMplDates(df)
    Y = df['Duration'].values

    ax = fig.add_subplot(gs[0,0])

    BarGraph = ax.bar(X, Y, color='salmon', edgecolor='black', width=25)

    ax.set_title('Monthly Total', fontsize='16')
    ax.set_ylim(ymin=0,ymax=max(Y)+15)

    labelBarGraph(BarGraph, ax)

    locator = mdates.MonthLocator()
    formatter = mdates.DateFormatter('%b %y')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

def convert_monthlyMplDates(df):
    dateIndex_np = df.index.values
    dateIndex_MonthOnly = np.array(
                    [np.datetime64(date,'M') for date in dateIndex_np])
    mplDates = mdates.date2num(dateIndex_MonthOnly)
    return mplDates

def labelBarGraph(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

def convert_weeklyMPLDates(df):
    dateIndex_np = df.index.values
    mplDates = mdates.date2num(dateIndex_np)
    return mplDates

def labelStemGraph(xValues,yValues, ax):
    for x,y in zip(xValues,yValues):
        ax.annotate(f"{y}", xy=(x, y), xytext = (0,5),
                    xycoords = 'data',
                    textcoords='offset points',
                    ha='center', va='bottom')

def plot_WeeklyDf(df, fig, gs):

    X = convert_weeklyMPLDates(df)
    Y = df['Duration']

    ax = fig.add_subplot(gs[1,0])

    StemGraph = ax.stem(X, Y, linefmt='salmon', markerfmt='oc', basefmt='grey',
            use_line_collection=True)

    labelStemGraph(X, Y, ax)

    locator = mdates.WeekdayLocator(byweekday=SU, interval=4)
    formatter = mdates.DateFormatter('%d %b')

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_title('Weekly Total', fontsize='16')
    ax.set_ylim(ymin=0, ymax= (max(Y)+10))
    ax.tick_params(labelrotation =90.0)

    h_center = np.median(Y)
    h_center = np.full((len(X),1),h_center)
    ax.plot(X, h_center, color='salmon', alpha=0.5, linestyle="--")

def plot_LabelDf(df, fig, gs):
    X = df.index
    Y = df['Duration'].values

    ax = fig.add_subplot(gs[:,-1])

    BarhGraph = ax.barh(X,Y, color='salmon', edgecolor='black')
    ax.set_title('How did I spend my time?', fontsize='16')
    ax.set_xlabel('Hours', fontsize='16')
    ax.set_xlim(xmin=0, xmax=max(Y)+15)

    labelHbar(BarhGraph, ax)

def labelHbar(BarhGraph, ax):
    for bar in BarhGraph:
        value = bar.get_width()
        height_bar = bar.get_height()

        if value == 0: continue

        ax.annotate(f"{value}",
                    xy=(value ,bar.get_y() + height_bar/2),
                    xytext=(3,-4),
                    xycoords='data',
                    textcoords='offset points')

def plot_Visual(weekly_df, monthly_df, label_df):
    fig = plt.figure(num=1, figsize=(18,9))
    gs = GridSpec(2, 2, figure=fig, wspace=0.4)

    plot_WeeklyDf(weekly_df, fig, gs)
    plot_MonthlyDf(monthly_df, fig, gs)
    plot_LabelDf(label_df, fig, gs)

    fig.suptitle('Engross Visualiser',
                  x=0.5, y=0.96,
                  fontsize='23',
                  fontweight= 'light')

    fig.text(0.085, 0.5, 'Hours',
             va='center', ha='center',
             rotation='vertical',
             fontsize= 18)
    return fig

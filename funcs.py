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

    """
    Returns the latest csv file as a string.

    Notes
    -----
    List of current files in folder is sorted to place
    latest files in front.
    The for loop will stop after finding the first matched
    file.

    """

    currentFolder = os.listdir()
    currentFolder.sort(reverse=True)

    pattern= r'sessions_lifetime_\d\d\d\d\d\d\d\d-\d\d\d\d\d\d.csv'
    fileFormat = re.compile(pattern)

    for items in currentFolder:
        if items.endswith('.csv'):
            m = fileFormat.match(items)
            if m != None:
                file = m.group()
                return file
    raise ValueError("File Not Found")


def set_DataFrame(file, choice):

    """Set up the dataframe to retrieve values to plot.


    Notes
    ------
    - Any missing value is dropped from the dataset.
    - Dataset is resampled into monthly, weekly timeseries datasets.
    - A groupby is also used to see time spent on different labels.

    Parameters
    ----------
    file: csv
        The data in Engross is exported into a csv file.
    choice: Int
        The int that is mapped to a time period chosen by the user to display.

    Helper Functions
    ----------------
    1. set_DataFrameTime

    Returns
    -------
    3 dataframes
    weekly_df: DataFrame
    monthly_df: DataFrame
    label_df: DataFrame
    """

    colns = ['Date','Time','Label','Duration(Minutes)']

    df = pd.read_csv(file, usecols=colns)

    df = df.dropna()
    df['date_time'] = df.loc[:,'Date'] + " " + df.loc[:,'Time']
    df['date_time'] = pd.to_datetime(df['date_time'])
    df.set_index(df['date_time'], inplace=True)
    df.drop(['Date','Time', 'date_time'], axis=1, inplace=True)
    df['Duration'] = df['Duration(Minutes)'].apply(lambda x: int(round(x/60,0)))

    df = set_DataFrameTime(df, choice)
    weekly_df = df.resample('W').sum()
    monthly_df = df.resample('M').sum()

    grouped_labels = df.groupby('Label')
    label_df = grouped_labels.sum()
    label_df.sort_values(by='Duration', ascending=True, inplace=True)
    return weekly_df, monthly_df, label_df


def set_DataFrameTime(df, choice):

    """Returns the DataFrame based to extract values from based on time period
    that user wants to view.

    Parameters
    -----------
    df: DataFrame
        The Engross data(csv) is converted into a DataFrame that can be used
        to plot data or sliced. This is the DataFrame after unecessary data has
        been removed.
    choice: Int
        The int that is mapped to a time period chosen by the user to display.

    Returns
    --------
    df[TIME]: DataFrame
        DataFrame is indexed or sliced depending on user choice.

    Helper Functions
    -----------------
    1. get_currentMonth()
    2. get_previousMonth()
    3. get_lastSixMonths()
    4. get_today()
    5. get_yesterday()
    6. print_timePeriod(choice)

    """

    try:
        if choice == 1:     # lifetime
            print_timePeriod(choice)
            return df
        elif choice == 2:   # 6 months
            sixMonths = get_lastSixMonths()
            thisMonth = get_currentMonth()
            print_timePeriod(choice)
            return df[sixMonths:thisMonth]
        elif choice == 3:   # This month
            month = get_currentMonth()
            print_timePeriod(choice)
            return df[month]
        elif choice == 4:   # Last month
            lastMonth = get_previousMonth()
            print_timePeriod(choice)
            return df[lastMonth]
        elif choice == 5:   # Today
            today = get_today()
            print_timePeriod(choice)
            return df[today]
        elif choice == 6:   # yesterday
            yesterday = get_yesterday()
            print_timePeriod(choice)
            return df[yesterday]

    except KeyError as error:
        print(f"KeyError: {error}")
        print(f"Did you download the latest Engross data?")
        print("Could not get time period of your choice.")
        print("Returning lifetime data.")
        return df


def get_currentMonth():
    today = dt.date.today()
    month = today.strftime('%Y-%m')
    return month


def get_previousMonth():
    today = dt.date.today()
    month = today.month
    prevMonth = (today.month - 1) % 12
    lastMonth = dt.date(today.year, prevMonth, 1).strftime('%Y-%m')
    return lastMonth


def get_lastSixMonths():
    today = dt.date.today()
    year = today.year
    month = today.month
    sixMonthsAgo = (today.month - 6) % 12 + 1

    if month < 6: year -= 1
    if sixMonthsAgo == 0: sixMonthsAgo = 12

    lastSixMonths = dt.date(year, sixMonthsAgo, 1).strftime('%Y-%m')
    return lastSixMonths


def get_today():
    today = dt.date.today().strftime('%Y-%m-%d')
    return today


def get_yesterday():
    today = dt.date.today()
    yesterdayNum = today.day - 1
    yesterday = dt.date(today.year, today.month, yesterdayNum).strftime('%Y-%m-%d')
    return yesterday


def print_timePeriod(choice):
    table = {1: 'Lifetime',2:'Last 6 Months', 3: 'This Month',
          4: 'Last Month', 5: 'Today', 6: 'Yesterday'}
    tpChosen = table[choice]
    print(f'Time Period Chosen: {tpChosen}')


def plot_Visual(weekly_df, monthly_df, label_df):
    """Returns a Figure with all the plots.

    Parameters
    ----------
    weekly_df: DataFrame
    monthly_df: DataFrame
    label_df: DataFrame

    Helper Functions
    ----------------
    1. plot_WeeklyDf
    2. plot_MonthlyDf
    3. plot_LabelDf

    """
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


def plot_MonthlyDf(df, fig, gs):

    """Plots a bar graph or stem graph of the monthly data at the top left hand corner of the figure.

       Notes
       -----
       If the dataset is larger than a year, a stem graph is plotted.
       Else, a bar graph is shown instead.

       Parameters
       -----------
       df: DataFrame
        This DataFrame refers to 'monthly_df' that was returned from
        set_DataFrame.
       fig: Figure
       gs: GridSpec

       Helper Functions
       ----------------
       1. convert_monthlyMplDates
       2. label_stemGraph
       3. label_barGraph


    """

    X = convert_monthlyMplDates(df)
    Y = df['Duration'].values

    ax = fig.add_subplot(gs[0,0])

    if len(X) > 12:
        StemGraph = ax.stem(X, Y, linefmt='salmon', markerfmt='oc',
                            basefmt='grey', use_line_collection=True)

        label_stemGraph(X, Y, ax)
        locator = mdates.MonthLocator(interval=3)

    else:
        BarGraph = ax.bar(X, Y, color='salmon', edgecolor='black', width=25)

        label_barGraph(BarGraph, ax)
        locator = mdates.MonthLocator()

    ax.set_title('Monthly Total', fontsize='16')
    ax.set_ylim(ymin=0,ymax=max(Y)+15)

    formatter = mdates.DateFormatter('%b %y')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def convert_monthlyMplDates(df):
    dateIndex_np = df.index.values
    dateIndex_MonthOnly = np.array(
                    [np.datetime64(date,'M') for date in dateIndex_np])
    mplDates = mdates.date2num(dateIndex_MonthOnly)
    return mplDates


def label_barGraph(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


def label_stemGraph(xValues,yValues, ax):
    for x, y in zip(xValues, yValues):
        if y == 0: continue
        ax.annotate(f"{y}", xy=(x, y), xytext = (0,5),
                    xycoords = 'data',
                    textcoords='offset points',
                    ha='center', va='bottom')


def plot_WeeklyDf(df, fig, gs):

    """Plots a stem graph or 2d graph of the weekly data at the bottom
     left hand corner of the figure.
     A horizontal line showing the median is also drawn.

       Notes
       -----
       If the dataset is larger than 30 weeks, a line graph is plotted.
       Else, a stem graph is shown instead.

       Parameters
       -----------
       df: DataFrame
        This DataFrame refers to 'weekly_df' that was returned from
        set_DataFrame.
       fig: Figure
       gs: GridSpec

       Helper Functions
       ----------------
       1. convert_weeklyMplDates
       2. label_stemGraph
       3. label_barGraph

    """

    X = convert_weeklyMPLDates(df)
    Y = df['Duration']

    ax = fig.add_subplot(gs[1,0])

    if len(X) > 30:
        ax.plot(X,Y, color='salmon', linestyle='solid')
        for x, y in zip(X,Y):
            if y == max(Y):
                ax.annotate(f"{y}", xy=(x, y), xytext= (0,5),
                            xycoords = 'data',
                            textcoords = 'offset points',
                            ha='center', va='bottom')

        locator = mdates.MonthLocator(interval=2)

    else:
        StemGraph = ax.stem(X, Y, linefmt='salmon', markerfmt='oc',
                        basefmt='grey', use_line_collection=True)

        label_stemGraph(X, Y, ax)
        if len(X) <=  5:
            locator = mdates.WeekdayLocator(byweekday=SU)
        else:
            locator = mdates.WeekdayLocator(byweekday=SU, interval=2)

    formatter = mdates.DateFormatter('%d %b')

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_title('Weekly Total', fontsize='16')
    ax.set_ylim(ymin=0, ymax= (max(Y)+10))
    ax.tick_params(labelrotation =90.0)

    h_center = np.median(Y)
    h_center = np.full((len(X),1),h_center)
    ax.plot(X, h_center, color='salmon', alpha=0.5, linestyle="--")


def convert_weeklyMPLDates(df):
    dateIndex_np = df.index.values
    mplDates = mdates.date2num(dateIndex_np)
    return mplDates


def plot_LabelDf(df, fig, gs):
    """Plots a horizontal bar graph of the minutes per labels.

    Parameter
    ---------
    df: DataFrame
        This is DataFrame refers to 'label_df' that is returned from
        set_DataFrame.
    fig: Figure
    gs: GridSpec

    Helper Functions
    ------------------
    1. label_hBar


    """



    X = df.index
    Y = df['Duration'].values

    ax = fig.add_subplot(gs[:,-1])

    BarhGraph = ax.barh(X,Y, color='salmon', edgecolor='black')
    ax.set_title('How did I spend my time?', fontsize='16')
    ax.set_xlabel('Hours', fontsize='16')
    ax.set_xlim(xmin=0, xmax=max(Y)+15)

    label_hBar(BarhGraph, ax)


def label_hBar(BarhGraph, ax):
    for bar in BarhGraph:
        value = bar.get_width()
        height_bar = bar.get_height()

        if value == 0: continue

        ax.annotate(f"{value}",
                    xy=(value ,bar.get_y() + height_bar/2),
                    xytext=(3,-4),
                    xycoords='data',
                    textcoords='offset points')

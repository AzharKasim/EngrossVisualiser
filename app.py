import matplotlib.pyplot as plt
import funcs

def main():
    plt.style.use('ggplot')

    print("Engross Visualiser")
    print("Looking for Engross data...")

    try:

        file = funcs.findFile()

    except Exception as error:

        print("Caught this error: {}".format(repr(error)))
        print("Check if engross file is in the same folder.")
        return None

    print("Engross data found.")
    print("Processing data...")

    msg = """
          Choose time period:
          1. Lifetime
          2. Last 6 months
          3. This month
          4. Last Month
          5. Today
          6. Yesterday

          """

    userChoice = int(input(msg))
    #try:
    df_weekly, df_monthly, df_label = funcs.set_dataFrame(file, userChoice)
    # except:
    #     print("Failed to extract data from Engross file.")
    #     return None

    print("Plotting graphs...")
    funcs.plot_Visual(df_weekly, df_monthly, df_label)

    #filename = str(input("Enter filename to save as: "))

    #plt.savefig(filename)
    plt.show()

    print("Program terminated.")

if __name__ == '__main__':
    main()

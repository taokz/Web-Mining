def analyze_cars(csvfile):
    # reading the csv
    df = pd.read_csv(csvfile)
    print("cars with top 3 mpg among those of origin = 1.")
    # using sort values to finding top 3
    print(df[df['origin'] == 1].sort_values('mpg', ascending=False)[['car', 'mpg']].head(3))
    print("\nmean, min, and max mpg values for each of these brands: ford, buick and honda")
    # new column name using car column
    df['brand'] = df.apply(lambda x: x.car.split()[0], axis=1)
    print("")
    print(df[df['brand'].isin(["ford", "buick", "honda"])].groupby('brand').agg(
        {'mpg': [pd.np.mean, pd.np.max, pd.np.min]}))
    print("\ncross tab to show the average mpg of each brand and each origin value")
    # printing the crosstab
    print(pd.crosstab(df['brand'], df['origin'], margins=True, values=df['mpg'], aggfunc=pd.np.average))
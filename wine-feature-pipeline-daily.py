import os
import modal

LOCAL = False

if LOCAL == False:
    stub = modal.Stub("wine_daily")
    image = modal.Image.debian_slim().pip_install(["hopsworks"])

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


def generate_wine(name, volatile_acidity_max, volatile_acidity_min, total_sulfur_dioxide_max, total_sulfur_dioxide_min,
                  ):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({"volatile_acidity": [random.uniform(volatile_acidity_max, volatile_acidity_min)],
                       "total_sulfur_dioxide": [random.uniform(total_sulfur_dioxide_max, total_sulfur_dioxide_min)],
                       })
    df['type'] = name
    return df


def get_random_wine():
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    white_df = generate_wine(
        "white", 0.1, 0.6, 50, 250)
    red_df = generate_wine(
        "red", 0.2, 1, 5, 150)

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.uniform(0, 2)
    if pick_random >= 1:
        wine_df = white_df
        print("White wine added")
    else:
        wine_df = red_df
        print("Red wine added")

    return wine_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_df = get_random_wine()

    wine_fg = fs.get_feature_group(name="wine", version=1)
    wine_fg.insert(wine_df)


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        modal.runner.deploy_stub(stub)
        f.remote()

import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model", version=2)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")


def wine(volatile_acidity, chlorides, density, alcohol):
    print("Calling function")
#     df = pd.DataFrame([[sepal_length],[sepal_width],[petal_length],[petal_width]],
    df = pd.DataFrame([[volatile_acidity, chlorides, density, alcohol]],
                      columns=['volatile_acidity', 'chlorides', 'density', 'alcohol'])
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df)
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want
    # the first element.
#     print("Res: {0}").format(res)
    print(res)
    if res == 3 or res == 4 or res == 5 or res == 6:
        wine_url = "https://upload.wikimedia.org/wikipedia/en/c/c0/Red_Wine_Glass.jpg"
    elif res == 7 or res == 8 or res == 9:
        wine_url = "https://upload.wikimedia.org/wikipedia/commons/7/71/White_Wine_Glas.jpg"
    response = requests.get(wine_url, stream=True)
    print("Content-Type:", response.headers.get('Content-Type'))
    img = Image.open(response.raw)
    return img


demo = gr.Interface(
    fn=wine,
    title="Wine Predictive Analytics",
    description="Experiment with volatile acidity and total sulfur dioxide to predict which flower it is.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=0.5, label="volatile acidity"),
        gr.inputs.Number(default=0.1, label="chlorides"),
        gr.inputs.Number(default=1.000, label="density"),
        gr.inputs.Number(default=10.0, label="alcohol"),
    ],
    outputs=gr.Image(type="pil"))

demo.launch(debug=True)

from typing import Literal
import pickle
import pandas as pd

def make_inference(model: Literal["LR", "KNN", "RF"], user_input: dict):

    model_path_dict = {
        "LR": "models/LR.pkl",
        "KNN": "models/KNN.pkl",
        "RF": "models/RF.pkl"
    }
    id2label = {
        "1": "Yes",
        "0": "No"
    }

    preprocessor_path = "preprocessor/preprocessor.pkl"
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    try:
        user_input_df = pd.DataFrame([user_input])
        preprocessed_input = preprocessor.transform(user_input_df)
        model_path = model_path_dict[model]
        model = pickle.load(open(model_path, 'rb'))
        prediction = model.predict(preprocessed_input)
    except Exception as e:
        print(f"An error occured: {e}")
        raise

    return id2label[f"{prediction[0].item()}"]

if __name__ == "__main__":
    model = "LR"
    user_input = {
        'Gender': "Male",
        'Age': 18,
        'Academic Pressure': 2,
        'CGPA': 5,
        'Study Satisfaction': 3,
        'Sleep Duration': "5-6 hours",
        'Dietary Habits': "Healthy",
        'Degree': "BCA",
        'Have you ever had suicidal thoughts ?': "Yes",
        'Work/Study Hours': 9,
        'Financial Stress': 3,
        'Family History of Mental Illness': "Yes"
    }
    print(make_inference(model, user_input))
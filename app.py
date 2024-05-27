import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# TODO: add the greek_symbol recognizer to the side bar and put these two models into a pages/ folder; also create the resume page and make it the main page

# Assuming the models are saved in the 'models' directory with filenames 'model1.pth', 'model2.pth', 'model3.pth'
MODEL_PATHS = {
    "VGG16": "artifacts/training/checkpoints/best.pt",
    "EfficientNetB0": "artifacts/training/checkpoints/best.pt",
    "InceptionV3": "artifacts/training/checkpoints/best.pt",
}

class_to_idx = {
    "adenocarcinoma": 0,
    "large.cell.carcinoma": 1,
    "normal": 2,
    "squamous.cell.carcinoma": 3,
}

idx_to_class = {
    0: 'Adenocarcinoma', 
    1: 'Large cell carcinoma', 
    2: 'Normal', 
    3: 'Squamous cell carcinoma'
}


# @st.cache_resource
def load_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model, device


# @st.cache_resource
def preprocess_image(image):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = preprocess(image)
    img = img.unsqueeze(0)  # Add batch dimension
    return img


# Cannot be hashed?
def predict(model, device, image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted = torch.argmax(probabilities, dim=1)

    return predicted.item(), probabilities.numpy().tolist()

def display_probabilities(probs, highlight_idx):
    st.write("Class Probabilities:")
    dataframe = pd.DataFrame(
        probs,
        columns=(idx_to_class.values())
    )
    st.dataframe(dataframe.style.highlight_max(axis=None), use_container_width=True, hide_index=True)


def main():
    st.title("Chest CT Scan Classification")

    # Upload image component
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Model selection dropdown
    model_name = st.selectbox("Select a model", list(MODEL_PATHS.keys()))

    if st.button("Predict"):
        if uploaded_file is None:
            st.write("Please upload an image first!")
        else:
            model_path = MODEL_PATHS[model_name]
            model, device = load_model(model_path)

            image_tensor = preprocess_image(image)

            predicted, probabilities = predict(model, device, image_tensor)

            display_probabilities(probabilities, predicted)
            st.write(f'Prediction: {idx_to_class[predicted]}')


if __name__ == "__main__":
    main()


# NOTE
# If you are pulling the same data for all users, you'd likely cache a function that retrieves that data. On the other hand,
# if you pull data specific to a user, such as querying their personal information, you may want to save that in Session State.
# That way, the queried data is only available in that one session.

import streamlit as st
import torch
import torch.nn as nn
import joblib
from PIL import Image
import numpy as np
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import pandas as pd
from datetime import datetime, timedelta

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3

# --- Define your CNN feature extractor ---
class EfficientFeatureExtractor(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b3') if pretrained else EfficientNet.from_name('efficientnet-b3')
    def forward(self, x):
        features = self.base.extract_features(x)
        out = nn.functional.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1)
        return out

# --- Define your Sensor feature extractor ---
class SensorNetFeat(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.model(x)

# --- Define your fusion model ---
class EarlyFusionModel(nn.Module):
    def __init__(self, cnn_feat_extractor, sensor_feat_extractor, img_dim, sensor_dim, num_classes=3):
        super().__init__()
        self.cnn = cnn_feat_extractor
        self.sensor = sensor_feat_extractor
        hidden = 256
        self.fusion = nn.Sequential(
            nn.Linear(img_dim + sensor_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, img, sensor):
        img_feat = self.cnn(img)
        sensor_feat = self.sensor(sensor)
        img_feat = img_feat * 3.0  # your weighting
        x = torch.cat([img_feat, sensor_feat], dim=1)
        out = self.fusion(x)
        return out

# --- Helper to infer dims ---
def infer_feature_dims(cnn_model, sensor_model, device):
    cnn_model.eval()
    sensor_model.eval()
    with torch.no_grad():
        dummy_img = torch.randn(1, 3, 224, 224, device=device)
        img_feat = cnn_model(dummy_img)
        img_dim = img_feat.shape[1]
        dummy_sensor = torch.randn(1, 4, device=device)
        sensor_feat = sensor_model(dummy_sensor)
        sensor_dim = sensor_feat.shape[1]
    return img_dim, sensor_dim

@st.cache_resource(show_spinner=False)
def load_model():
    cnn_extractor = EfficientFeatureExtractor(pretrained=False).to(DEVICE)
    cnn_checkpoint = torch.load("Banana_CNN.pth", map_location=DEVICE)
    if isinstance(cnn_checkpoint, dict) and 'model_state_dict' in cnn_checkpoint:
        cnn_extractor.base.load_state_dict(cnn_checkpoint['model_state_dict'], strict=False)
    else:
        cnn_extractor.base.load_state_dict(cnn_checkpoint, strict=False)

    sensor_feat = SensorNetFeat(input_dim=4, hidden_dim=64, output_dim=128).to(DEVICE)
    sensor_checkpoint = torch.load("banana_900_sensor_model_September.pth", map_location=DEVICE)
    sensor_feat.load_state_dict(sensor_checkpoint, strict=False)

    img_dim, sensor_dim = infer_feature_dims(cnn_extractor, sensor_feat, DEVICE)

    fusion_model = EarlyFusionModel(cnn_extractor, sensor_feat, img_dim, sensor_dim, num_classes=NUM_CLASSES).to(DEVICE)
    fusion_model.load_state_dict(torch.load("banana_early_fusion_model_September.pth", map_location=DEVICE,weights_only=False))
    fusion_model.eval()
    return fusion_model

@st.cache_resource(show_spinner=False)
def load_scaler():
    return joblib.load('sensor_scaler_September.save')

@st.cache_resource(show_spinner=False)
def load_label_encoder():
    return joblib.load('label_encoder_September.save')

fusion_model = load_model()
scaler = load_scaler()
label_encoder = load_label_encoder()

val_img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_ripeness(image, sensor_values):
    img = image.convert('RGB')
    img = val_img_transform(img).unsqueeze(0).to(DEVICE)

    sensor_np = np.array(sensor_values).reshape(1, -1).astype(np.float32)
    sensor_scaled = scaler.transform(sensor_np)
    sensor_tensor = torch.tensor(sensor_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        outputs = fusion_model(img, sensor_tensor)
        pred_class = outputs.argmax(dim=1).item()

    return pred_class

# -----------------------------------------------
# -----------------------------------------------
# --- Load regression model (already trained and saved) ---
@st.cache_resource(show_spinner=False)
def load_regression_model():
    return joblib.load("ripeness_regression_model_with_labels.pkl")

regression_model = load_regression_model()

# --- New function: predict days to rotten ---
def predict_days_to_rotten(sensor_values, predicted_label):
    # Convert into dataframe for the regression pipeline
    new_sample = pd.DataFrame([{
        "ppm1(mq4)": sensor_values[0],
        "ppm2(mq135)": sensor_values[1],
        "ppm3(tgs2602)": sensor_values[2],
        "NIR values": sensor_values[3],
        "label": predicted_label
    }])
    days_pred = regression_model.predict(new_sample)
    return int(round(days_pred[0]))

# -----------------------------------------------
# -----------------------------------------------


# Streamlit UI as before...
st.title("Banana Ripeness Prediction")


uploaded_file = st.file_uploader("Upload a banana image", type=["jpg", "jpeg", "png"])
mq4 = st.number_input("MQ4 Sensor Value", min_value=0.0, step=0.01, format="%.2f")
mq135 = st.number_input("MQ135 Sensor Value", min_value=0.0, step=0.01, format="%.2f")
tgs2602 = st.number_input("TGS2602 Sensor Value", min_value=0.0, step=0.01, format="%.2f")
nir = st.number_input("NIR Sensor Value", min_value=0.0, step=0.01, format="%.2f")

# if st.button("Predict Ripeness"):
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         sensor_values = [mq4, mq135, tgs2602,nir]

#         pred_idx = predict_ripeness(image, sensor_values)
#         pred_name = label_encoder.classes_[pred_idx]

#         st.image(image, caption="Uploaded Image", use_column_width=True)
#         st.write(f"Predicted Ripeness Class: **{pred_name}**")    
#     else:
#         st.error("Please upload an image.")


if st.button("Predict Ripeness"):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        sensor_values = [mq4, mq135, tgs2602, nir]

        # Step 1: Ripeness classification
        pred_idx = predict_ripeness(image, sensor_values)
        pred_name = label_encoder.classes_[pred_idx]

        # Step 2: Days-to-rotten regression
        predicted_days = predict_days_to_rotten(sensor_values, pred_name)
        # days_to_rotten = int(round(predicted_days[0]))
        
        # Step 3: Calculate estimated rotten date               
        rotten_date = datetime.today() + timedelta(days=predicted_days)
        rotten_date_str = rotten_date.strftime("%Y-%m-%d")
        
        # --- Show results ---
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(f"🍌 Predicted Ripeness Class: **{pred_name}**")
        # st.success(f"Predicted Days to Rotten: {days_to_rotten} days")
        st.write(f"⏳ Estimated Days to Rotten: **{predicted_days} days**")                 

        # st.success(f"Predicted Days to Rotten: {days_to_rotten} days")
        st.info(f"Estimated Date When It Will Rotten: {rotten_date_str}")
        
        
    else:
        st.error("Please upload an image.")


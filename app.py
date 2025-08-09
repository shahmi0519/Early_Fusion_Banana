import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3

# Define CNN extractor exactly as before
class EfficientFeatureExtractor(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            self.base = EfficientNet.from_pretrained('efficientnet-b3')
        else:
            self.base = EfficientNet.from_name('efficientnet-b3')
    def forward(self, x):
        features = self.base.extract_features(x)
        out = nn.functional.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1)
        return out

# Define Sensor extractor exactly as before
class SensorNetFeat(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=128):
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

# Define Fusion model as before
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
        x = torch.cat([img_feat, sensor_feat], dim=1)
        out = self.fusion(x)
        return out

@st.cache_resource
def load_model():
    # Instantiate CNN and sensor extractors
    cnn_extractor = EfficientFeatureExtractor(pretrained=False).to(DEVICE)
    sensor_feat = SensorNetFeat(input_dim=3).to(DEVICE)

    # Infer feature dims dynamically
    cnn_extractor.eval()
    sensor_feat.eval()
    with torch.no_grad():
        dummy_img = torch.randn(1, 3, 300, 300).to(DEVICE)
        img_dim = cnn_extractor(dummy_img).shape[1]
        dummy_sensor = torch.randn(1, 3).to(DEVICE)
        sensor_dim = sensor_feat(dummy_sensor).shape[1]

    # Build fusion model with inferred dims
    fusion_model = EarlyFusionModel(cnn_extractor, sensor_feat, img_dim, sensor_dim, num_classes=NUM_CLASSES).to(DEVICE)

    # Load the fusion model weights from your single checkpoint
    fusion_model.load_state_dict(torch.load("../banana_early_fusion_model.pth", map_location=DEVICE))
    # model_path = os.path.join(current_dir, '..', 'models', 'banana_early_fusion_model.pth')

    fusion_model.eval()
    return fusion_model

model = load_model()

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return preprocess(image).unsqueeze(0).to(DEVICE)

def preprocess_sensor(sensor_vals):
    sensor_tensor = torch.tensor(sensor_vals, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return sensor_tensor

st.title("Fusion Model Prediction")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

sensor1 = st.number_input("MQ4 sensor value", format="%.3f")
sensor2 = st.number_input("MQ135 sensor value", format="%.3f")
sensor3 = st.number_input("TGS2602 sensor value", format="%.3f")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

if st.button("Predict"):
    if not uploaded_file:
        st.error("Please upload an image.")
    else:
        img_tensor = preprocess_image(image)
        sensor_tensor = preprocess_sensor([sensor1, sensor2, sensor3])

        with torch.no_grad():
            outputs = model(img_tensor, sensor_tensor)
            _, pred = torch.max(outputs, 1)
            pred_class = pred.item()

        class_names = {0: "Unripe", 1: "Ripe", 2: "Rotten"}
        st.success(f"Prediction: {class_names.get(pred_class, 'Unknown')}")

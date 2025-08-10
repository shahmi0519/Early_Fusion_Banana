# # import streamlit as st
# # import torch
# # import torch.nn as nn
# # from PIL import Image
# # import torchvision.transforms as transforms
# # from efficientnet_pytorch import EfficientNet

# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # NUM_CLASSES = 3

# # # Define CNN extractor exactly as before
# # class EfficientFeatureExtractor(nn.Module):
# #     def __init__(self, pretrained=False):
# #         super().__init__()
# #         if pretrained:
# #             self.base = EfficientNet.from_pretrained('efficientnet-b3')
# #         else:
# #             self.base = EfficientNet.from_name('efficientnet-b3')
# #     def forward(self, x):
# #         features = self.base.extract_features(x)
# #         out = nn.functional.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1)
# #         return out

# # # Define Sensor extractor exactly as before
# # class SensorNetFeat(nn.Module):
# #     def __init__(self, input_dim=3, hidden_dim=64, output_dim=128):
# #         super().__init__()
# #         self.model = nn.Sequential(
# #             nn.Linear(input_dim, hidden_dim),
# #             nn.ReLU(),
# #             nn.BatchNorm1d(hidden_dim),
# #             nn.Dropout(0.2),
# #             nn.Linear(hidden_dim, output_dim),
# #             nn.ReLU()
# #         )
# #     def forward(self, x):
# #         return self.model(x)

# # # Define Fusion model as before
# # class EarlyFusionModel(nn.Module):
# #     def __init__(self, cnn_feat_extractor, sensor_feat_extractor, img_dim, sensor_dim, num_classes=3):
# #         super().__init__()
# #         self.cnn = cnn_feat_extractor
# #         self.sensor = sensor_feat_extractor
# #         hidden = 256
# #         self.fusion = nn.Sequential(
# #             nn.Linear(img_dim + sensor_dim, hidden),
# #             nn.BatchNorm1d(hidden),
# #             nn.ReLU(),
# #             nn.Dropout(0.4),
# #             nn.Linear(hidden, 128),
# #             nn.ReLU(),
# #             nn.Dropout(0.2),
# #             nn.Linear(128, num_classes)
# #         )
# #     def forward(self, img, sensor):
# #         img_feat = self.cnn(img)
# #         sensor_feat = self.sensor(sensor)
# #         x = torch.cat([img_feat, sensor_feat], dim=1)
# #         out = self.fusion(x)
# #         return out

# # @st.cache_resource
# # def load_model():
# #     # Instantiate CNN and sensor extractors
# #     cnn_extractor = EfficientFeatureExtractor(pretrained=False).to(DEVICE)
# #     sensor_feat = SensorNetFeat(input_dim=3).to(DEVICE)

# #     # Infer feature dims dynamically
# #     cnn_extractor.eval()
# #     sensor_feat.eval()
# #     with torch.no_grad():
# #         dummy_img = torch.randn(1, 3, 300, 300).to(DEVICE)
# #         img_dim = cnn_extractor(dummy_img).shape[1]
# #         dummy_sensor = torch.randn(1, 3).to(DEVICE)
# #         sensor_dim = sensor_feat(dummy_sensor).shape[1]

# #     # Build fusion model with inferred dims
# #     fusion_model = EarlyFusionModel(cnn_extractor, sensor_feat, img_dim, sensor_dim, num_classes=NUM_CLASSES).to(DEVICE)

# #     # Load the fusion model weights from your single checkpoint
# #     fusion_model.load_state_dict(torch.load("banana_early_fusion_model_V2.pth", map_location=DEVICE))
# #     # model_path = os.path.join(current_dir, '..', 'models', 'banana_early_fusion_model.pth')

# #     fusion_model.eval()
# #     return fusion_model

# # model = load_model()

# # def preprocess_image(image):
# #     preprocess = transforms.Compose([
# #         transforms.Resize((300, 300)),
# #         transforms.ToTensor(),
# #         transforms.Normalize(
# #             mean=[0.485, 0.456, 0.406],
# #             std=[0.229, 0.224, 0.225]
# #         ),
# #     ])
# #     return preprocess(image).unsqueeze(0).to(DEVICE)

# # def preprocess_sensor(sensor_vals):
# #     sensor_tensor = torch.tensor(sensor_vals, dtype=torch.float32).unsqueeze(0).to(DEVICE)
# #     return sensor_tensor

# # st.title("Fusion Model Prediction")

# # uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# # sensor1 = st.number_input("MQ4 sensor value", format="%.3f")
# # sensor2 = st.number_input("MQ135 sensor value", format="%.3f")
# # sensor3 = st.number_input("TGS2602 sensor value", format="%.3f")

# # if uploaded_file:
# #     image = Image.open(uploaded_file).convert("RGB")
# #     st.image(image, caption="Uploaded Image", use_column_width=True)

# # if st.button("Predict"):
# #     if not uploaded_file:
# #         st.error("Please upload an image.")
# #     else:
# #         img_tensor = preprocess_image(image)
# #         sensor_tensor = preprocess_sensor([sensor1, sensor2, sensor3])

# #         with torch.no_grad():
# #             outputs = model(img_tensor, sensor_tensor)
# #             _, pred = torch.max(outputs, 1)
# #             pred_class = pred.item()

# #         class_names = {0: "Unripe", 1: "Ripe", 2: "Rotten"}
# #         st.success(f"Prediction: {class_names.get(pred_class, 'Unknown')}")

# # import numpy as np  # make sure to import numpy for sensor preprocessing
# # import streamlit as st
# # import torch
# # import torch.nn as nn
# # from PIL import Image
# # import torchvision.transforms as transforms
# # from efficientnet_pytorch import EfficientNet
# # import joblib  # to load scaler
# # # import joblib
# # # import joblib
# # scaler = joblib.load('sensor_scaler.save')

# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # NUM_CLASSES = 3

# # # Define CNN extractor exactly as before
# # class EfficientFeatureExtractor(nn.Module):
# #     def __init__(self, pretrained=False):
# #         super().__init__()
# #         if pretrained:
# #             self.base = EfficientNet.from_pretrained('efficientnet-b3')
# #         else:
# #             self.base = EfficientNet.from_name('efficientnet-b3')
# #     def forward(self, x):
# #         features = self.base.extract_features(x)
# #         out = nn.functional.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1)
# #         return out

# # # Define Sensor extractor exactly as before
# # class SensorNetFeat(nn.Module):
# #     def __init__(self, input_dim=3, hidden_dim=64, output_dim=128):
# #         super().__init__()
# #         self.model = nn.Sequential(
# #             nn.Linear(input_dim, hidden_dim),
# #             nn.ReLU(),
# #             nn.BatchNorm1d(hidden_dim),
# #             nn.Dropout(0.2),
# #             nn.Linear(hidden_dim, output_dim),
# #             nn.ReLU()
# #         )
# #     def forward(self, x):
# #         return self.model(x)

# # # Define Fusion model as before
# # class EarlyFusionModel(nn.Module):
# #     def __init__(self, cnn_feat_extractor, sensor_feat_extractor, img_dim, sensor_dim, num_classes=3):
# #         super().__init__()
# #         self.cnn = cnn_feat_extractor
# #         self.sensor = sensor_feat_extractor
# #         hidden = 256
# #         self.fusion = nn.Sequential(
# #             nn.Linear(img_dim + sensor_dim, hidden),
# #             nn.BatchNorm1d(hidden),
# #             nn.ReLU(),
# #             nn.Dropout(0.4),
# #             nn.Linear(hidden, 128),
# #             nn.ReLU(),
# #             nn.Dropout(0.2),
# #             nn.Linear(128, num_classes)
# #         )
# #     def forward(self, img, sensor):
# #         img_feat = self.cnn(img)
# #         sensor_feat = self.sensor(sensor)
# #         x = torch.cat([img_feat, sensor_feat], dim=1)
# #         out = self.fusion(x)
# #         return out

# # @st.cache_resource
# # def load_model_and_scaler():
# #     # Load scaler (make sure the file 'sensor_scaler.save' is in your app folder)
# #     scaler = joblib.load('sensor_scaler.save')

# #     # Instantiate CNN and sensor extractors
# #     cnn_extractor = EfficientFeatureExtractor(pretrained=False).to(DEVICE)
# #     sensor_feat = SensorNetFeat(input_dim=3).to(DEVICE)

# #     # Infer feature dims dynamically
# #     cnn_extractor.eval()
# #     sensor_feat.eval()
# #     with torch.no_grad():
# #         dummy_img = torch.randn(1, 3, 224, 224).to(DEVICE)  # use 224,224 as in training
# #         img_dim = cnn_extractor(dummy_img).shape[1]
# #         dummy_sensor = torch.randn(1, 3).to(DEVICE)
# #         sensor_dim = sensor_feat(dummy_sensor).shape[1]

# #     # Build fusion model with inferred dims
# #     fusion_model = EarlyFusionModel(cnn_extractor, sensor_feat, img_dim, sensor_dim, num_classes=NUM_CLASSES).to(DEVICE)

# #     # Load the fusion model weights
# #     fusion_model.load_state_dict(torch.load("banana_early_fusion_model_V2.pth", map_location=DEVICE))

# #     fusion_model.eval()
# #     return fusion_model, scaler

# # model, scaler = load_model_and_scaler()

# # def preprocess_image(image):
# #     preprocess = transforms.Compose([
# #         transforms.Resize((224, 224)),  # must match training size!
# #         transforms.ToTensor(),
# #         transforms.Normalize(
# #             mean=[0.485, 0.456, 0.406],
# #             std=[0.229, 0.224, 0.225]
# #         ),
# #     ])
# #     return preprocess(image).unsqueeze(0).to(DEVICE)

# # def preprocess_sensor(sensor_vals, scaler):
# #     # sensor_vals: list or array of 3 raw sensor values
# #     sensor_np = np.array(sensor_vals).reshape(1, -1).astype(np.float32)
# #     sensor_scaled = scaler.transform(sensor_np)  # use scaler to normalize
# #     sensor_tensor = torch.tensor(sensor_scaled, dtype=torch.float32).to(DEVICE)
# #     return sensor_tensor

# # st.title("Fusion Model Prediction")

# # uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# # sensor1 = st.number_input("MQ4 sensor value", format="%.3f")
# # sensor2 = st.number_input("MQ135 sensor value", format="%.3f")
# # sensor3 = st.number_input("TGS2602 sensor value", format="%.3f")

# # if uploaded_file:
# #     image = Image.open(uploaded_file).convert("RGB")
# #     st.image(image, caption="Uploaded Image", use_column_width=True)

# # if st.button("Predict"):
# #     if not uploaded_file:
# #         st.error("Please upload an image.")
# #     else:
# #         img_tensor = preprocess_image(image)
# #         sensor_tensor = preprocess_sensor([sensor1, sensor2, sensor3], scaler)

# #         with torch.no_grad():
# #             outputs = model(img_tensor, sensor_tensor)
# #             _, pred = torch.max(outputs, 1)
# #             pred_class = pred.item()

# #         class_names = {0: "Unripe", 1: "Ripe", 2: "Rotten"}
# #         st.success(f"Prediction: {class_names.get(pred_class, 'Unknown')}")

# # import streamlit as st
# # import torch
# # import torch.nn as nn
# # from PIL import Image
# # import torchvision.transforms as transforms
# # from efficientnet_pytorch import EfficientNet
# # import joblib
# # import numpy as np

# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # NUM_CLASSES = 3

# # # Define CNN extractor exactly as before
# # class EfficientFeatureExtractor(nn.Module):
# #     def __init__(self, pretrained=False):
# #         super().__init__()
# #         if pretrained:
# #             self.base = EfficientNet.from_pretrained('efficientnet-b3')
# #         else:
# #             self.base = EfficientNet.from_name('efficientnet-b3')
# #     def forward(self, x):
# #         features = self.base.extract_features(x)
# #         out = nn.functional.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1)
# #         return out

# # # Define Sensor extractor exactly as before
# # class SensorNetFeat(nn.Module):
# #     def __init__(self, input_dim=3, hidden_dim=64, output_dim=128):
# #         super().__init__()
# #         self.model = nn.Sequential(
# #             nn.Linear(input_dim, hidden_dim),
# #             nn.ReLU(),
# #             nn.BatchNorm1d(hidden_dim),
# #             nn.Dropout(0.2),
# #             nn.Linear(hidden_dim, output_dim),
# #             nn.ReLU()
# #         )
# #     def forward(self, x):
# #         return self.model(x)

# # # Define Fusion model as before
# # class EarlyFusionModel(nn.Module):
# #     def __init__(self, cnn_feat_extractor, sensor_feat_extractor, img_dim, sensor_dim, num_classes=3):
# #         super().__init__()
# #         self.cnn = cnn_feat_extractor
# #         self.sensor = sensor_feat_extractor
# #         hidden = 256
# #         self.fusion = nn.Sequential(
# #             nn.Linear(img_dim + sensor_dim, hidden),
# #             nn.BatchNorm1d(hidden),
# #             nn.ReLU(),
# #             nn.Dropout(0.4),
# #             nn.Linear(hidden, 128),
# #             nn.ReLU(),
# #             nn.Dropout(0.2),
# #             nn.Linear(128, num_classes)
# #         )
# #     def forward(self, img, sensor):
# #         img_feat = self.cnn(img)
# #         sensor_feat = self.sensor(sensor)
# #         x = torch.cat([img_feat, sensor_feat], dim=1)
# #         out = self.fusion(x)
# #         return out

# # @st.cache_resource
# # def load_model_and_scaler():
# #     # Load scaler
# #     scaler = joblib.load("sensor_scaler.save")

# #     # Instantiate CNN and sensor extractors
# #     cnn_extractor = EfficientFeatureExtractor(pretrained=False).to(DEVICE)
# #     sensor_feat = SensorNetFeat(input_dim=3).to(DEVICE)

# #     # Infer feature dims dynamically
# #     cnn_extractor.eval()
# #     sensor_feat.eval()
# #     with torch.no_grad():
# #         dummy_img = torch.randn(1, 3, 300, 300).to(DEVICE)
# #         img_dim = cnn_extractor(dummy_img).shape[1]
# #         dummy_sensor = torch.randn(1, 3).to(DEVICE)
# #         sensor_dim = sensor_feat(dummy_sensor).shape[1]

# #     # Build fusion model with inferred dims
# #     fusion_model = EarlyFusionModel(cnn_extractor, sensor_feat, img_dim, sensor_dim, num_classes=NUM_CLASSES).to(DEVICE)

# #     # Load the fusion model weights
# #     fusion_model.load_state_dict(torch.load("banana_early_fusion_model_V2.pth", map_location=DEVICE))
# #     fusion_model.eval()

# #     return fusion_model, scaler

# # model, scaler = load_model_and_scaler()

# # def preprocess_image(image):
# #     preprocess = transforms.Compose([
# #         transforms.Resize((300, 300)),
# #         transforms.ToTensor(),
# #         transforms.Normalize(
# #             mean=[0.485, 0.456, 0.406],
# #             std=[0.229, 0.224, 0.225]
# #         ),
# #     ])
# #     return preprocess(image).unsqueeze(0).to(DEVICE)

# # def preprocess_sensor(sensor_vals):
# #     # sensor_vals is a list or array like [sensor1, sensor2, sensor3]
# #     scaled_vals = scaler.transform(np.array([sensor_vals]))  # 2D array for scaler
# #     sensor_tensor = torch.tensor(scaled_vals, dtype=torch.float32).to(DEVICE)
# #     return sensor_tensor

# # st.title("Fusion Model Prediction")

# # uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# # sensor1 = st.number_input("MQ4 sensor value", format="%.3f")
# # sensor2 = st.number_input("MQ135 sensor value", format="%.3f")
# # sensor3 = st.number_input("TGS2602 sensor value", format="%.3f")

# # if uploaded_file:
# #     image = Image.open(uploaded_file).convert("RGB")
# #     st.image(image, caption="Uploaded Image", use_column_width=True)

# # if st.button("Predict"):
# #     if not uploaded_file:
# #         st.error("Please upload an image.")
# #     else:
# #         img_tensor = preprocess_image(image)
# #         sensor_tensor = preprocess_sensor([sensor1, sensor2, sensor3])

# #         with torch.no_grad():
# #             outputs = model(img_tensor, sensor_tensor)
# #             _, pred = torch.max(outputs, 1)
# #             pred_class = pred.item()

# #         class_names = {0: "Unripe", 1: "Ripe", 2: "Rotten"}
# #         st.success(f"Prediction: {class_names.get(pred_class, 'Unknown')}")


# import streamlit as st
# import torch
# import torch.nn as nn
# from PIL import Image
# import torchvision.transforms as transforms
# from efficientnet_pytorch import EfficientNet
# import joblib
# import numpy as np

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NUM_CLASSES = 3

# # Define CNN extractor exactly as before
# class EfficientFeatureExtractor(nn.Module):
#     def __init__(self, pretrained=False):
#         super().__init__()
#         if pretrained:
#             self.base = EfficientNet.from_pretrained('efficientnet-b3')
#         else:
#             self.base = EfficientNet.from_name('efficientnet-b3')
#     def forward(self, x):
#         features = self.base.extract_features(x)
#         out = nn.functional.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1)
#         return out

# # Define Sensor extractor exactly as before
# class SensorNetFeat(nn.Module):
#     def __init__(self, input_dim=3, hidden_dim=64, output_dim=128):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim, output_dim),
#             nn.ReLU()
#         )
#     def forward(self, x):
#         return self.model(x)

# # Define Fusion model as before
# class EarlyFusionModel(nn.Module):
#     def __init__(self, cnn_feat_extractor, sensor_feat_extractor, img_dim, sensor_dim, num_classes=3):
#         super().__init__()
#         self.cnn = cnn_feat_extractor
#         self.sensor = sensor_feat_extractor
#         hidden = 256
#         self.fusion = nn.Sequential(
#             nn.Linear(img_dim + sensor_dim, hidden),
#             nn.BatchNorm1d(hidden),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(hidden, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, num_classes)
#         )
#     def forward(self, img, sensor):
#         img_feat = self.cnn(img)
#         sensor_feat = self.sensor(sensor)
#         x = torch.cat([img_feat, sensor_feat], dim=1)
#         out = self.fusion(x)
#         return out

# @st.cache_resource
# def load_model_and_scaler():
#     # Load scaler for sensor data
#     scaler = joblib.load('sensor_scaler_v2.save')


#     # Instantiate CNN and sensor extractors
#     cnn_extractor = EfficientFeatureExtractor(pretrained=False).to(DEVICE)
#     sensor_feat = SensorNetFeat(input_dim=3).to(DEVICE)

#     # Infer feature dims dynamically
#     cnn_extractor.eval()
#     sensor_feat.eval()
#     with torch.no_grad():
#         dummy_img = torch.randn(1, 3, 300, 300).to(DEVICE)
#         img_dim = cnn_extractor(dummy_img).shape[1]
#         dummy_sensor = torch.randn(1, 3).to(DEVICE)
#         sensor_dim = sensor_feat(dummy_sensor).shape[1]

#     # Build fusion model with inferred dims
#     fusion_model = EarlyFusionModel(cnn_extractor, sensor_feat, img_dim, sensor_dim, num_classes=NUM_CLASSES).to(DEVICE)

#     # Load the fusion model weights from your checkpoint
#     fusion_model.load_state_dict(torch.load("banana_early_fusion_model_V3.pth", map_location=DEVICE))
#     fusion_model.eval()

#     return fusion_model, scaler

# model, scaler = load_model_and_scaler()

# def preprocess_image(image):
#     preprocess = transforms.Compose([
#         transforms.Resize((300, 300)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         ),
#     ])
#     return preprocess(image).unsqueeze(0).to(DEVICE)

# def preprocess_sensor(sensor_vals, scaler):
#     # sensor_vals: list or array of 3 floats
#     sensor_array = np.array(sensor_vals).reshape(1, -1)
#     sensor_scaled = scaler.transform(sensor_array)
#     sensor_tensor = torch.tensor(sensor_scaled, dtype=torch.float32).to(DEVICE)
#     return sensor_tensor

# st.title("Fusion Model Prediction")

# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# sensor1 = st.number_input("MQ4 sensor value", format="%.3f")
# sensor2 = st.number_input("MQ135 sensor value", format="%.3f")
# sensor3 = st.number_input("TGS2602 sensor value", format="%.3f")

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

# if st.button("Predict"):
#     if not uploaded_file:
#         st.error("Please upload an image.")
#     else:
#         img_tensor = preprocess_image(image)
#         sensor_tensor = preprocess_sensor([sensor1, sensor2, sensor3], scaler)

#         with torch.no_grad():
#             outputs = model(img_tensor, sensor_tensor)
#             _, pred = torch.max(outputs, 1)
#             pred_class = pred.item()

#         class_names = {0: "Unripe", 1: "Ripe", 2: "Rotten"}
#         st.success(f"Prediction: {class_names.get(pred_class, 'Unknown')}")

import streamlit as st
import torch
import torch.nn as nn
import joblib
from PIL import Image
import numpy as np
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

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
        dummy_sensor = torch.randn(1, 3, device=device)
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

    sensor_feat = SensorNetFeat(input_dim=3, hidden_dim=64, output_dim=128).to(DEVICE)
    sensor_checkpoint = torch.load("Banana_Sensor.pth", map_location=DEVICE)
    sensor_feat.load_state_dict(sensor_checkpoint, strict=False)

    img_dim, sensor_dim = infer_feature_dims(cnn_extractor, sensor_feat, DEVICE)

    fusion_model = EarlyFusionModel(cnn_extractor, sensor_feat, img_dim, sensor_dim, num_classes=NUM_CLASSES).to(DEVICE)
    fusion_model.load_state_dict(torch.load("banana_early_fusion_model_V3.pth", map_location=DEVICE))
    fusion_model.eval()
    return fusion_model

@st.cache_resource(show_spinner=False)
def load_scaler():
    return joblib.load('sensor_scaler_v2.save')

@st.cache_resource(show_spinner=False)
def load_label_encoder():
    return joblib.load('label_encoder.save')

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

# Streamlit UI as before...
st.title("Banana Ripeness Prediction")


uploaded_file = st.file_uploader("Upload a banana image", type=["jpg", "jpeg", "png"])
mq4 = st.number_input("MQ4 Sensor Value", min_value=0.0, step=0.01, format="%.2f")
mq135 = st.number_input("MQ135 Sensor Value", min_value=0.0, step=0.01, format="%.2f")
tgs2602 = st.number_input("TGS2602 Sensor Value", min_value=0.0, step=0.01, format="%.2f")

if st.button("Predict Ripeness"):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        sensor_values = [mq4, mq135, tgs2602]

        pred_idx = predict_ripeness(image, sensor_values)
        pred_name = label_encoder.classes_[pred_idx]

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(f"Predicted Ripeness Class: **{pred_name}**")    
    else:
        st.error("Please upload an image.")

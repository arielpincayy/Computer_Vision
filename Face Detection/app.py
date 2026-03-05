import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

st.set_page_config(page_title="¿Soy yo?", page_icon="🧬", layout="centered")
st.title("¿Soy yo? 🧬")
st.write("Sube una foto y el modelo dirá si eres tú o no.")


class FaceClassifier(nn.Module):
    def __init__(self, hidden_layers=None, dropout_rate=0.3, num_classes=2):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        for param in self.facenet.parameters():
            param.requires_grad = False
        if hidden_layers is None:
            hidden_layers = [256, 128]
        layers = []
        in_features = 512
        for h in hidden_layers:
            layers += [nn.Linear(in_features, h), nn.ReLU(), nn.Dropout(dropout_rate)]
            in_features = h
        layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        with torch.no_grad():
            emb = self.facenet(x)
        return self.classifier(emb)


@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaceClassifier()
    state_dict = torch.load('model.pth', map_location=device)
    fixed = {k.replace('backbone.', 'facenet.', 1): v for k, v in state_dict.items()}
    model.load_state_dict(fixed)
    model.to(device).eval()
    return model, device


def predict(image: Image.Image, model, device):
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(image_size=160, margin=20, device=device, post_process=False)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img_rgb = image.convert('RGB')
    face = mtcnn(img_rgb)

    if face is None:
        # Sin detección: redimensionar imagen completa a 160x160 y usar directamente
        fallback = img_rgb.resize((160, 160))
        face_tensor = transform(fallback).unsqueeze(0).to(device)
    else:
        face_pil = transforms.ToPILImage()(face.byte())
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(face_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        prob_yo = probs[0][1].item() * 100
        is_me = torch.argmax(outputs, dim=1).item() == 1

    return is_me, prob_yo


# ── UI ────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, use_container_width=True)

    with st.spinner("Analizando..."):
        try:
            model, device = load_model()
            is_me, prob_yo = predict(image, model, device)
        except FileNotFoundError:
            st.error("No se encontró `model.pth`. Pon el archivo junto a `app.py`.")
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    if is_me:
        st.success(f"✅ ¡Eres tú! — Confianza: {prob_yo:.1f}%")
    else:
        st.error(f"❌ No eres tú. — Confianza de que seas tú: {prob_yo:.1f}%")
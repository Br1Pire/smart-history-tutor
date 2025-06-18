from sentence_transformers import SentenceTransformer
import os


def download_mpnet_model(save_path):
    """
    Descarga el modelo 'all-mpnet-base-v2' desde Hugging Face y lo guarda en save_path.
    """
    print(f"🚀 Descargando modelo 'all-mpnet-base-v2'...")
    model = SentenceTransformer("all-mpnet-base-v2")

    # Guardar el modelo en el path especificado
    model.save(save_path)

    print(f"✅ Modelo guardado en: {save_path}")


if __name__ == "__main__":
    # Ruta donde guardar el modelo
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "src", "models", "all-mpnet-base-v2")

    os.makedirs(model_dir, exist_ok=True)

    download_mpnet_model(model_dir)

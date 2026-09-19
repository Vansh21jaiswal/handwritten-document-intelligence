import sys
import io
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.pipeline import HTRPipeline


@st.cache_resource(show_spinner="Loading model checkpoint...")
def load_model_pipeline() -> HTRPipeline:
    """Load the trained model pipeline, cached by Streamlit."""
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "best.pt"
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    
    if not checkpoint_path.exists():
        st.error(
            f"Model checkpoint not found. Please ensure the baseline model is trained "
            f"and saved at 'checkpoints/best.pt'."
        )
        st.stop()
        
    try:
        pipeline = HTRPipeline(
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path),
            device="cpu"  # Keep CPU for web deployment compatibility
        )
        return pipeline
    except Exception as e:
        st.error(f"Failed to load the model pipeline: {str(e)}")
        st.stop()


def main():
    st.set_page_config(
        page_title="Handwritten Document Intelligence",
        page_icon="📝",
        layout="centered"
    )

    # 1. Main UI Headers
    st.title("Handwritten Document Intelligence")
    st.markdown("### Handwritten text recognition powered by a CNN-BiLSTM-CTC model.")
    
    st.write("Upload a handwritten line image to instantly extract the transcribed text.")
    
    # 2. File Uploader
    uploaded_file = st.file_uploader(
        "Choose a handwritten line image...",
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None:
        try:
            # Process uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Handwritten Line", use_column_width=True)
            
            # Predict Button
            if st.button("Recognize Handwriting", type="primary"):
                pipeline = load_model_pipeline()
                
                with st.spinner("Analyzing handwriting..."):
                    try:
                        transcription = pipeline.predict(image)
                        
                        st.success("Recognition Complete!")
                        st.markdown("#### Recognized Text:")
                        
                        # st.code provides a native copy-to-clipboard button
                        st.code(transcription, language="text")
                        
                        # Download button
                        st.download_button(
                            label="Download Transcription",
                            data=transcription,
                            file_name="transcription.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"An error occurred during inference: {str(e)}")
                        
        except UnidentifiedImageError:
            st.error("The uploaded file could not be identified as an image. Please upload a valid PNG or JPEG.")
        except Exception as e:
            st.error(f"An unexpected error occurred while processing the image: {str(e)}")

    # 3. About the Model (Sidebar)
    st.sidebar.title("About the Model")
    st.sidebar.markdown(
        """
        **Architecture:**
        `CNN → BiLSTM → Linear → CTC`
        
        **Dataset:**
        IAM Handwriting Database, line-level
        """
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Evaluation Metrics")
    st.sidebar.info(
        """
        **Baseline Test CER:** 12.12%
        
        **Baseline Test WER:** 42.06%
        """
    )
    st.sidebar.caption(
        "These metrics represent the baseline evaluation on the held-out **IAM Test Set**. "
        "Lower Character Error Rate (CER) and Word Error Rate (WER) indicate fewer recognition errors. "
        "Performance on arbitrary user-uploaded handwriting may vary."
    )


if __name__ == "__main__":
    main()

import streamlit as st
from src import img_embeddings, classify

# Define your class labels here
classes = ["Negative Intra-Epithelial Lesion", "Low Intra-Epithelial Lesion", "High Intra-Epithelial Lesion"]  # Replace with actual class names

def main():
    st.title('ACVSS Group 5')
    st.write('A simple prototype for Pathology Image Classification')

    # File uploader allows only image files
    uploaded_file = st.file_uploader("Choose a Pathology Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:        
        st.spinner("Classifying Pathology Scan.......")
        
        # Process and classify the image
        embeddings = img_embeddings(uploaded_file)
        output = classify(embeddings)

        # Get the class label from the output
        final_class = classes[output]

        st.write(f"Output Prediction: {final_class}")

if __name__ == "__main__":
    main()

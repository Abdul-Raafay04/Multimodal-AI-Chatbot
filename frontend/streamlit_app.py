# frontend/streamlit_app.py
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Multimodal Chatbot", page_icon="🤖")

st.title("🤖 Multimodal AI Chatbot")
# st.write(
#     "Ask questions with **text** or **images**. "
#     "Backend: Flask + LangChain + LlamaIndex + Hugging Face."
# )

tab_text, tab_image = st.tabs(["Text Query", "Image Query"])

with tab_text:
    st.subheader("Text Chat")
    question = st.text_area("Enter your question:")

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Querying backend..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/query/text",
                        json={"question": question},
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        st.success("Response:")
                        st.write(resp.json().get("answer", "No answer field."))
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

with tab_image:
    st.subheader("Image + Question")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    img_question = st.text_input(
        "Optional question about the image (or leave blank to just describe it):"
    )

    if st.button("Send Image", type="primary"):
        if uploaded_file is None:
            st.warning("Please upload an image.")
        else:
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            data = {"question": img_question}
            with st.spinner("Querying backend..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/query/image",
                        files=files,
                        data=data,
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        st.image(uploaded_file, caption="Uploaded image", use_column_width=True)
                        st.success("Response:")
                        st.write(resp.json().get("answer", "No answer field."))
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

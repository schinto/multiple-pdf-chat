from typing import List, Optional, Dict, Any
import pathlib
import tempfile
from dotenv import load_dotenv
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain.vectorstores.base import VectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFDirectoryLoader

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# from langchain.embeddings import HuggingFaceInstructEmbeddings
# from langchain.llms import HuggingFaceHub

from langchain.indexes import VectorstoreIndexCreator
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import messages_to_dict
from langchain.prompts import SystemMessagePromptTemplate

from htmlTemplates import bot_template, css, user_template

# Define parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# TODO:
# - Select LLM model and model parameters
# - Select embedding model
# - Save history to text file
# - Load old history file
# - Save session
# - How to add the reference to the page of the PDF document, which the LLM used?
# - Select different memory types and corresponding parameters
# - Add persistent vectorstore like Chroma, Pinecone, etc.
# - Use PDFs stored in Google Drive


def load_pdfs_to_vectorstore(pdf_docs: List[UploadedFile]) -> Optional[VectorStore]:
    """
    This function reads the PDF files, creates a vectorstore, and returns it.

    Parameters:
    pdf_docs (List[UploadedFile]): A list of uploaded PDF files

    Returns:
    vectorstore (Optional[VectorStore]): A VectorStore object containing vector representations of the documents
    """
    with tempfile.TemporaryDirectory(prefix="chatPDFs_") as tmpdirname:
        for pdf in pdf_docs:
            fullname = str(pathlib.Path(tmpdirname).joinpath(pdf.name))
            contents = pdf.read()
            with open(fullname, "wb") as f:
                f.write(contents)
        loader = PyPDFDirectoryLoader(tmpdirname)
        vectorstore = (
            VectorstoreIndexCreator(
                vectorstore_cls=FAISS,
                embedding=OpenAIEmbeddings(),
                # embedding=HuggingFaceInstructEmbeddings(
                #     model_name="hkunlp/instructor-xl"
                # ),
                text_splitter=CharacterTextSplitter(
                    separator="\n",
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                ),
            )
            .from_loaders([loader])
            .vectorstore
        )
        return vectorstore


class AnswerConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        return super(AnswerConversationBufferMemory, self).save_context(
            inputs, {"response": outputs["answer"]}
        )


def get_conversation_chain(vectorestore: VectorStore) -> ConversationalRetrievalChain:
    """
    This function creates a conversation chain and returns it.

    Parameters:
    vectorestore (VectorStore): A VectorStore object containing vector representations of the documents

    Returns:
    conversation_chain (ConversationalRetrievalChain): A ConversationalRetrievalChain object for handling conversations
    """
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(
    #     repo_id="google/flan-t5-xxl",
    #     model_kwargs={"temperature": 0.7, "max_length": 512},
    # )
    # llm = HuggingFaceHub(
    #     repo_id="tiiuae/falcon-7b-instruct",
    #     model_kwargs={"temperature": 0.7, "max_new_tokens": 500},
    # )
    memory = AnswerConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorestore.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )


def handle_userinput(user_question: str) -> None:
    """
    This function handles user input, fetches a response from the conversation, and updates the UI.

    Parameters:
    user_question (str): The question asked by the user

    Returns:
    None
    """
    response = st.session_state.conversation({"question": user_question})
    # st.write(response)
    st.session_state.chat_history = response["chat_history"]


def main():
    """
    Main function to initiate the application
    """
    load_dotenv()

    st.set_page_config(
        page_title="Chat with multiple PDFs", page_icon=":books:", layout="wide"
    )
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    st.session_state.conversation = st.session_state.get("conversation", None)
    st.session_state.chat_history = st.session_state.get("chat_history", None)

    st.header("Chat with multiple PDFs :books:")

    system_container = st.container()
    # Show response container first and then the input container
    # container for chat history
    response_container = st.container()
    # container for user input
    input_container = st.container()

    with input_container:
        with st.form(key="input_form", clear_on_submit=True):
            user_question = st.text_area(
                "Ask a question about your documents:", key="input", height=20
            )
            submit_button = st.form_submit_button(label="Send")

        if user_question and submit_button:
            if st.session_state.conversation is not None:
                handle_userinput(user_question)
            else:
                st.error("Upload and process PDFs first!", icon="⚠️")

    with response_container:
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(
                        user_template.replace("{{MSG}}", message.content),
                        unsafe_allow_html=True,
                    )
                else:
                    st.write(
                        bot_template.replace("{{MSG}}", message.content),
                        unsafe_allow_html=True,
                    )
        if st.session_state.conversation is not None:
            # st.markdown(st.session_state.conversation)
            dicts = messages_to_dict(
                st.session_state.conversation.memory.chat_memory.messages
            )
            with st.expander("Conversation", expanded=False):
                st.write(dicts)

    with st.sidebar:
        # st.subheader("Parameters")
        # with st.expander("Models", expanded=False):
        #     model_name = st.selectbox(
        #         "Choose a model:",
        #         ("OpenAI", "google/flan-t5-xxl", "tiiuae/falcon-7b-instruct"),
        #     )

        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload the PDFs here and click on 'Process'", accept_multiple_files=True
        )

        if st.button("Process"):
            with st.spinner("Processing..."):
                try:
                    # Load all pages from PDF files
                    vectorstore = load_pdfs_to_vectorstore(pdf_docs)
                except Exception as e:
                    st.error(f"Error while loading pdfs: {str(e)}")
                else:
                    st.success("Successfully created vectorstore!", icon="✅")
                    # Create conversion chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.chat_history = None
                    st.experimental_rerun()

    with system_container:
        with st.expander("System", expanded=False):
            if st.session_state.conversation is not None:
                with st.form(key="system_form"):
                    system_template = st.session_state.conversation.combine_docs_chain.llm_chain.prompt.messages[
                        0
                    ].prompt.template
                    new_system_template = st.text_area(
                        label="Prompt Template",
                        value=system_template,
                        key="system",
                        height=100,
                    )
                    submit_button = st.form_submit_button(label="Save")
                if new_system_template and submit_button:
                    st.session_state.conversation.combine_docs_chain.llm_chain.prompt.messages[
                        0
                    ] = SystemMessagePromptTemplate.from_template(
                        new_system_template
                    )
                    st.experimental_rerun()


if __name__ == "__main__":
    main()

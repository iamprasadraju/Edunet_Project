import streamlit as st
import nltk
from questions import top_files,load_files,tokenize,compute_idfs,top_sentences

# ... (Paste the code for the functions here)

FILE_MATCHES = 1
SENTENCE_MATCHES = 1

# ... (Paste the rest of the provided code here)


def streamlit_app():
    st.title("QueryMaster")

    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown(
        "This app uses TF-IDF to perform information retrieval on a corpus of text documents."
    )

    st.sidebar.header("Parameters")
    FILE_MATCHES = st.sidebar.slider(
        "Number of Top Files", min_value=1, max_value=10, value=1
    )
    SENTENCE_MATCHES = st.sidebar.slider(
        "Number of Top Sentences", min_value=1, max_value=10, value=1
    )
    st.sidebar.header("Created by")
    st.sidebar.markdown(
        "G.Prasad Raju"
    )
    st.sidebar.markdown(
        "P.Ramana"
    )
    st.sidebar.markdown(
        "N.Sabareesh"
    ) 
    st.sidebar.markdown(
        "J.Ravi Surya"
    )

     

    # User input query
    query = st.text_input("Enter your query:")

    if st.button("Search"):
        # Calculate IDF values across files
        files = load_files("corpus")  # Make sure to provide the correct directory
        file_words = {
            filename: tokenize(files[filename])
            for filename in files
        }
        file_idfs = compute_idfs(file_words)

        # Determine top file matches according to TF-IDF
        query_tokens = set(tokenize(query))
        filenames = top_files(query_tokens, file_words, file_idfs, n=FILE_MATCHES)

        # Extract sentences from top files
        sentences = dict()
        for filename in filenames:
            for passage in files[filename].split("\n"):
                for sentence in nltk.sent_tokenize(passage):
                    tokens = tokenize(sentence)
                    if tokens:
                        sentences[sentence] = tokens

        # Compute IDF values across sentences
        idfs = compute_idfs(sentences)

        # Determine top sentence matches
        matches = top_sentences(query_tokens, sentences, idfs, n=SENTENCE_MATCHES)

        st.header("Top Matching Sentences:")
        for match in matches:
            st.write(match)


if __name__ == "__main__":
    streamlit_app()

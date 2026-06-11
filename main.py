import grpc
from concurrent import futures
import os
import io
import PyPDF2
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import tempfile
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai import OpenAI

import grpc_proto.brain_pb2 as brain_pb2
import grpc_proto.brain_pb2_grpc as brain_pb2_grpc

load_dotenv()
nvidia_api_key = os.getenv("NVIDIA_API_KEY")

if not nvidia_api_key:
    raise ValueError("NVIDIA_API_KEY not set in .env file!")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=nvidia_api_key
)

# storage
chroma_client = chromadb.PersistentClient(path="./chroma_data")

# text translator
multilingual_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

collection = chroma_client.get_or_create_collection(
    name="documents", 
    embedding_function=multilingual_ef
)

class BrainService(brain_pb2_grpc.BrainServiceServicer):

    # name must match the RPC definition in the .proto file
    def ProcessDocument(self, request, context):
        print(f"Document received by AI Engine.")
        print(f"   - File Name : {request.file_name}")
        print(f"   - Author    : {request.author}")
        print(f"   - Size      : {len(request.content)} bytes")

        print(f"Extracting text from PDF...")
        document_text = ""
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(request.content)
                temp_pdf_path = temp_pdf.name

            document_text = pymupdf4llm.to_markdown(temp_pdf_path)
            os.remove(temp_pdf_path)

            print(f"   - Extracted : {len(document_text)} characters (Markdown Format)")

        except Exception as e:
            print(f"Error. Failed to extract PDF text: {e}")
            return brain_pb2.DocumentResponse(
                success=False,
                message=f"System failed to read the PDF file: {str(e)}",
                document_id="ERROR-PDF-001"
            )

        print("Starting document analysis & Chunking...")

        # PROMPT TO AI
        prompt = f"""
        You are SecondBrain AI, an expert Executive Summarizer and Knowledge Extractor.
        Your task is to analyze the provided document text and generate a highly accurate, structured, and concise executive summary.

        [DOCUMENT TEXT]
        {document_text}

        [STRICT INSTRUCTIONS]
        1. GREETING: You MUST start your response exactly with: "Hello {request.author}, here is the summary of your document:"
        2. OCR NOISE REDUCTION: The text is extracted from a PDF and may contain irregular spacing, missing punctuation, or fragmented words. Mentally reconstruct the text to understand its true meaning before summarizing.
        3. NO HALLUCINATION: Base your summary EXCLUSIVELY on the provided text. Do not add outside information or personal opinions.
        4. TONE & LENGTH: Be professional, objective, and clear.

        [REQUIRED OUTPUT FORMAT]
        Strictly use the following Markdown structure for your response:

        **Document Overview:**
        (Provide 1-3 sentences explaining the main topic, purpose, or core theme of the document)

        **Key Takeaways:**
        - (Bullet point 1: The most critical fact, argument, or data point)
        - (Bullet point 2: Another crucial insight)
        - (Bullet point 3: Another crucial insight)
        - (Add 1-2 more bullet points only if absolutely necessary)

        **Conclusion / Outcome:**
        (Provide a 1-sentence wrap-up, final decision, or next steps if mentioned in the text. If none, summarize the final thought.)
        """

        try:
            # Call AI

            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
                
                chunks = text_splitter.split_text(document_text)
                ids = [f"{request.file_name}_{i}" for i in range(len(chunks))]
                metadatas = [{"source": request.file_name, "author": request.author} for _ in chunks]

                collection.add(
                    documents=chunks,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"   - Saved {len(chunks)} SMART chunks to Vector Database.")
            except Exception as e:
                print(f"   - Warning: Failed to save to ChromaDB: {e}")

            # B. AI SUMMARY (add num_predict so it fits)
            print("Requesting summary from NVIDIA NIM...")
            completion = client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
                stream=False
            )
            ai_response = completion.choices[0].message.content
            
            print("[SUCCESS] AI analysis completed.")
            success_status = True

        except Exception as e:
            # If Error
            print(f"[ERROR] AI processing failed: {e}")
            ai_response = f"AI Engine encountered an error: {str(e)}"
            success_status = False
        
        

        print("Dispatching response back to Go Gateway.")

        # returning to go
        return brain_pb2.DocumentResponse(
            success=success_status,
            message=ai_response,
            document_id="DOC-GEMINI-001"
        )
    
    def Chat(self, request, context):
        print(f"Incoming query: {request.query}")

        try:
            results = collection.query(
                query_texts=[request.query],
                n_results=15,
                include=['documents', 'metadatas']
            )

            context_text = ""
            if results['documents'] and len(results['documents'][0]) > 0:
                for i, doc_chunk in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    filename = metadata.get('source', 'Unknown Document')
                    
                    context_text += f"[Sumber: {filename}]\n{doc_chunk}\n\n"

            # DEBUG: Print teks yang ditemukan ke terminal Python
            print(f"--- Context found for query '{request.query}': ---")
            print(context_text) 
            print("--------------------------------------------------")

            if not context_text:
                return brain_pb2.ChatResponse(answer="I couldn't find any relevant information in your documents.")

            chat_history_text = ""
            if len(request.history) >0:
                for msg in request.history:
                    role_label = "User" if msg.role == "user" else "AI"
                    chat_history_text += f"{role_label}: {msg.content}\n"

            prompt = f"""You are SecondBrain, a precise Document Intelligence assistant operating as a strict RAG (Retrieval-Augmented Generation) system.
 
<context>
{context_text}
</context>
 
<chat_history>
{chat_history_text}
</chat_history>
 
<user_query>
{request.query}
</user_query>
 
---
 
## ABSOLUTE RULES
 
**1. LANGUAGE**
Detect the language of <user_query>.
Write your ENTIRE response in that exact language — every word, header, bullet point, and label.
Never default to any specific language. Mirror the user's language precisely, even if it differs from the document language.
 
**2. CONTEXT LOCK — ZERO HALLUCINATION**
Your ONLY source of truth is the content inside the current <context> block above.
- Every claim, number, name, date, and conclusion MUST be directly traceable to <context>.
- Do NOT use your training knowledge, general assumptions, or anything outside <context>.
- Do NOT blend information from <chat_history> into factual answers — use <chat_history> only to resolve ambiguous references (e.g., "it", "that document", "the previous topic").
- If a detail is not explicitly stated in <context>, do not infer or extrapolate it.
 
**3. DOCUMENT ISOLATION**
Each document in <context> is separated by a [Source: filename] tag. Treat each as a fully independent document.
- Never transfer, merge, or blend information from one source into another.
- Only reference multiple sources together when the user explicitly asks for comparison or cross-document analysis.
 
**4. GRACEFUL FALLBACK**
If the requested information is entirely absent from <context>:
- Do NOT guess or answer from outside knowledge.
- Respond briefly and politely — in the user's language — that the information is not found in the provided documents.
- Do not speculate or suggest what the answer "might" be.
 
**5. INLINE CITATION**
When stating key facts, naturally reference the source document by name.
Example patterns: "According to [filename]..." / "Berdasarkan [nama_file]..." / "D'après [nom_fichier]..."
Match the citation phrasing to whatever language the user is writing in.
 
---
 
## RESPONSE FORMAT
 
Adapt your structure based on what the user is asking — do not force a rigid template onto every query.
 
**Direct question / specific lookup**
→ Answer concisely and precisely. One to a few paragraphs. Cite source inline.
 
**Single document summary**
→ Structure your summary to match the document's nature:
   - What is this document about? (main topic / purpose)
   - What are the key points or arguments?
   - What are the important supporting details, data, or findings?
   - What conclusions or outcomes are stated (if any)?
   
   Adapt the depth and shape to the document type. A contract, a research paper, a transcript, and a financial report each warrant a different summary structure. Do not use the same rigid format for all.
 
**Multi-document comparison** (only when explicitly requested)
→ Dedicate a clearly labeled section to each document, then write a comparative analysis.
→ Translate all section headers to the user's language.
→ Highlight similarities and key differences based strictly on <context>.
 
**Follow-up / contextual question**
→ Use <chat_history> to understand the reference, then answer from <context>.
 
---
 
Response:"""

            completion = client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                top_p=0.7,
                max_tokens=1024,
                stream=False
            )

            ai_response = completion.choices[0].message.content
            return brain_pb2.ChatResponse(answer=ai_response)

        except Exception as e:
            print(f"Error in Chat: {e}")
            return brain_pb2.ChatResponse(answer=f"Error processing your request: {str(e)}")

# start server
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10)) # make a grpc server with max 10 threads.
    brain_pb2_grpc.add_BrainServiceServicer_to_server(BrainService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()

    print("AI is running!")

    server.wait_for_termination()

if __name__ == '__main__':
    serve()
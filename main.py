import grpc
from concurrent import futures
import os
import io
import PyPDF2
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# import google.generativeai as genai # uncomment if use google ai studio
import ollama # uncomment if use ollama

import grpc_proto.brain_pb2 as brain_pb2
import grpc_proto.brain_pb2_grpc as brain_pb2_grpc

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not set!")

# genai.configure(api_key=api_key) # uncomment if gemini

# model = genai.GenerativeModel('gemini-2.5-flash') # uncomment if gemini


# Buat folder penyimpanan permanen
chroma_client = chromadb.PersistentClient(path="./chroma_data")

# Siapkan "mesin" pengubah teks ke angka
default_ef = embedding_functions.DefaultEmbeddingFunction()

# Buat wadah dokumen
collection = chroma_client.get_or_create_collection(
    name="documents", 
    embedding_function=default_ef
)

def split_text(text, chunk_size=600, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


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
            # convert raw bytes into a file-like object in memory
            pdf_stream = io.BytesIO(request.content)
            pdf_reader = PyPDF2.PdfReader(pdf_stream)

            # Loop through all pages and extract text
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    document_text += extracted + "\n"

            print(f"   - Extracted : {len(document_text)} characters")

        except Exception as e:
            print(f"Error. Failed to extract PDF text: {e}")
            return brain_pb2.DocumentResponse(
                success=False,
                message=f"System failed to read the PDF file: {str(e)}",
                document_id="ERROR-PDF-001"
            )

        print("Starting document analysis...")

        # PROMPT TO AI
        prompt = f"""
        [STRICT INSTRUCTIONS]
        - You MUST answer in ENGLISH.
        - You MUST start the response by saying: "Hello {request.author}, here is your SecondBrain analysis:"
        - Provide a VERY BRIEF executive summary.
        - Focus ONLY on the provided content.

        [DOCUMENT CONTENT]
        {document_text}

        [RE-CONFIRMATION]
        Remember, {request.author}, answer in ENGLISH and stay concise.
        """

        try:
            # Call AI

            # uncomment if gemini
            # response = model.generate_content(prompt)
            # ai_response = response.text

            try:
                chunks = split_text(document_text)
                ids = [f"{request.file_name}_{i}" for i in range(len(chunks))]
                metadatas = [{"source": request.file_name, "author": request.author} for _ in chunks]

                collection.add(
                    documents=chunks,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"   - Saved {len(chunks)} chunks to Vector Database.")
            except Exception as e:
                print(f"   - Warning: Failed to save to ChromaDB: {e}")

            # B. AI SUMMARY (Tambah num_predict agar tidak terpotong)
            response = ollama.generate(
                model='qwen2.5:3b',
                prompt=prompt,
                options={
                    "num_predict": 1500,
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            )
            ai_response = response['response']
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
            message=ai_response, # Kita kirim jawaban AI ke Go!
            document_id="DOC-GEMINI-001"
        )
    
    def Chat(self, request, context):
        print(f"Incoming query: {request.query}")

        try:
            # 1. SEARCH: Cari 3 potongan teks paling relevan di ChromaDB
            # Ini akan mencari teks berdasarkan "makna", bukan cuma kata kunci
            results = collection.query(
                query_texts=[request.query],
                n_results=7
            )

            # Gabungkan potongan teks yang ditemukan menjadi satu paragraf konteks
            context_text = "\n".join(results['documents'][0]) if results['documents'] else ""

            # DEBUG: Print teks yang ditemukan ke terminal Python
            print(f"--- Context found for query '{request.query}': ---")
            print(context_text) 
            print("--------------------------------------------------")

            if not context_text:
                return brain_pb2.ChatResponse(answer="I couldn't find any relevant information in your documents.")

            # 2. GENERATE: Minta Qwen menjawab berdasarkan konteks tersebut
            # Kita beri instruksi ketat agar AI tidak ngawur (halusinasi)
            prompt = f"""
            You are SecondBrain AI, a professional assistant. 
            Answer the user's question accurately using ONLY the context provided below.
            
            [CONTEXT]
            {context_text}
            
            [QUESTION]
            {request.query}
            
            [INSTRUCTION]
            - Answer in a direct and professional manner.
            - If the information is not in the context, say: "I'm sorry, I don't have that information in my current database."
            """

            # Call Ollama (Qwen)
            response = ollama.generate(model='qwen2.5:3b', prompt=prompt)

            return brain_pb2.ChatResponse(answer=response['response'])

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
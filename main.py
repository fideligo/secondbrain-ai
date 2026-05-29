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

            # uncomment if gemini
            # response = model.generate_content(prompt)
            # ai_response = response.text

            try:
                # LangChain Recursive Splitter: Memotong di paragraf (\n\n) dulu, baru kalimat (\n), lalu spasi
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

            # 2. GENERATE: Minta Qwen menjawab berdasarkan konteks tersebut
            # Kita beri instruksi ketat agar AI tidak ngawur (halusinasi)
            prompt = f"""Anda adalah Analis Data Khusus Dokumen yang sangat teliti. Tugas Anda adalah menjawab pertanyaan pengguna SECARA EKSKLUSIF berdasarkan informasi di dalam <context> yang diberikan.

            <context>
            {context_text}
            </context>

            <user_query>
            {request.query}
            </user_query>

            <aturan_wajib>
            1. BAHASA: Anda WAJIB menjawab sepenuhnya dalam Bahasa Indonesia yang baik dan benar.
            2. ANTI-HALUSINASI: JANGAN PERNAH menambahkan informasi, pengetahuan umum, opini, atau asumsi dari luar <context>. 
            3. ISOLASI PROYEK (SANGAT PENTING): <context> berisi gabungan dari beberapa proyek yang berbeda (ditandai dengan [Sumber: nama_file]). JANGAN MENCAMPURADUKKAN FITUR ANTAR PROYEK! JantungSinyal dan Hapta adalah dua alat yang sama sekali berbeda. Pastikan fitur bayi hanya untuk JantungSinyal, dan fitur pesepeda hanya untuk Hapta.
            4. JALUR AMAN: Jika <user_query> hanya berupa sapaan, error terminal, obrolan santai, atau informasinya benar-benar tidak ada di dalam <context>, Anda WAJIB membatalkan template dan HANYA menjawab: "Maaf, saya tidak menemukan informasi tersebut di dokumen Anda."
            </aturan_wajib>

            <format_jawaban>
            JIKA pertanyaan pengguna meminta untuk membandingkan atau membahas kedua ide lomba, Anda WAJIB menjawab dengan mengisi template persis seperti di bawah ini:

            ### Ringkasan JantungSinyal
            - **Fungsi Utama:** [Isi dengan fungsi utama berdasarkan konteks JantungSinyal]
            - **Target Pengguna:** [Isi dengan audiens/target berdasarkan konteks JantungSinyal]
            - **Teknologi Utama:** [Isi dengan sensor/teknologi berdasarkan konteks JantungSinyal]

            ### Ringkasan Hapta
            - **Fungsi Utama:** [Isi dengan fungsi utama berdasarkan konteks Hapta]
            - **Target Pengguna:** [Isi dengan audiens/target berdasarkan konteks Hapta]
            - **Teknologi Utama:** [Isi dengan sensor/teknologi berdasarkan konteks Hapta]

            ### Perbedaan Utama
            - **Fokus Solusi:** [Tuliskan perbandingan tujuan utama kedua alat]
            - **Implementasi Perangkat:** [Tuliskan perbandingan bentuk fisik atau teknologi yang dipakai]
            </format_jawaban>

            Berdasarkan <aturan_wajib>, berikan jawaban Anda sekarang:"""

            response = ollama.generate(
                model='qwen2.5:3b', 
                prompt=prompt,
                options={
                    "temperature": 0.1, # Pastikan tetap rendah agar tidak halusinasi
                    "num_predict": 800
                }
            )

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
import grpc
from concurrent import futures
import os
import io
import PyPDF2
from dotenv import load_dotenv
import google.generativeai as genai

import grpc_proto.brain_pb2 as brain_pb2
import grpc_proto.brain_pb2_grpc as brain_pb2_grpc

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not set!")

genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-2.5-flash')


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
        System Role: You are the core AI assistant for SecondBrain Enterprise.
        Task: Analyze the provided document and generate a concise summary.
        
        [Document Metadata]
        - File Name: {request.file_name}
        - Author: {request.author}
        
        [Document Content]
        {document_text}
        
        [Output Requirements]
        1. Greet the author professionally by their name.
        2. Provide a 2-3 sentence summary of the document's core message.
        3. Maintain a formal, enterprise-grade tone.
        """

        try:
            # Call AI
            response = model.generate_content(prompt)
            ai_response = response.text
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

# start server
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10)) # make a grpc server with max 10 threads.
    brain_pb2_grpc.add_BrainServiceServicer_to_server(BrainService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()

    print("AI is running on port 50051...")
    print("AI Engine Powered By Gemini")

    server.wait_for_termination()

if __name__ == '__main__':
    serve()
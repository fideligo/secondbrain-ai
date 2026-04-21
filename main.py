import grpc
from concurrent import futures
import time

import proto.brain_pb2 as brain_pb2
import proto.brain_pb2_grpc as brain_pb2_grpc

class BrainService(brain_pb2_grpc.BrainServiceServicer):

    # name must be the same with the one thats in the proto file
    def ProcessDocument(self, request, context):
        print(f"AI has received the document.")
        print(f"   - File Name : {request.file_name}")
        print(f"   - Author    : {request.author}")
        print(f"   - Size      : {len(request.content)} bytes")

        # AI Logic (RAG, LLM)
        # for now, make it as if the ai is thinking
        print("Processing document...")
        time.sleep(2)

        print("Done. Sending response to the Go Gateway")

        # returning to go
        return brain_pb2.DocumentResponse(
            success = True,
            message = f"The document {request.file_name} has been successfully analyzed by the Python AI!",
            document_id = "DUMMY-6767"
        )

# start server
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10)) # make a grpc server with max 10 threads.

    brain_pb2_grpc.add_BrainServiceServicer_to_server(BrainService(), server)

    server.add_insecure_port('[::]:50051')
    server.start()

    print("Python AI is running on port 50051...")

    server.wait_for_termination()

if __name__ == '__main__':
    serve()
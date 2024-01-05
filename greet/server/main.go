package main

import (
    "log"
    "net"
    pb "com.greet"
)

var addr string = "0.0.0.0:5051"

type Server struct {
    pb.GreetServiceServer
}

func main() {
    lis, err := net.Listen("tcp", addr)
    if err != nil {
        log.Fatalf("failed to listen: %v", err)
    }

    log.Printf("server listening on %s", addr)

    s := grpc.NewServer()
    pb.RegisterGreetServiceServer(s, &Server{})

    if err = s.Serve(lis); err != nil {
        log.Fatalf("failed to serve: %v", err)
    }
}
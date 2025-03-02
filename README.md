# ONNX Model Web Interface in Rust

A web interface for running ONNX models built with [Axum](https://github.com/tokio-rs/axum) (Rust), [MiniJinja](https://docs.rs/minijinja/latest/minijinja), [HTMX](https://htmx.org/),[ort](https://ort.pyke.io/) and [BeerCSS](https://www.beercss.com/). 
<br />Currently supports YOLOv8 for object detection in images.

![demo](https://github.com/user-attachments/assets/44bdc391-1b05-450b-8a0b-70a61c9f110b)

### Installation
```bash
git clone https://github.com/LuMarans30/onnx-playground-axum.git
cd onnx-playground-axum 

# Install dependencies and build
cargo build --release

# Start server
./target/release/onnx-playground-axum
```

Server starts at `0.0.0.0:8080`

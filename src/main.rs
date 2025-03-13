use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Request, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
};
use axum_typed_multipart::{FieldData, TryFromMultipart, TypedMultipart};
use minijinja::{Environment, context, path_loader};
use serde::Serialize;
use std::{ffi::OsStr, path::Path, sync::Arc};
use tempfile::NamedTempFile;
use tower_http::services::ServeDir;

mod yolov8m;

use yolov8m::process_image;

#[derive(Clone)]
struct AppState {
    env: Environment<'static>,
}

#[derive(TryFromMultipart)]
struct UploadAssetRequest {
    #[form_data(limit = "unlimited")]
    image: FieldData<NamedTempFile>,
}

#[derive(Serialize)]
struct UploadResponse {
    image_id: String,
    message: String,
}

pub const UPLOAD_DIR: &str = "/tmp/uploaded/";
pub const PROCESS_DIR: &str = "static/";

#[tokio::main]
async fn main() {
    let mut env = Environment::new();
    env.set_loader(path_loader("templates"));

    // Create directories for storing files
    //std::fs::create_dir_all("/tmp/processed").unwrap_or_default();
    std::fs::create_dir_all(UPLOAD_DIR).unwrap_or_default();

    let app_state = Arc::new(AppState { env });

    let app = Router::new()
        .route("/", get(root))
        .route("/pages/{page}", get(handle_page))
        .route("/yolo/upload", post(upload_image))
        .route("/yolo/process", post(process_image))
        .nest_service("/static", ServeDir::new("static"))
        .layer(DefaultBodyLimit::max(1024 * 1024 * 10)) // 10MB is probably sufficient
        .fallback(fallback)
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

async fn root(State(state): State<Arc<AppState>>) -> Result<Html<String>, StatusCode> {
    let template = state.env.get_template("base.jinja").unwrap();
    let rendered = template.render(context!()).unwrap();

    Ok(Html(rendered))
}

async fn handle_page(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(page): axum::extract::Path<String>,
    _request: Request,
) -> Result<Html<String>, StatusCode> {
    let target = format!(
        "pages/{}",
        &page.to_ascii_lowercase().replace("html", "jinja")
    );

    let template = match state.env.get_template(&target) {
        Ok(template) => template,
        Err(_) => return Err(StatusCode::NOT_FOUND),
    };

    let rendered = match template.render(context!()) {
        Ok(rendered) => rendered,
        Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
    };

    Ok(Html(rendered))
}

async fn fallback() -> (StatusCode, &'static str) {
    (StatusCode::NOT_FOUND, "Not Found")
}

async fn upload_image(
    TypedMultipart(UploadAssetRequest { image }): TypedMultipart<UploadAssetRequest>,
) -> impl IntoResponse {
    let id = uuid::Uuid::new_v4().to_string();
    let file_name = image.metadata.file_name.unwrap();
    let image_extension = get_extension_from_filename(file_name.as_str()).unwrap();
    let file_path = format!("{}{}.{}", UPLOAD_DIR, id, image_extension);

    // Ensure the file is an image
    match image.metadata.content_type {
        Some(content_type) if content_type.contains("image") => {
            // Persist the file
            match image.contents.persist(&file_path) {
                Ok(_) => (
                    StatusCode::OK,
                    Json(UploadResponse {
                        image_id: id,
                        message: "Image uploaded successfully".to_string(),
                    }),
                ),
                Err(e) => {
                    eprintln!("Error saving file: {:?}", e);
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(UploadResponse {
                            image_id: String::new(),
                            message: "Failed to save image".to_string(),
                        }),
                    )
                }
            }
        }
        _ => (
            StatusCode::IM_A_TEAPOT,
            Json(UploadResponse {
                image_id: String::new(),
                message: "The file must be an image".to_string(),
            }),
        ),
    }
}

pub fn get_extension_from_filename(filename: &str) -> Option<&str> {
    Path::new(filename).extension().and_then(OsStr::to_str)
}

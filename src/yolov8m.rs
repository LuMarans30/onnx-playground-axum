use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    sync::Arc,
};

use ab_glyph::{FontRef, PxScale};
use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use image::{GenericImageView, Rgba, imageops::FilterType};
use imageproc::{
    drawing::{draw_hollow_rect_mut, draw_text_mut},
    rect::Rect,
};
use ndarray::{Array, Axis, s};
use ort::{
    execution_providers::CUDAExecutionProvider,
    inputs,
    session::{Session, SessionOutputs},
    value::Tensor,
};
use serde::{Deserialize, Serialize};

use crate::{AppState, PROCESS_DIR, UPLOAD_DIR, get_extension_from_filename};

#[derive(Deserialize)]
pub struct ProcessImageRequest {
    image_id: String,
}

#[derive(Serialize)]
pub struct ProcessImageResponse {
    status: String,
    image_path: Option<String>,
}

pub async fn process_image(
    State(_state): State<Arc<AppState>>,
    Json(payload): Json<ProcessImageRequest>,
) -> impl IntoResponse {
    let image_id = payload.image_id;

    let mut input_path = String::new();
    let mut file_ext = "";
    let mut file_name = OsString::new();
    let mut file_path = PathBuf::new();

    for path in std::fs::read_dir(UPLOAD_DIR).unwrap() {
        let dir_entry = path.unwrap();
        file_name = dir_entry.file_name();
        let file_name_str = file_name.to_str().unwrap();
        file_path = dir_entry.path();
        let path_str = file_path.to_str().unwrap();
        println!("File name: {}", file_name_str);
        println!("Image ID: {}", image_id.as_str());
        println!("File path: {}", path_str);
        file_ext = match get_extension_from_filename(path_str) {
            Some(ext) => ext,
            None => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ProcessImageResponse {
                        status: "Failed to get file extension".to_string(),
                        image_path: None,
                    }),
                );
            }
        };
        if file_name_str == format!("{}.{}", image_id, file_ext).as_str() {
            input_path = dir_entry.path().to_str().unwrap().to_string();
        }
    }

    if input_path.is_empty() {
        return (
            StatusCode::NOT_FOUND,
            Json(ProcessImageResponse {
                status: "Image not found".to_string(),
                image_path: None,
            }),
        );
    }

    let output_path = format!("{}{}.{}", PROCESS_DIR, image_id, file_ext);

    println!(
        "Uploaded image: {}, output path: {}",
        input_path, output_path
    );

    match identify_objects(&input_path, &output_path).await {
        Ok(_) => (
            StatusCode::OK,
            Json(ProcessImageResponse {
                status: "Image processed successfully".to_string(),
                image_path: Some(output_path),
            }),
        ),
        Err(err) => {
            eprintln!("Error processing image: {:?}", err);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ProcessImageResponse {
                    status: format!("Error processing image: {}", err),
                    image_path: None,
                }),
            )
        }
    }
}

async fn identify_objects(input_path: &str, output_dir: &str) -> Result<(), String> {
    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()
        .unwrap();

    let original_img = image::open(Path::new(input_path)).unwrap();
    let (img_width, img_height) = (original_img.width(), original_img.height());
    let img = original_img.resize_exact(640, 640, FilterType::CatmullRom);
    let mut input = Array::zeros((1, 3, 640, 640));
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }

    let model = Session::builder()
        .unwrap()
        .commit_from_file(YOLOV8M_PATH)
        .unwrap();

    // Run YOLOv8 inference
    let outputs: SessionOutputs = model
        .run(inputs!["images" => Tensor::from_array(input).unwrap()].unwrap())
        .unwrap();
    let output = outputs["output0"]
        .try_extract_tensor::<f32>()
        .unwrap()
        .t()
        .into_owned();

    let mut boxes = Vec::new();
    let output = output.slice(s![.., .., 0]);
    for row in output.axis_iter(Axis(0)) {
        let row: Vec<_> = row.iter().copied().collect();
        let (class_id, prob) = row
            .iter()
            // skip bounding box coordinates
            .skip(4)
            .enumerate()
            .map(|(index, value)| (index, *value))
            .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
            .unwrap();
        if prob < 0.5 {
            continue;
        }
        let label = YOLOV8_CLASS_LABELS[class_id];
        let xc = row[0] / 640. * (img_width as f32);
        let yc = row[1] / 640. * (img_height as f32);
        let w = row[2] / 640. * (img_width as f32);
        let h = row[3] / 640. * (img_height as f32);
        boxes.push((
            BoundingBox {
                x1: xc - w / 2.,
                y1: yc - h / 2.,
                x2: xc + w / 2.,
                y2: yc + h / 2.,
            },
            label,
            prob,
        ));

        println!("{}: {:.2}%", label, prob * 100.);
    }

    boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2));
    let mut result = Vec::new();

    let mut gray = img.to_rgba8();
    let white = Rgba([255u8, 255u8, 255u8, 255u8]);

    while !boxes.is_empty() {
        result.push(boxes[0]);
        boxes = boxes
            .iter()
            .filter(|box1| intersection(&boxes[0].0, &box1.0) / union(&boxes[0].0, &box1.0) < 0.7)
            .copied()
            .collect();
    }

    let font = FontRef::try_from_slice(include_bytes!("../assets/Roboto.ttf")).unwrap();
    let height = 15.0;
    let scale = PxScale {
        x: height * 2.0,
        y: height,
    };

    for (box1, label, prob) in result.clone() {
        draw_hollow_rect_mut(
            &mut gray,
            Rect::at(box1.x1.floor() as i32, box1.y1.floor() as i32).of_size(
                box1.x2.floor() as u32 - box1.x1.floor() as u32,
                box1.y2.floor() as u32 - box1.y1.floor() as u32,
            ),
            white,
        );
        draw_text_mut(
            &mut gray,
            white,
            box1.x1 as i32 + 10,
            box1.y1 as i32 + 10,
            scale,
            &font,
            format!("{}: {:.2}%", label, prob * 100.0).as_str(),
        );
    }

    gray.save(output_dir).unwrap();

    println!("{:?}", result);

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
        - intersection(box1, box2)
}

const YOLOV8M_PATH: &str = "assets/yolov8m.onnx";

#[rustfmt::skip]
const YOLOV8_CLASS_LABELS:[&str;80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

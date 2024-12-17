from flask import Flask, request, jsonify
import os
import logging
import tempfile
import shutil
import concurrent.futures
import warnings
import torch
import ffmpeg
from google.cloud import storage
from google.cloud import aiplatform
from pyannote.audio import Pipeline
import stable_whisper

# Suppress warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
)

# Initialize Flask app
app = Flask(__name__)

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Initialize GCS client
storage_client = storage.Client()


# Helper Functions
def check_env_variable(var_name, default_value=None, require_value=False):
    value = os.getenv(var_name, default_value)
    if not value and require_value:
        logging.error(f"Environment variable {var_name} is required but not provided.")
        raise ValueError(f"Environment variable {var_name} is missing.")
    return value


def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logging.info(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}.")


def upload_to_gcs(bucket_name, destination_blob_name, source_file_name):
    if os.path.exists(source_file_name):
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        logging.info(f"Uploaded {source_file_name} to bucket {bucket_name} as {destination_blob_name}.")
    else:
        logging.error(f"File {source_file_name} not found. Skipping upload.")


def convert_audio_format(input_path, output_path):
    try:
        ffmpeg.input(input_path).output(output_path, ar=16000, ac=1).run(overwrite_output=True, quiet=True)
        logging.info(f"Converted {input_path} to {output_path} with 16kHz sample rate and mono channel.")
    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error converting {input_path}: {e.stderr.decode()}")
        raise


def download_model_from_gcs(gcs_uri, local_path):
    bucket_name, blob_prefix = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=blob_prefix)
    os.makedirs(local_path, exist_ok=True)

    for blob in blobs:
        if not blob.name.endswith("/"):
            local_file_path = os.path.join(local_path, os.path.basename(blob.name))
            blob.download_to_filename(local_file_path)
            logging.info(f"Downloaded {blob.name} to {local_file_path}.")


def initialize_diarization_pipeline(hf_token):
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization", use_auth_token=hf_token
        )
        diarization_pipeline.to(device)
        diarization_pipeline.overlap_detection = True
        return diarization_pipeline
    except Exception as e:
        logging.error(f"Error initializing diarization pipeline: {e}")
        raise


def run_inference_on_whisper_model(model, input_audio_path, output_text_path, diarization_pipeline):
    temp_audio_path = input_audio_path.replace(".wav", "_converted.wav")
    convert_audio_format(input_audio_path, temp_audio_path)

    logging.info(f"Running Whisper transcription on {temp_audio_path}...")
    model.to(device)
    result = model.transcribe(temp_audio_path, word_timestamps=True, fp16=torch.cuda.is_available())

    diarization_result = diarization_pipeline(temp_audio_path)
    speaker_segments = list(diarization_result.itertracks(yield_label=True))

    with open(output_text_path, "w") as output_file:
        for segment in result.segments:
            speaker = "UNKNOWN"
            for turn, _, label in speaker_segments:
                if turn.start <= segment.start < turn.end or turn.start < segment.end <= turn.end:
                    speaker = label
                    break
            output_file.write(f"[{segment.start:.2f} - {segment.end:.2f}] {speaker}: {segment.text}\n")
    logging.info(f"Transcription with speaker labels saved to {output_text_path}")


@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        # Parse request data
        data = request.get_json()

        # Required parameters from JSON payload
        project_id = data.get('PROJECT_ID', 'core-dev-435517')
        location = data.get('LOCATION', 'us-central1')
        model_name = data.get('MODEL_NAME', '194293600232669184')
        input_bucket = data.get('INPUT_BUCKET')
        input_audio_blob = data.get('INPUT_AUDIO_BLOB')
        output_bucket = data.get('OUTPUT_BUCKET')
        output_transcription_blob = data.get('OUTPUT_TRANSCRIPTION_BLOB')
        num_workers = data.get('NUM_WORKERS', 1)
        hf_auth_token = data.get('HF_AUTH_TOKEN')

        # Validate required parameters
        if not input_bucket:
            raise ValueError("INPUT_BUCKET is required but not provided.")
        if not input_audio_blob:
            raise ValueError("INPUT_AUDIO_BLOB is required but not provided.")
        if not output_bucket:
            raise ValueError("OUTPUT_BUCKET is required but not provided.")
        if not hf_auth_token:
            raise ValueError("HF_AUTH_TOKEN is required but not provided.")

        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        vertex_model = aiplatform.Model(model_name)
        artifact_uri = vertex_model.uri
        logging.info(f"Model GCS URI: {artifact_uri}")

        # Download model files
        local_model_dir = "/tmp/stable_whisper_model"
        os.makedirs(local_model_dir, exist_ok=True)
        download_model_from_gcs(artifact_uri, local_model_dir)

        # Load Whisper model
        try:
            model_path = os.path.join(local_model_dir, "small.pt")
            model = stable_whisper.load_model(model_path, device=device)
        except Exception as e:
            raise ValueError(f"Failed to load Whisper model: {e}")

        # Initialize diarization pipeline
        diarization_pipeline = initialize_diarization_pipeline(hf_auth_token)

        # Temporary directory for processing files
        temp_dir = tempfile.mkdtemp()

        # Check if the input is a single file or a prefix
        if input_audio_blob.endswith(".wav"):
            wav_blobs = [storage_client.bucket(input_bucket).blob(input_audio_blob)]
        else:
            blobs = storage_client.list_blobs(input_bucket, prefix=input_audio_blob)
            wav_blobs = [blob for blob in blobs if blob.name.endswith(".wav")]

        # Adjust number of workers
        num_workers = min(len(wav_blobs), int(num_workers))

        logging.info(f"Processing {len(wav_blobs)} files with {num_workers} workers.")

        def process_file(blob):
            try:
                # Download the input file
                local_audio_path = os.path.join(temp_dir, os.path.basename(blob.name))
                download_from_gcs(input_bucket, blob.name, local_audio_path)

                # Define output transcription file paths
                output_transcription_path = local_audio_path.replace(".wav", "_transcription.txt")
                output_transcription_blob_path = f"{output_transcription_blob}/{os.path.basename(output_transcription_path)}"

                # Run transcription and diarization
                run_inference_on_whisper_model(
                    model=model,
                    input_audio_path=local_audio_path,
                    output_text_path=output_transcription_path,
                    diarization_pipeline=diarization_pipeline
                )

                # Upload transcription result
                upload_to_gcs(output_bucket, output_transcription_blob_path, output_transcription_path)
            except Exception as e:
                logging.error(f"Error processing file {blob.name}: {str(e)}")

        # Process files using concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.map(process_file, wav_blobs)

        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        logging.info("All files processed successfully.")
        return jsonify({"message": "Transcription completed successfully", "status": "success"})

    except ValueError as ve:
        logging.error(str(ve))
        return jsonify({"message": str(ve), "status": "error"}), 400
    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}")
        return jsonify({"message": str(e), "status": "error"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083)

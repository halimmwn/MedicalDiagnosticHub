import requests
import pydicom
import hashlib
import io
import logging
from pydicom.uid import generate_uid
from PIL import Image

# Orthanc Configuration
ORTHANC_URL = "http://localhost:8042"
ORTHANC_AUTH = ('orthanc', 'orthanc')

def generate_study_uid_from_batch(batch_id):
    """Generates a consistent DICOM UID based on a batch string to group files together."""
    if not batch_id: return generate_uid()
    hash_object = hashlib.md5(batch_id.encode())
    hash_int = int(hash_object.hexdigest(), 16)
    return f"1.2.826.0.1.3680043.9.{str(hash_int)[:20]}"

def upload_dicom(file_bytes, batch_id=None):
    """
    Uploads raw DICOM bytes to Orthanc.
    Returns (instance_id, study_uid, num_frames) or None.
    """
    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        
        # If batch_id provided, force them into the same Study
        if batch_id:
            ds.StudyInstanceUID = generate_study_uid_from_batch(batch_id)
            
        with io.BytesIO() as out:
            ds.save_as(out)
            modified_bytes = out.getvalue()

        resp = requests.post(f"{ORTHANC_URL}/instances", data=modified_bytes, auth=ORTHANC_AUTH)
        if resp.status_code != 200: 
            logging.error(f"Orthanc Upload Failed: {resp.text}")
            return None
        
        instance_id = resp.json()['ID']
        
        # Get Study UID for Viewer Link
        tags = requests.get(f"{ORTHANC_URL}/instances/{instance_id}/tags", auth=ORTHANC_AUTH).json()
        study_uid = tags.get('0020,000d', {}).get('Value')
        
        # Check frames (for multi-frame DICOMs)
        frames = 1
        try:
             val = tags.get('0028,0008', {}).get('Value')
             if val: frames = int(val)
        except: pass

        return instance_id, study_uid, frames
    except Exception as e:
        logging.error(f"Orthanc Client Error: {e}")
        return None

def get_orthanc_preview(instance_id, frame=0):
    """
    Fetches a rendered preview image (JPG/PNG) from Orthanc.
    """
    try:
        url = f"{ORTHANC_URL}/instances/{instance_id}/frames/{frame}/preview" if frame > 0 else f"{ORTHANC_URL}/instances/{instance_id}/preview"
        resp = requests.get(url, auth=ORTHANC_AUTH)
        if resp.status_code == 200:
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        logging.error(f"Preview Fetch Error: {e}")
    return None
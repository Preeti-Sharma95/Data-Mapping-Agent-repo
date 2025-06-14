#!/usr/bin/env python3
"""
Simple FastAPI Web Service for Data Mapping Agent
Standalone version without DataMappingAPI dependency
"""

import os
import json
import logging
import tempfile
import pandas as pd
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Data Mapping Agent API",
    description="Simple data mapping service with file upload and basic auto-mapping",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for session management
uploaded_file_path: Optional[str] = None
insights_data: Optional[Dict] = None
mapping_results: Optional[Dict] = None
mapped_file_path: Optional[str] = None

# Standard target schema for mapping
STANDARD_TARGET_SCHEMA = [
    {"name": "customer_id", "description": "Unique customer identifier", "data_type": "string"},
    {"name": "customer_name", "description": "Full customer name", "data_type": "string"},
    {"name": "first_name", "description": "Customer first name", "data_type": "string"},
    {"name": "last_name", "description": "Customer last name", "data_type": "string"},
    {"name": "email", "description": "Customer email address", "data_type": "string"},
    {"name": "phone", "description": "Customer phone number", "data_type": "string"},
    {"name": "address", "description": "Customer street address", "data_type": "string"},
    {"name": "city", "description": "Customer city", "data_type": "string"},
    {"name": "state", "description": "Customer state/province", "data_type": "string"},
    {"name": "country", "description": "Customer country", "data_type": "string"},
    {"name": "postal_code", "description": "Customer postal/zip code", "data_type": "string"},
    {"name": "date_of_birth", "description": "Customer date of birth", "data_type": "date"},
    {"name": "registration_date", "description": "Customer registration date", "data_type": "datetime"},
    {"name": "status", "description": "Customer status (active, inactive, etc.)", "data_type": "string"},
    {"name": "category", "description": "Customer category or type", "data_type": "string"}
]


def simple_field_matching(source_columns: List[str]) -> List[Dict]:
    """
    Simple field matching logic based on keyword similarity
    This replaces the complex LLM-based mapping for demo purposes
    """

    # Keyword mappings for common field names
    keyword_mappings = {
        'customer_id': ['id', 'customer_id', 'cust_id', 'customer_number', 'account_id'],
        'customer_name': ['name', 'customer_name', 'full_name', 'customer', 'client_name'],
        'first_name': ['first_name', 'fname', 'first', 'given_name'],
        'last_name': ['last_name', 'lname', 'last', 'surname', 'family_name'],
        'email': ['email', 'email_address', 'e_mail', 'mail', 'contact_email'],
        'phone': ['phone', 'phone_number', 'telephone', 'mobile', 'contact_number'],
        'address': ['address', 'street', 'address_line_1', 'street_address'],
        'city': ['city', 'town', 'locality'],
        'state': ['state', 'province', 'region', 'state_province'],
        'country': ['country', 'nation', 'country_code'],
        'postal_code': ['postal_code', 'zip', 'zip_code', 'postcode', 'postal'],
        'date_of_birth': ['dob', 'date_of_birth', 'birth_date', 'birthdate'],
        'registration_date': ['registration_date', 'reg_date', 'created_date', 'signup_date'],
        'status': ['status', 'state', 'account_status', 'customer_status'],
        'category': ['category', 'type', 'customer_type', 'segment']
    }

    mappings = []

    for source_col in source_columns:
        source_col_lower = source_col.lower().strip()
        best_match = None
        best_score = 0

        # Check each target field
        for target_field in STANDARD_TARGET_SCHEMA:
            target_name = target_field['name']
            keywords = keyword_mappings.get(target_name, [target_name])

            # Calculate similarity score
            score = 0
            for keyword in keywords:
                if keyword.lower() in source_col_lower:
                    score = max(score, len(keyword) / len(source_col_lower))
                if source_col_lower in keyword.lower():
                    score = max(score, len(source_col_lower) / len(keyword))
                if keyword.lower() == source_col_lower:
                    score = 1.0
                    break

            if score > best_score and score > 0.3:  # Minimum threshold
                best_score = score
                best_match = target_field

        # Create mapping result
        mapping = {
            'source_field': {
                'name': source_col,
                'description': f"Source column '{source_col}'"
            },
            'best_match': best_match,
            'similarity_score': best_score,
            'mapping_type': 'auto' if best_score >= 0.8 else 'manual_review'
        }

        mappings.append(mapping)

    return mappings


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Data Mapping Agent API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - Upload file",
            "insights": "GET /insights - Get file insights",
            "auto_map": "GET /auto-map - Auto-map with simple matching",
            "download": "GET /download - Download mapped file",
            "health": "GET /health - Health check",
            "docs": "GET /docs - API documentation"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Data Mapping Agent API",
        "version": "1.0.0"
    }


# File Upload endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload file to backend
    Supports CSV, Excel (xlsx, xls), and JSON files
    """
    global uploaded_file_path, insights_data, mapping_results, mapped_file_path

    try:
        # Reset previous session data
        insights_data = None
        mapping_results = None
        mapped_file_path = None

        # Validate file type
        allowed_extensions = ['.csv', '.xlsx', '.xls', '.json']
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            uploaded_file_path = tmp_file.name

        # Generate basic insights
        try:
            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(uploaded_file_path)
            elif file_extension == '.json':
                df = pd.read_json(uploaded_file_path)

            # Create insights
            insights_data = {
                "filename": file.filename,
                "file_type": file_extension,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "file_size_bytes": len(content),
                "data_types": df.dtypes.astype(str).to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "sample_data": df.head(3).to_dict('records') if len(df) > 0 else [],
                "upload_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            # If file parsing fails, create basic insights
            insights_data = {
                "filename": file.filename,
                "file_type": file_extension,
                "file_size_bytes": len(content),
                "upload_timestamp": datetime.now().isoformat(),
                "error": f"Could not parse file: {str(e)}"
            }

        logger.info(f"File uploaded successfully: {file.filename}")

        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "file_type": file_extension,
            "size_bytes": len(content),
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


# Get Insights endpoint
@app.get("/insights")
async def get_insights():
    """
    Retrieve backend-generated insights about the uploaded file
    """
    global insights_data

    if insights_data is None:
        raise HTTPException(
            status_code=404,
            detail="No file uploaded yet. Please upload a file first using POST /upload"
        )

    return {
        "insights": insights_data,
        "generated_at": datetime.now().isoformat()
    }


# Auto-Map endpoint
@app.get("/auto-map")
async def auto_map():
    """
    Use simple keyword matching to auto-map uploaded data against standard schema
    """
    global uploaded_file_path, mapping_results

    if uploaded_file_path is None:
        raise HTTPException(
            status_code=404,
            detail="No file uploaded yet. Please upload a file first using POST /upload"
        )

    try:
        # Read the uploaded file
        file_extension = os.path.splitext(uploaded_file_path)[1].lower()

        if file_extension == '.csv':
            df = pd.read_csv(uploaded_file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file_path)
        elif file_extension == '.json':
            df = pd.read_json(uploaded_file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type for mapping")

        logger.info(f"Starting auto-mapping with {len(df.columns)} source fields")

        # Use simple field matching
        mappings = simple_field_matching(df.columns.tolist())

        # Calculate statistics
        auto_mapped = sum(1 for m in mappings if m['mapping_type'] == 'auto')
        manual_review = sum(1 for m in mappings if m['mapping_type'] == 'manual_review')
        high_confidence = sum(1 for m in mappings if m['similarity_score'] >= 0.8)

        # Create session result
        session_id = str(uuid.uuid4())
        mapping_results = {
            "session_id": session_id,
            "total_mappings": len(mappings),
            "auto_mapped": auto_mapped,
            "manual_review_needed": manual_review,
            "high_confidence": high_confidence,
            "mappings": mappings,
            "processing_time_seconds": 0.5  # Simulated processing time
        }

        # Extract mapping summary
        mappings_summary = []
        for mapping in mappings:
            mappings_summary.append({
                "source_field": mapping['source_field']['name'],
                "target_field": mapping['best_match']['name'] if mapping['best_match'] else 'No match',
                "confidence": round(mapping['similarity_score'] * 100, 2),
                "mapping_type": mapping['mapping_type']
            })

        logger.info(f"Auto-mapping completed with {len(mappings_summary)} mappings")

        return {
            "message": "Auto-mapping completed successfully",
            "session_id": session_id,
            "source_columns": df.columns.tolist(),
            "total_mappings": len(mappings),
            "auto_mapped": auto_mapped,
            "high_confidence": high_confidence,
            "mapping_summary": mappings_summary,
            "processing_time_seconds": 0.5,
            "processed_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Auto-mapping failed: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-mapping failed: {str(e)}")


# Download endpoint
@app.get("/download")
async def download_mapped_file():
    """
    Download the processed/mapped version of the uploaded file
    Returns the mapping results as CSV
    """
    global uploaded_file_path, mapping_results, mapped_file_path

    if uploaded_file_path is None:
        raise HTTPException(
            status_code=404,
            detail="No file uploaded yet. Please upload a file first using POST /upload"
        )

    if mapping_results is None:
        raise HTTPException(
            status_code=404,
            detail="No mapping performed yet. Please run auto-mapping first using GET /auto-map"
        )

    try:
        # Create mapping information DataFrame
        mapping_info = []
        for mapping in mapping_results.get('mappings', []):
            mapping_info.append({
                'source_field': mapping['source_field']['name'],
                'target_field': mapping['best_match']['name'] if mapping['best_match'] else 'No match',
                'confidence_percentage': round(mapping['similarity_score'] * 100, 2),
                'mapping_type': mapping['mapping_type'],
                'target_description': mapping['best_match']['description'] if mapping['best_match'] else ''
            })

        # Create temporary file for download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mapping_filename = f"mapping_results_{timestamp}.csv"
        mapping_path = os.path.join(tempfile.gettempdir(), mapping_filename)

        # Save mapping results
        mapping_df = pd.DataFrame(mapping_info)
        mapping_df.to_csv(mapping_path, index=False)

        mapped_file_path = mapping_path

        logger.info(f"Prepared mapped file for download: {mapping_filename}")

        return FileResponse(
            path=mapping_path,
            filename=mapping_filename,
            media_type='text/csv',
            headers={"Content-Disposition": f"attachment; filename={mapping_filename}"}
        )

    except Exception as e:
        logger.error(f"Download preparation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


# Additional helpful endpoints
@app.get("/status")
async def get_status():
    """Get current session status"""
    return {
        "file_uploaded": uploaded_file_path is not None,
        "insights_generated": insights_data is not None,
        "mapping_completed": mapping_results is not None,
        "ready_for_download": mapped_file_path is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.delete("/reset")
async def reset_session():
    """Reset current session"""
    global uploaded_file_path, insights_data, mapping_results, mapped_file_path

    # Clean up temporary files
    for file_path in [uploaded_file_path, mapped_file_path]:
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except:
                pass

    uploaded_file_path = None
    insights_data = None
    mapping_results = None
    mapped_file_path = None

    return {"message": "Session reset successfully"}


if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "fastapi_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
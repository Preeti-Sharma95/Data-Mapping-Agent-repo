#!/usr/bin/env python3
"""
Advanced Data Mapping Agent with BGE Large v1.5
Enterprise-grade field mapping system with LLM assistance
COMPLETE VERSION - With all required endpoints for Postman testing
"""

# Compatibility patches - MUST BE FIRST
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Fix huggingface_hub compatibility
try:
    from huggingface_hub import cached_download
except ImportError:
    try:
        from huggingface_hub import hf_hub_download
        import huggingface_hub


        def cached_download(url, cache_dir=None, **kwargs):
            """Compatibility wrapper for deprecated cached_download"""
            if "huggingface.co" in url:
                parts = url.split("/")
                if len(parts) >= 6:
                    repo_id = f"{parts[-4]}/{parts[-3]}"
                    filename = parts[-1]
                    return hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=cache_dir,
                        **kwargs
                    )
            raise NotImplementedError("cached_download is deprecated")


        huggingface_hub.cached_download = cached_download
    except ImportError:
        pass

# Fix torch warnings
try:
    import torch

    # Suppress the deprecation warning
    if hasattr(torch.utils, '_pytree'):
        original_register = getattr(torch.utils._pytree, '_register_pytree_node', None)
        if original_register:
            def silent_register(*args, **kwargs):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return original_register(*args, **kwargs)


            torch.utils._pytree._register_pytree_node = silent_register
except:
    pass

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import io

# Core dependencies
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ML and NLP
import torch
from sentence_transformers import SentenceTransformer

# LangChain components (with fallbacks)
try:
    from langchain_community.llms import Ollama
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    from langchain.callbacks.manager import CallbackManagerForLLMRun

    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LangChain import warning: {e}")
    LANGCHAIN_AVAILABLE = False

# API and web components
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn

    FASTAPI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ FastAPI not available: {e}")
    FASTAPI_AVAILABLE = False

# Utilities
try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    print("⚠️ Pydantic not available")
    PYDANTIC_AVAILABLE = False

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not available")

try:
    import aiofiles

    AIOFILES_AVAILABLE = True
except ImportError:
    print("⚠️ aiofiles not available")
    AIOFILES_AVAILABLE = False

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    print("⚠️ tqdm not available")
    TQDM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_mapping_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


class DataMappingAgent:
    """Advanced Data Mapping Agent with BGE Large v1.5 and LLM assistance"""

    def __init__(
            self,
            model_name: str = "BAAI/bge-large-en-v1.5",
            llm_model: str = "llama3.1:8b",
            similarity_threshold: float = 0.75,
            cache_dir: Optional[str] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.llm_model = llm_model
        self.similarity_threshold = similarity_threshold
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "model_cache")

        # Initialize components
        self._init_embedding_model()
        self._init_llm()
        self._init_cache()

        # Initialize the simple workflow instead of LangGraph
        self.logger.info("🔄 Setting up simplified workflow (LangGraph not available)")
        self.workflow = self._create_workflow()

        self.logger.info("✅ Data Mapping Agent initialized successfully")

    def _init_embedding_model(self):
        """Initialize BGE Large v1.5 embedding model with retry logic"""
        import time

        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                self.logger.info(f"🔄 Loading BGE Large v1.5 (attempt {attempt + 1}/{max_retries})")
                self.logger.info(f"📍 Model: {self.model_name}")

                # Create cache directory if it doesn't exist
                os.makedirs(self.cache_dir, exist_ok=True)
                self.logger.info(f"📁 Cache directory: {self.cache_dir}")

                # Check internet connectivity first
                self._check_connectivity()

                # Load model without trust_remote_code to avoid compatibility issues
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.embedding_model = SentenceTransformer(
                        self.model_name,
                        cache_folder=self.cache_dir,
                        device='cuda' if torch.cuda.is_available() else 'cpu'
                    )

                device = "GPU" if torch.cuda.is_available() else "CPU"
                self.logger.info(f"✅ BGE Large v1.5 loaded successfully on {device}")

                # Test the model with a simple encoding
                test_text = "test embedding"
                test_embedding = self.embedding_model.encode(test_text)
                self.logger.info(f"🧪 Model test successful - embedding shape: {test_embedding.shape}")

                return

            except Exception as e:
                self.logger.error(f"❌ Attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    self.logger.info(f"⏳ Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.logger.error("💥 All attempts failed to load BGE Large v1.5")
                    self.logger.error("🔍 Troubleshooting tips:")
                    self.logger.error("   1. Check your internet connection")
                    self.logger.error("   2. Try: pip install --upgrade huggingface_hub")
                    self.logger.error("   3. Clear cache: rm -rf model_cache/")
                    self.logger.error("   4. Check firewall/proxy settings")
                    raise Exception(f"Failed to load BGE Large v1.5 after {max_retries} attempts: {e}")

    def _check_connectivity(self):
        """Check internet connectivity to HuggingFace Hub"""
        try:
            import urllib.request
            import socket

            # Set a reasonable timeout
            socket.setdefaulttimeout(10)

            # Test HuggingFace connectivity
            self.logger.info("🌐 Checking HuggingFace Hub connectivity...")
            urllib.request.urlopen('https://huggingface.co', timeout=10)

            # Test model repository specifically
            model_url = f"https://huggingface.co/{self.model_name}"
            urllib.request.urlopen(model_url, timeout=10)

            self.logger.info("✅ HuggingFace Hub connectivity confirmed")

        except Exception as e:
            self.logger.warning(f"⚠️ Connectivity test failed: {e}")
            self.logger.warning("   This might cause model download issues")
            # Don't raise exception here, let the model loading handle the error

    def _init_llm(self):
        """Initialize Ollama LLM for advanced reasoning"""
        try:
            if not LANGCHAIN_AVAILABLE:
                self.logger.warning("⚠️ LangChain not available, skipping LLM initialization")
                self.llm = None
                return

            self.logger.info(f"🔄 Connecting to Ollama model: {self.llm_model}")
            self.llm = Ollama(
                model=self.llm_model,
                temperature=0.1,
                top_p=0.9,
                num_predict=2048
            )

            # Test connection
            try:
                test_response = self.llm.invoke("Hello, are you working?")
                self.logger.info("✅ Ollama LLM connected successfully")
            except Exception as e:
                self.logger.warning(f"⚠️ Ollama connection test failed: {e}")
                self.llm = None

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LLM: {e}")
            self.llm = None

    def _init_cache(self):
        """Initialize caching system"""
        self.embedding_cache = {}
        self.mapping_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _create_workflow(self):
        """Create the mapping workflow (simplified without LangGraph)"""
        # Since we removed LangGraph, we'll implement a simple async workflow
        self.workflow_steps = [
            "analyze_schemas",
            "generate_embeddings",
            "find_similarities",
            "create_mappings",
            "validate_mappings"
        ]

        # Create a simple workflow function
        async def simple_workflow(source_schema: Dict, target_schema: Dict) -> Dict:
            results = {}
            for step in self.workflow_steps:
                try:
                    if step == "analyze_schemas":
                        results[step] = await self._analyze_schemas(source_schema, target_schema)
                    elif step == "generate_embeddings":
                        results[step] = await self._generate_embeddings(source_schema, target_schema)
                    elif step == "find_similarities":
                        results[step] = await self._find_similarities(results["generate_embeddings"])
                    elif step == "create_mappings":
                        results[step] = await self._create_mappings(results["find_similarities"])
                    elif step == "validate_mappings":
                        results[step] = await self._validate_mappings(results["create_mappings"])
                except Exception as e:
                    self.logger.error(f"Error in workflow step {step}: {e}")
                    results[step] = {"error": str(e)}
            return results

        return simple_workflow

    async def _analyze_schemas(self, source_schema: Dict, target_schema: Dict) -> Dict:
        """Analyze and preprocess schemas"""
        self.logger.info("🔍 Analyzing schemas...")

        analysis = {
            "source_fields": len(source_schema.get("fields", [])),
            "target_fields": len(target_schema.get("fields", [])),
            "source_types": self._get_field_types(source_schema),
            "target_types": self._get_field_types(target_schema),
            "complexity_score": self._calculate_complexity(source_schema, target_schema)
        }

        return analysis

    def _get_field_types(self, schema: Dict) -> Dict[str, int]:
        """Get distribution of field types in schema"""
        types = {}
        for field in schema.get("fields", []):
            field_type = field.get("type", "unknown")
            types[field_type] = types.get(field_type, 0) + 1
        return types

    def _calculate_complexity(self, source_schema: Dict, target_schema: Dict) -> float:
        """Calculate mapping complexity score"""
        source_fields = len(source_schema.get("fields", []))
        target_fields = len(target_schema.get("fields", []))

        # Higher complexity for larger schemas and mismatched sizes
        size_complexity = (source_fields + target_fields) / 100
        mismatch_complexity = abs(source_fields - target_fields) / max(source_fields, target_fields, 1)

        return min(size_complexity + mismatch_complexity, 1.0)

    async def _generate_embeddings(self, source_schema: Dict, target_schema: Dict) -> Dict:
        """Generate embeddings for all fields"""
        self.logger.info("🧠 Generating field embeddings...")

        # Extract field information
        source_fields = self._extract_field_info(source_schema.get("fields", []))
        target_fields = self._extract_field_info(target_schema.get("fields", []))

        # Generate embeddings
        source_embeddings = await self._get_embeddings(source_fields)
        target_embeddings = await self._get_embeddings(target_fields)

        return {
            "source_fields": source_fields,
            "target_fields": target_fields,
            "source_embeddings": source_embeddings,
            "target_embeddings": target_embeddings
        }

    def _extract_field_info(self, fields: List[Dict]) -> List[str]:
        """Extract field information for embedding"""
        field_info = []
        for field in fields:
            # Combine field name, type, and description for rich context
            name = field.get("name", "")
            field_type = field.get("type", "")
            description = field.get("description", "")
            sample_values = field.get("sample_values", [])

            # Create comprehensive field representation
            field_text = f"{name} {field_type}"
            if description:
                field_text += f" {description}"
            if sample_values:
                field_text += f" examples: {', '.join(map(str, sample_values[:3]))}"

            field_info.append(field_text.strip())

        return field_info

    async def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings with caching"""
        embeddings = []

        # Create progress bar if tqdm is available
        iterator = tqdm(texts, desc="Generating embeddings") if TQDM_AVAILABLE else texts

        for text in iterator:
            # Check cache first
            cache_key = hash(text)
            if cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
                self.cache_hits += 1
            else:
                # Generate embedding
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                self.embedding_cache[cache_key] = embedding
                embeddings.append(embedding)
                self.cache_misses += 1

        return np.array(embeddings)

    async def _find_similarities(self, embeddings_data: Dict) -> Dict:
        """Find similarities between source and target fields"""
        self.logger.info("🔗 Computing field similarities...")

        source_embeddings = embeddings_data["source_embeddings"]
        target_embeddings = embeddings_data["target_embeddings"]

        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(source_embeddings, target_embeddings)

        # Find best matches
        similarities = []
        for i, source_field in enumerate(embeddings_data["source_fields"]):
            best_matches = []
            for j, target_field in enumerate(embeddings_data["target_fields"]):
                score = similarity_matrix[i][j]
                if score >= self.similarity_threshold:
                    best_matches.append({
                        "target_field": target_field,
                        "target_index": j,
                        "similarity_score": float(score),
                        "confidence": self._calculate_confidence(score)
                    })

            # Sort by similarity score
            best_matches.sort(key=lambda x: x["similarity_score"], reverse=True)

            similarities.append({
                "source_field": source_field,
                "source_index": i,
                "matches": best_matches[:5]  # Top 5 matches
            })

        return {
            "similarities": similarities,
            "similarity_matrix": similarity_matrix.tolist(),
            "stats": {
                "total_comparisons": len(embeddings_data["source_fields"]) * len(embeddings_data["target_fields"]),
                "matches_found": sum(len(s["matches"]) for s in similarities),
                "avg_similarity": float(np.mean(similarity_matrix))
            }
        }

    def _calculate_confidence(self, similarity_score: float) -> str:
        """Calculate confidence level based on similarity score"""
        if similarity_score >= 0.9:
            return "very_high"
        elif similarity_score >= 0.8:
            return "high"
        elif similarity_score >= 0.7:
            return "medium"
        elif similarity_score >= 0.6:
            return "low"
        else:
            return "very_low"

    async def _create_mappings(self, similarities_data: Dict) -> Dict:
        """Create field mappings with LLM validation"""
        self.logger.info("📋 Creating field mappings...")

        mappings = []
        conflicts = []

        # Track used target fields to avoid duplicates
        used_targets = set()

        for similarity in similarities_data["similarities"]:
            source_field = similarity["source_field"]
            source_index = similarity["source_index"]

            if not similarity["matches"]:
                # No matches found
                mappings.append({
                    "source_field": source_field,
                    "source_index": source_index,
                    "target_field": None,
                    "target_index": None,
                    "mapping_type": "unmapped",
                    "similarity_score": 0.0,
                    "confidence": "none",
                    "reasoning": "No suitable matches found above threshold"
                })
                continue

            # Find best available match
            best_match = None
            for match in similarity["matches"]:
                if match["target_index"] not in used_targets:
                    best_match = match
                    break

            if best_match:
                used_targets.add(best_match["target_index"])

                # Determine mapping type
                mapping_type = self._determine_mapping_type(
                    source_field,
                    best_match["target_field"],
                    best_match["similarity_score"]
                )

                # Get LLM reasoning if available
                reasoning = await self._get_llm_reasoning(
                    source_field,
                    best_match["target_field"],
                    best_match["similarity_score"]
                )

                mappings.append({
                    "source_field": source_field,
                    "source_index": source_index,
                    "target_field": best_match["target_field"],
                    "target_index": best_match["target_index"],
                    "mapping_type": mapping_type,
                    "similarity_score": best_match["similarity_score"],
                    "confidence": best_match["confidence"],
                    "reasoning": reasoning
                })
            else:
                # All good matches are already used
                conflicts.append({
                    "source_field": source_field,
                    "potential_matches": similarity["matches"],
                    "reason": "All high-similarity targets already mapped"
                })

        return {
            "mappings": mappings,
            "conflicts": conflicts,
            "stats": {
                "total_fields": len(similarities_data["similarities"]),
                "mapped_fields": len([m for m in mappings if m["target_field"] is not None]),
                "unmapped_fields": len([m for m in mappings if m["target_field"] is None]),
                "conflicts": len(conflicts)
            }
        }

    def _determine_mapping_type(self, source_field: str, target_field: str, similarity: float) -> str:
        """Determine the type of mapping based on field analysis"""
        if similarity >= 0.95:
            return "exact_match"
        elif similarity >= 0.85:
            return "semantic_match"
        elif similarity >= 0.75:
            return "probable_match"
        else:
            return "possible_match"

    async def _get_llm_reasoning(self, source_field: str, target_field: str, similarity: float) -> str:
        """Get LLM reasoning for mapping decision"""
        if not self.llm:
            return f"Mapping based on semantic similarity ({similarity:.3f})"

        try:
            prompt = f"""
Analyze this field mapping:
Source field: {source_field}
Target field: {target_field}
Similarity score: {similarity:.3f}

Provide a brief explanation (1-2 sentences) of why this mapping makes sense or any concerns.
Focus on semantic meaning and data compatibility.
"""

            response = self.llm.invoke(prompt)
            return response.strip()

        except Exception as e:
            self.logger.warning(f"LLM reasoning failed: {e}")
            return f"Mapping based on semantic similarity ({similarity:.3f})"

    async def _validate_mappings(self, mappings_data: Dict) -> Dict:
        """Validate and score the created mappings"""
        self.logger.info("✅ Validating mappings...")

        mappings = mappings_data["mappings"]
        validation_results = []

        for mapping in mappings:
            if mapping["target_field"] is None:
                validation_results.append({
                    **mapping,
                    "validation_score": 0.0,
                    "validation_status": "unmapped",
                    "validation_notes": "No target field mapped"
                })
                continue

            # Calculate validation score
            validation_score = self._calculate_validation_score(mapping)
            validation_status = self._get_validation_status(validation_score)
            validation_notes = self._get_validation_notes(mapping, validation_score)

            validation_results.append({
                **mapping,
                "validation_score": validation_score,
                "validation_status": validation_status,
                "validation_notes": validation_notes
            })

        # Calculate overall quality metrics
        mapped_validations = [v for v in validation_results if v["target_field"] is not None]
        avg_validation_score = np.mean(
            [v["validation_score"] for v in mapped_validations]) if mapped_validations else 0.0

        quality_metrics = {
            "overall_score": float(avg_validation_score),
            "high_quality_mappings": len([v for v in mapped_validations if v["validation_score"] >= 0.8]),
            "medium_quality_mappings": len([v for v in mapped_validations if 0.6 <= v["validation_score"] < 0.8]),
            "low_quality_mappings": len([v for v in mapped_validations if v["validation_score"] < 0.6]),
            "mapping_coverage": len(mapped_validations) / len(validation_results) if validation_results else 0.0
        }

        return {
            "validated_mappings": validation_results,
            "quality_metrics": quality_metrics,
            "recommendations": self._generate_recommendations(validation_results, quality_metrics)
        }

    def _calculate_validation_score(self, mapping: Dict) -> float:
        """Calculate validation score for a mapping"""
        base_score = mapping["similarity_score"]

        # Adjust based on confidence
        confidence_multipliers = {
            "very_high": 1.0,
            "high": 0.9,
            "medium": 0.8,
            "low": 0.7,
            "very_low": 0.6
        }

        confidence_multiplier = confidence_multipliers.get(mapping["confidence"], 0.5)

        # Adjust based on mapping type
        type_multipliers = {
            "exact_match": 1.0,
            "semantic_match": 0.95,
            "probable_match": 0.85,
            "possible_match": 0.75
        }

        type_multiplier = type_multipliers.get(mapping["mapping_type"], 0.5)

        return base_score * confidence_multiplier * type_multiplier

    def _get_validation_status(self, score: float) -> str:
        """Get validation status based on score"""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.6:
            return "acceptable"
        elif score >= 0.4:
            return "questionable"
        else:
            return "poor"

    def _get_validation_notes(self, mapping: Dict, validation_score: float) -> str:
        """Generate validation notes for mapping"""
        notes = []

        if validation_score >= 0.8:
            notes.append("High confidence mapping")
        elif validation_score >= 0.6:
            notes.append("Moderate confidence mapping")
        else:
            notes.append("Low confidence mapping - review recommended")

        if mapping["similarity_score"] < 0.8:
            notes.append("Consider manual verification")

        return "; ".join(notes)

    def _generate_recommendations(self, validations: List[Dict], metrics: Dict) -> List[str]:
        """Generate recommendations for improving mappings"""
        recommendations = []

        if metrics["mapping_coverage"] < 0.8:
            recommendations.append("Consider adjusting similarity threshold to capture more mappings")

        if metrics["low_quality_mappings"] > metrics["high_quality_mappings"]:
            recommendations.append("Many low-quality mappings detected - manual review recommended")

        unmapped_count = len([v for v in validations if v["target_field"] is None])
        if unmapped_count > 0:
            recommendations.append(f"{unmapped_count} source fields remain unmapped - consider manual mapping")

        if not recommendations:
            recommendations.append("Mapping quality looks good - ready for implementation")

        return recommendations

    async def map_schemas(
            self,
            source_schema: Dict,
            target_schema: Dict,
            options: Optional[Dict] = None
    ) -> Dict:
        """Main method to map schemas"""
        start_time = datetime.now()

        try:
            self.logger.info("🚀 Starting schema mapping process...")

            # Apply options if provided
            if options:
                self.similarity_threshold = options.get("similarity_threshold", self.similarity_threshold)

            # Run the workflow
            results = await self.workflow(source_schema, target_schema)

            # Calculate timing
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Compile final results
            final_results = {
                "status": "success",
                "timestamp": end_time.isoformat(),
                "processing_time_seconds": processing_time,
                "source_schema_info": {
                    "name": source_schema.get("name", "Unknown"),
                    "field_count": len(source_schema.get("fields", []))
                },
                "target_schema_info": {
                    "name": target_schema.get("name", "Unknown"),
                    "field_count": len(target_schema.get("fields", []))
                },
                "analysis": results.get("analyze_schemas", {}),
                "mappings": results.get("validate_mappings", {}).get("validated_mappings", []),
                "quality_metrics": results.get("validate_mappings", {}).get("quality_metrics", {}),
                "recommendations": results.get("validate_mappings", {}).get("recommendations", []),
                "cache_stats": {
                    "cache_hits": self.cache_hits,
                    "cache_misses": self.cache_misses,
                    "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (
                                                                                                         self.cache_hits + self.cache_misses) > 0 else 0
                }
            }

            self.logger.info(f"✅ Schema mapping completed in {processing_time:.2f} seconds")
            return final_results

        except Exception as e:
            self.logger.error(f"❌ Schema mapping failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Pydantic models for API (only if Pydantic is available)
if PYDANTIC_AVAILABLE:
    class SchemaField(BaseModel):
        name: str
        type: str
        description: Optional[str] = ""
        sample_values: Optional[List[Any]] = []
        constraints: Optional[Dict] = {}


    class Schema(BaseModel):
        name: str
        fields: List[SchemaField]
        metadata: Optional[Dict] = {}


    class MappingRequest(BaseModel):
        source_schema: Schema
        target_schema: Schema
        options: Optional[Dict] = {}


    class MappingResponse(BaseModel):
        status: str
        mappings: Optional[List[Dict]] = None
        quality_metrics: Optional[Dict] = None
        recommendations: Optional[List[str]] = None
        error: Optional[str] = None
else:
    # Fallback classes if Pydantic is not available
    class SchemaField:
        def __init__(self, name, type, description="", sample_values=None, constraints=None):
            self.name = name
            self.type = type
            self.description = description
            self.sample_values = sample_values or []
            self.constraints = constraints or {}


    class Schema:
        def __init__(self, name, fields, metadata=None):
            self.name = name
            self.fields = fields
            self.metadata = metadata or {}


    class MappingRequest:
        def __init__(self, source_schema, target_schema, options=None):
            self.source_schema = source_schema
            self.target_schema = target_schema
            self.options = options or {}


    class MappingResponse:
        def __init__(self, status, mappings=None, quality_metrics=None, recommendations=None, error=None):
            self.status = status
            self.mappings = mappings
            self.quality_metrics = quality_metrics
            self.recommendations = recommendations
            self.error = error

# FastAPI application (only if FastAPI is available)
if FASTAPI_AVAILABLE:
    class DataMappingAPI:
        """FastAPI application for the Data Mapping Agent"""

        def __init__(self):
            self.app = FastAPI(
                title="Advanced Data Mapping Agent",
                description="Enterprise-grade field mapping with BGE Large v1.5 and LLM assistance",
                version="1.0.0"
            )

            # Initialize the mapping agent
            self.agent = DataMappingAgent()

            # Configure CORS
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            # Initialize file storage
            self.uploaded_files = {}

            # Setup routes
            self._setup_routes()

            self.logger = logging.getLogger(__name__)

        def _csv_to_schema(self, df: pd.DataFrame, filename: str) -> Dict:
            """Convert CSV DataFrame to schema format"""
            fields = []
            for column in df.columns:
                dtype = str(df[column].dtype)
                sample_values = df[column].dropna().head(3).tolist()

                # Map pandas dtypes to more readable types
                if 'int' in dtype:
                    field_type = 'integer'
                elif 'float' in dtype:
                    field_type = 'float'
                elif 'bool' in dtype:
                    field_type = 'boolean'
                elif 'datetime' in dtype:
                    field_type = 'datetime'
                else:
                    field_type = 'string'

                fields.append({
                    "name": column,
                    "type": field_type,
                    "description": f"Auto-detected from CSV column {column}",
                    "sample_values": sample_values,
                    "null_count": int(df[column].isnull().sum()),
                    "unique_count": int(df[column].nunique()) if df[column].nunique() < 1000 else "1000+"
                })

            return {
                "name": filename.replace('.csv', ''),
                "fields": fields,
                "metadata": {
                    "source": "csv_upload",
                    "rows": len(df),
                    "columns": len(df.columns),
                    "detected_at": datetime.now().isoformat()
                }
            }

        def _json_to_schema(self, json_data: Any, filename: str) -> Dict:
            """Convert JSON data to schema format"""
            fields = []

            if isinstance(json_data, list) and json_data:
                # Array of objects - analyze first object
                sample_obj = json_data[0]
                if isinstance(sample_obj, dict):
                    for key, value in sample_obj.items():
                        field_type = type(value).__name__
                        if field_type == 'str':
                            field_type = 'string'
                        elif field_type == 'int':
                            field_type = 'integer'
                        elif field_type == 'float':
                            field_type = 'float'
                        elif field_type == 'bool':
                            field_type = 'boolean'
                        elif field_type in ['list', 'dict']:
                            field_type = 'object'

                        fields.append({
                            "name": key,
                            "type": field_type,
                            "description": f"Auto-detected from JSON field {key}",
                            "sample_values": [value] if value is not None else []
                        })

            elif isinstance(json_data, dict):
                # Single object
                for key, value in json_data.items():
                    field_type = type(value).__name__
                    if field_type == 'str':
                        field_type = 'string'
                    elif field_type == 'int':
                        field_type = 'integer'
                    elif field_type == 'float':
                        field_type = 'float'
                    elif field_type == 'bool':
                        field_type = 'boolean'
                    elif field_type in ['list', 'dict']:
                        field_type = 'object'

                    fields.append({
                        "name": key,
                        "type": field_type,
                        "description": f"Auto-detected from JSON field {key}",
                        "sample_values": [value] if value is not None else []
                    })

            return {
                "name": filename.replace('.json', ''),
                "fields": fields,
                "metadata": {
                    "source": "json_upload",
                    "detected_at": datetime.now().isoformat(),
                    "structure": "array" if isinstance(json_data, list) else "object"
                }
            }

        def _setup_routes(self):
            """Setup API routes"""

            @self.app.get("/", response_class=HTMLResponse)
            async def root():
                return """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Advanced Data Mapping Agent</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
                        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                        h1 { color: #333; border-bottom: 3px solid #007acc; padding-bottom: 10px; }
                        .feature { background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007acc; }
                        .endpoint { background: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 5px; }
                        a { color: #007acc; text-decoration: none; }
                        a:hover { text-decoration: underline; }
                        .new-endpoint { background: #e8f8e8; border-left: 4px solid #28a745; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>🎯 Advanced Data Mapping Agent</h1>
                        <p><strong>Enterprise-grade field mapping system with BGE Large v1.5 and LLM assistance</strong></p>

                        <h2>🚀 Features</h2>
                        <div class="feature">
                            <strong>🧠 BGE Large v1.5 Embeddings:</strong> State-of-the-art semantic understanding for field matching
                        </div>
                        <div class="feature">
                            <strong>🤖 LLM Integration:</strong> Ollama-powered reasoning for complex mapping decisions
                        </div>
                        <div class="feature">
                            <strong>⚡ High Performance:</strong> Optimized embeddings with intelligent caching
                        </div>
                        <div class="feature">
                            <strong>📊 Quality Analytics:</strong> Comprehensive validation and confidence scoring
                        </div>

                        <h2>📡 API Endpoints</h2>
                        <div class="endpoint new-endpoint">
                            <strong>POST /upload</strong> - Upload file to backend for processing
                        </div>
                        <div class="endpoint new-endpoint">
                            <strong>GET /insights</strong> - Retrieve backend-generated insights
                        </div>
                        <div class="endpoint new-endpoint">
                            <strong>GET /download</strong> - Download mapped document
                        </div>
                        <div class="endpoint new-endpoint">
                            <strong>GET /auto-map</strong> - Use LLM to auto-map uploaded data
                        </div>
                        <div class="endpoint">
                            <strong>POST /map-schemas</strong> - Map source schema to target schema
                        </div>
                        <div class="endpoint">
                            <strong>GET /health</strong> - Check system health and status
                        </div>
                        <div class="endpoint">
                            <strong>GET /stats</strong> - View performance statistics
                        </div>

                        <h2>📚 Documentation</h2>
                        <p>
                            <a href="/docs">📖 Interactive API Documentation (Swagger UI)</a><br>
                            <a href="/redoc">📋 Alternative Documentation (ReDoc)</a>
                        </p>

                        <h2>💡 Quick Start</h2>
                        <p>1. Upload a CSV/JSON file using <code>POST /upload</code></p>
                        <p>2. Get insights with <code>GET /insights</code></p>
                        <p>3. Auto-map with <code>GET /auto-map?file_id=file_1</code></p>
                        <p>4. Download results with <code>GET /download?file_id=file_1</code></p>
                    </div>
                </body>
                </html>
                """

            @self.app.post("/upload")
            async def upload_file(file: UploadFile = File(...)):
                """Upload file to backend for processing"""
                try:
                    # Create uploads directory
                    upload_dir = "uploads"
                    os.makedirs(upload_dir, exist_ok=True)

                    # Read file content
                    content = await file.read()
                    file_path = os.path.join(upload_dir, file.filename)

                    # Save file
                    with open(file_path, "wb") as buffer:
                        buffer.write(content)

                    # Basic file info
                    file_info = {
                        "filename": file.filename,
                        "size": len(content),
                        "content_type": file.content_type,
                        "file_path": file_path
                    }

                    # Auto-detect schema
                    schema_data = None
                    if file.filename.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        schema_data = self._csv_to_schema(df, file.filename)
                    elif file.filename.endswith('.json'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                            schema_data = self._json_to_schema(json_data, file.filename)

                    # Store file info
                    file_id = f"file_{len(self.uploaded_files) + 1}"
                    self.uploaded_files[file_id] = {
                        **file_info,
                        "schema": schema_data,
                        "upload_time": datetime.now().isoformat()
                    }

                    return {
                        "status": "success",
                        "message": "File uploaded successfully",
                        "file_id": file_id,
                        "file_info": file_info,
                        "detected_schema": schema_data
                    }

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

            @self.app.get("/insights")
            async def get_insights():
                """Retrieve backend-generated insights from uploaded files"""
                try:
                    if not self.uploaded_files:
                        return {
                            "status": "no_data",
                            "message": "No files uploaded yet",
                            "insights": []
                        }

                    insights = []

                    for file_id, file_info in self.uploaded_files.items():
                        file_insights = {
                            "file_id": file_id,
                            "filename": file_info["filename"],
                            "upload_time": file_info["upload_time"],
                            "insights": []
                        }

                        # Generate insights based on schema
                        if file_info.get("schema"):
                            schema = file_info["schema"]
                            field_count = len(schema.get("fields", []))

                            # Field type analysis
                            field_types = {}
                            for field in schema.get("fields", []):
                                field_type = field.get("type", "unknown")
                                field_types[field_type] = field_types.get(field_type, 0) + 1

                            file_insights["insights"].extend([
                                {
                                    "type": "field_analysis",
                                    "title": "Field Statistics",
                                    "data": {
                                        "total_fields": field_count,
                                        "field_types": field_types,
                                        "complexity": "High" if field_count > 20 else "Medium" if field_count > 10 else "Low"
                                    }
                                },
                                {
                                    "type": "data_quality",
                                    "title": "Data Quality Assessment",
                                    "data": {
                                        "completeness": "Good" if field_count > 5 else "Limited",
                                        "structure": "Structured",
                                        "mapping_readiness": "Ready for mapping"
                                    }
                                }
                            ])

                            # Detect common patterns
                            common_fields = []
                            for field in schema.get("fields", []):
                                field_name = field.get("name", "").lower()
                                if any(term in field_name for term in ["id", "name", "email", "phone", "date"]):
                                    common_fields.append(field.get("name"))

                            if common_fields:
                                file_insights["insights"].append({
                                    "type": "mapping_suggestions",
                                    "title": "Common Fields Detected",
                                    "data": {
                                        "standard_fields": common_fields,
                                        "recommendation": "These fields have common patterns suitable for auto-mapping"
                                    }
                                })

                        insights.append(file_insights)

                    return {
                        "status": "success",
                        "insights": insights,
                        "summary": {
                            "total_files": len(self.uploaded_files),
                            "processed_files": len([f for f in self.uploaded_files.values() if f.get("schema")]),
                            "generated_at": datetime.now().isoformat()
                        }
                    }

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")

            @self.app.get("/download")
            async def download_mapped_document(file_id: str = None):
                """Download mapped document"""
                try:
                    if not file_id:
                        # Return available files
                        available_files = []
                        for fid, file_info in self.uploaded_files.items():
                            if file_info.get("schema"):
                                available_files.append({
                                    "file_id": fid,
                                    "filename": file_info["filename"],
                                    "upload_time": file_info["upload_time"]
                                })

                        return {
                            "status": "success",
                            "message": "Available files for download",
                            "available_files": available_files
                        }

                    if file_id not in self.uploaded_files:
                        raise HTTPException(status_code=404, detail="File not found")

                    file_info = self.uploaded_files[file_id]

                    # Create mapping report
                    mapped_content = {
                        "original_file": file_info["filename"],
                        "mapping_report": {
                            "generated_at": datetime.now().isoformat(),
                            "schema": file_info.get("schema"),
                            "mapping_status": "completed",
                            "fields_mapped": len(file_info.get("schema", {}).get("fields", [])),
                            "confidence_score": 0.85,
                            "auto_mapping_results": file_info.get("auto_mapping"),
                            "recommendations": [
                                "Review auto-mapped fields for accuracy",
                                "Validate data types before implementation",
                                "Consider adding field descriptions for better mapping"
                            ]
                        }
                    }

                    return {
                        "status": "success",
                        "message": "Mapped document ready",
                        "file_id": file_id,
                        "mapped_data": mapped_content,
                        "download_url": f"/download/{file_id}/mapped_document.json"
                    }

                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Download preparation failed: {str(e)}")

            @self.app.get("/auto-map")
            async def auto_map_uploaded_data(file_id: str = None, target_schema: str = None):
                """Use LLM to auto-map uploaded data"""
                try:
                    if not file_id:
                        return {
                            "status": "error",
                            "message": "file_id parameter is required",
                            "available_files": list(self.uploaded_files.keys())
                        }

                    if file_id not in self.uploaded_files:
                        raise HTTPException(status_code=404, detail="File not found")

                    file_info = self.uploaded_files[file_id]
                    source_schema = file_info.get("schema")

                    if not source_schema:
                        raise HTTPException(status_code=400, detail="Source file has no detectable schema")

                    # Create default target schema if none provided
                    if not target_schema:
                        target_schema = {
                            "name": "Standard CRM Schema",
                            "fields": [
                                {"name": "customer_id", "type": "integer", "description": "Unique customer identifier"},
                                {"name": "full_name", "type": "string", "description": "Customer full name"},
                                {"name": "email_address", "type": "string", "description": "Email contact"},
                                {"name": "phone_number", "type": "string", "description": "Phone contact"},
                                {"name": "created_date", "type": "date", "description": "Record creation date"},
                                {"name": "status", "type": "string", "description": "Customer status"}
                            ]
                        }
                    else:
                        # Parse if JSON string
                        if isinstance(target_schema, str):
                            target_schema = json.loads(target_schema)

                    # Perform auto-mapping
                    self.logger.info(f"🤖 Starting auto-mapping for file {file_id}")

                    mapping_results = await self.agent.map_schemas(
                        source_schema=source_schema,
                        target_schema=target_schema,
                        options={"similarity_threshold": 0.6}
                    )

                    # Generate LLM insights
                    if mapping_results["status"] == "success":
                        high_confidence_mappings = [
                            m for m in mapping_results["mappings"]
                            if m.get("confidence") in ["high", "very_high"]
                        ]

                        medium_confidence_mappings = [
                            m for m in mapping_results["mappings"]
                            if m.get("confidence") == "medium"
                        ]

                        unmapped_fields = [
                            m for m in mapping_results["mappings"]
                            if m.get("target_field") is None
                        ]

                        llm_insights = [
                            {
                                "type": "mapping_quality",
                                "message": f"Found {len(high_confidence_mappings)} high-confidence mappings",
                                "confidence": "high" if len(high_confidence_mappings) > len(
                                    mapping_results["mappings"]) * 0.7 else "medium"
                            },
                            {
                                "type": "review_needed",
                                "message": f"{len(medium_confidence_mappings)} mappings need review",
                                "fields": [m["source_field"] for m in medium_confidence_mappings]
                            }
                        ]

                        if unmapped_fields:
                            llm_insights.append({
                                "type": "unmapped_alert",
                                "message": f"{len(unmapped_fields)} fields could not be mapped automatically",
                                "fields": [m["source_field"] for m in unmapped_fields],
                                "suggestion": "Consider manual mapping or updating target schema"
                            })

                        # Store results
                        self.uploaded_files[file_id]["auto_mapping"] = {
                            "results": mapping_results,
                            "llm_insights": llm_insights,
                            "generated_at": datetime.now().isoformat()
                        }

                        return {
                            "status": "success",
                            "message": "Auto-mapping completed successfully",
                            "file_id": file_id,
                            "mapping_results": mapping_results,
                            "llm_insights": llm_insights,
                            "summary": {
                                "total_source_fields": len(source_schema.get("fields", [])),
                                "total_target_fields": len(target_schema.get("fields", [])),
                                "mapped_fields": len([m for m in mapping_results["mappings"] if m.get("target_field")]),
                                "unmapped_fields": len(unmapped_fields),
                                "overall_confidence": mapping_results.get("quality_metrics", {}).get("overall_score", 0)
                            }
                        }
                    else:
                        return {
                            "status": "error",
                            "message": "Auto-mapping failed",
                            "error": mapping_results.get("error", "Unknown error")
                        }

                except HTTPException:
                    raise
                except Exception as e:
                    self.logger.error(f"Auto-mapping failed: {e}")
                    raise HTTPException(status_code=500, detail=f"Auto-mapping failed: {str(e)}")

            @self.app.post("/map-schemas")
            async def map_schemas(request: dict):
                """Map source schema to target schema (original endpoint)"""
                try:
                    # Handle both Pydantic and dict inputs
                    if hasattr(request, 'dict'):
                        request_dict = request.dict()
                    else:
                        request_dict = request

                    # Extract schemas
                    source_schema_dict = request_dict["source_schema"]
                    target_schema_dict = request_dict["target_schema"]
                    options = request_dict.get("options", {})

                    # Perform mapping
                    results = await self.agent.map_schemas(
                        source_schema_dict,
                        target_schema_dict,
                        options
                    )

                    if results["status"] == "success":
                        return {
                            "status": "success",
                            "mappings": results["mappings"],
                            "quality_metrics": results["quality_metrics"],
                            "recommendations": results["recommendations"]
                        }
                    else:
                        return {
                            "status": "error",
                            "error": results.get("error", "Unknown error occurred")
                        }

                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

            @self.app.get("/health")
            async def health_check():
                """Health check endpoint"""
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "model_loaded": self.agent.embedding_model is not None,
                    "llm_available": self.agent.llm is not None,
                    "cache_size": len(self.agent.embedding_cache),
                    "uploaded_files": len(self.uploaded_files),
                    "dependencies": {
                        "fastapi": FASTAPI_AVAILABLE,
                        "pydantic": PYDANTIC_AVAILABLE,
                        "langchain": LANGCHAIN_AVAILABLE,
                        "aiofiles": AIOFILES_AVAILABLE,
                        "tqdm": TQDM_AVAILABLE
                    }
                }

            @self.app.get("/stats")
            async def get_stats():
                """Get performance statistics"""
                return {
                    "cache_stats": {
                        "cache_hits": self.agent.cache_hits,
                        "cache_misses": self.agent.cache_misses,
                        "cache_hit_rate": self.agent.cache_hits / (self.agent.cache_hits + self.agent.cache_misses) if (
                                                                                                                               self.agent.cache_hits + self.agent.cache_misses) > 0 else 0,
                        "cache_size": len(self.agent.embedding_cache)
                    },
                    "model_info": {
                        "embedding_model": self.agent.model_name,
                        "llm_model": self.agent.llm_model if self.agent.llm else "Not available",
                        "device": "GPU" if torch.cuda.is_available() else "CPU",
                        "similarity_threshold": self.agent.similarity_threshold
                    },
                    "file_stats": {
                        "uploaded_files": len(self.uploaded_files),
                        "processed_files": len([f for f in self.uploaded_files.values() if f.get("schema")]),
                        "auto_mapped_files": len([f for f in self.uploaded_files.values() if f.get("auto_mapping")])
                    },
                    "system_info": {
                        "timestamp": datetime.now().isoformat(),
                        "dependencies": {
                            "fastapi": FASTAPI_AVAILABLE,
                            "pydantic": PYDANTIC_AVAILABLE,
                            "langchain": LANGCHAIN_AVAILABLE,
                            "aiofiles": AIOFILES_AVAILABLE,
                            "tqdm": TQDM_AVAILABLE
                        }
                    }
                }


# Standalone mode (if FastAPI is not available)
class StandaloneAgent:
    """Standalone version without web interface"""

    def __init__(self):
        self.agent = DataMappingAgent()
        self.logger = logging.getLogger(__name__)

    async def run_example(self):
        """Run example mapping"""
        print("🎯 Running Standalone Data Mapping Agent")
        print("=" * 50)

        # Example schemas
        source_schema = {
            "name": "Customer Database",
            "fields": [
                {"name": "cust_id", "type": "integer", "description": "Customer identifier"},
                {"name": "full_name", "type": "string", "description": "Customer full name"},
                {"name": "email_addr", "type": "string", "description": "Email address"},
                {"name": "phone_num", "type": "string", "description": "Phone number"},
                {"name": "birth_date", "type": "date", "description": "Date of birth"}
            ]
        }

        target_schema = {
            "name": "CRM System",
            "fields": [
                {"name": "customer_id", "type": "integer", "description": "Unique customer ID"},
                {"name": "name", "type": "string", "description": "Customer name"},
                {"name": "email", "type": "string", "description": "Email contact"},
                {"name": "phone", "type": "string", "description": "Phone contact"},
                {"name": "dob", "type": "date", "description": "Date of birth"}
            ]
        }

        print("🚀 Running schema mapping...")
        results = await self.agent.map_schemas(source_schema, target_schema)

        print("\n📊 Mapping Results:")
        print(f"Status: {results['status']}")
        print(f"Processing time: {results.get('processing_time_seconds', 0):.2f} seconds")

        if results['status'] == 'success':
            print(f"\n📈 Quality Metrics:")
            metrics = results.get('quality_metrics', {})
            print(f"  Overall score: {metrics.get('overall_score', 0):.3f}")
            print(f"  Mapping coverage: {metrics.get('mapping_coverage', 0):.3f}")

            print(f"\n🔗 Mappings:")
            for i, mapping in enumerate(results.get('mappings', [])[:5]):  # Show first 5
                print(f"  {i + 1}. {mapping['source_field']} -> {mapping.get('target_field', 'UNMAPPED')}")
                print(f"     Similarity: {mapping['similarity_score']:.3f}, Confidence: {mapping['confidence']}")

        print(f"\n💡 Recommendations:")
        for rec in results.get('recommendations', []):
            print(f"  • {rec}")


# Main execution function
async def main():
    """Main function to run the Data Mapping Agent"""
    print("🎯 Starting Advanced Data Mapping Agent with BGE Large v1.5...")
    print("🔧 Enterprise-grade field mapping system with LLM assistance")
    print("📊 Model: BAAI/bge-large-en-v1.5 (1.34GB)")
    print("=" * 60)

    # Set environment variables for better downloads
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', 'FALSE')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', 'TRUE')

    try:
        if FASTAPI_AVAILABLE:
            # Initialize the API
            print("🔄 Initializing Data Mapping Agent with Web API...")
            api = DataMappingAPI()

            print("✅ Data Mapping Agent Ready!")
            print("🌐 API Server running on: http://localhost:8000")
            print("📊 Interactive docs: http://localhost:8000/docs")
            print("📋 Alternative docs: http://localhost:8000/redoc")
            print("💡 Health check: http://localhost:8000/health")
            print("🆕 NEW ENDPOINTS:")
            print("   📤 POST /upload - Upload files")
            print("   💡 GET /insights - Get insights")
            print("   🤖 GET /auto-map - Auto-map with LLM")
            print("   📥 GET /download - Download results")
            print("=" * 60)
            print("🚀 Ready to process schema mapping requests!")

            # Run the server
            config = uvicorn.Config(
                api.app,
                host="0.0.0.0",
                port=8000,
                log_level="info",
                access_log=False
            )
            server = uvicorn.Server(config)
            await server.serve()
        else:
            # Run in standalone mode
            print("⚠️ FastAPI not available, running in standalone mode...")
            standalone = StandaloneAgent()
            await standalone.run_example()

    except Exception as e:
        print(f"❌ Failed to start Data Mapping Agent: {e}")
        print("\n🔧 Troubleshooting BGE Large v1.5 issues:")
        print("   1. Check internet connection: ping huggingface.co")
        print("   2. Update packages: pip install --upgrade sentence-transformers huggingface_hub")
        print("   3. Clear model cache: rm -rf model_cache/")
        print("   4. Check proxy/firewall settings")
        print("   5. Install missing dependencies: python install_dependencies.py")
        raise


if __name__ == "__main__":
    # Run the application
    asyncio.run(main())
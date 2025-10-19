# MVP Specification: AI Pipeline for Meal Recognition and Nutritional Estimation

## 1. Objective
Develop a modular AI system that estimates the nutritional composition of a meal from a single image.  
The pipeline performs segmentation, classification, and nutrition retrieval, producing a structured nutritional summary in JSON format.

---

## 2. High-Level Pipeline

```plaintext
[ Input: Meal photo (.jpg/.png) ]
    ↓
[ Segmentation + Volume Estimation ]
    ↓
[ LLM-based Food Classification ]
    ↓
[ Nutrition Retrieval via Embedding Search ]
    ↓
[ Nutrient Aggregation + Final JSON Output ]
```

Each component outputs structured JSON and can be executed independently.  
Pipeline orchestration is handled by **LangGraph** or a similar graph-based workflow manager.

---

## 3. Component Specifications

### 3.1 Segmentation + Volume Estimation

**Input:**
- `meal_image` (path or base64)

**Output JSON Example:**
```json
[
  {
    "mask_id": 1,
    "crop_image_path": "segments/1_chicken.png",
    "volume_ml": 125.4
  },
  {
    "mask_id": 2,
    "crop_image_path": "segments/2_rice.png",
    "volume_ml": 110.2
  }
]
```

**Implementation Plan:**
- Use `Mask2Former` (Detectron2) pretrained on COCO.
- Fine-tune on food datasets: `UEC-FoodPix Complete`, `UNIMIB2016`, `FoodSeg103/154`.
- Optionally add 150–300 Swedish meal samples annotated via **SAM (Segment Anything)**.
- Estimate volume using:
  - Pixel area + monocular depth (`MiDaS` or `DPT`), or  
  - Scale reference (plate diameter / known object).

**Dependencies:**
- PyTorch, Detectron2, OpenCV, MiDaS, NumPy

---

### 3.2 LLM-Based Classification

**Input:**
- Segment crop image  
- Optional text description (string)

**Prompt Template:**
```
You are a food classifier.
Input: [image] + [short description]
Output JSON:
{
  "dish_name": "...",
  "main_ingredients": [...],
  "confidence": 0.0-1.0
}
```

**Output Example:**
```json
{
  "mask_id": 1,
  "dish_name": "chicken with rice and broccoli",
  "main_ingredients": ["chicken", "rice", "broccoli"],
  "confidence": 0.94
}
```

**Implementation Plan:**
- Use GPT-4V (vision + JSON mode) or open-source equivalent (e.g., Llava, Qwen-VL).
- Enforce structured output using JSON schema validation or function-calling constraints.

---

### 3.3 Nutrition Retrieval (Embedding Search)

**Goal:** Map `dish_name` or `main_ingredients` to nutritional database entries.

**Input Example:**
```json
{
  "dish_name": "boiled rice",
  "main_ingredients": ["rice"]
}
```

**Output Example:**
```json
{
  "matched_food": "boiled rice",
  "carbohydrates_per_100g": 28.0,
  "protein_per_100g": 2.4,
  "fat_per_100g": 0.3,
  "match_score": 0.91
}
```

**Implementation Plan:**
- Encode text with `text-embedding-3-small` (OpenAI).
- Perform nearest-neighbour search via FAISS or LanceDB.
- Databases:
  - Swedish Food Agency dataset
  - USDA FoodData Central
  - Open Food Facts

**Dependencies:**
- FAISS, OpenAI API, Pandas, NumPy

---

### 3.4 Nutrient Aggregation

**Input:**
- Segmentation output (volumes)
- Nutrition data (per-100g)
- Density lookup table (`ρ_i`)

**Formula:**
\[
N_k = \sum_i \frac{\rho_i \, V_i}{100} \, C_{i,k}
\]

**Output JSON:**
```json
{
  "meal": [
    {"food": "chicken", "mass_g": 140, "protein_g": 42},
    {"food": "rice", "mass_g": 120, "carbohydrates_g": 34},
    {"food": "broccoli", "mass_g": 70, "carbohydrates_g": 3}
  ],
  "total": {
    "carbohydrates_g": 37,
    "protein_g": 44,
    "fat_g": 6
  },
  "confidence": 0.88
}
```

---

### 3.5 LangGraph Orchestration

**Graph Structure:**
```
Segmenter → VolumeEstimator → LLMClassifier → VectorDBRetriever → Aggregator
```

**Execution Features:**
- Each node reads/writes structured JSON.
- Nodes can be run independently for debugging.
- Logs pipeline trace with timestamps and intermediate results.

**Dependencies:**
- LangGraph, JSONSchema, Python logging

---

## 4. Evaluation Metrics

| Stage | Metric | Description |
|-------|---------|-------------|
| Segmentation | mIoU, mAP | Object-level accuracy |
| Classification | Accuracy, semantic similarity | JSON-based or cosine similarity |
| Volume Estimation | RMSE | Deviation from true volume |
| Nutrition Estimation | MAE | Nutrient error (g or %) |

---

## 5. Risks & Mitigations

| Risk | Mitigation |
|------|-------------|
| Ambiguous dishes | Ingredient-level reasoning + user confirmation |
| Scale uncertainty | Fiducial marker or monocular depth model |
| LLM inconsistency | Function-calling schema enforcement |
| Database mismatch | Vector similarity fallback |

---

## 6. Deliverables (MVP Scope)

| File | Purpose |
|------|----------|
| `segmenter.py` | Food segmentation & volume estimation |
| `classifier.py` | LLM-based food classification |
| `nutrition_search.py` | Embedding-based database retrieval |
| `aggregator.py` | Nutrient computation and summary |
| `graph.py` | LangGraph pipeline orchestration |
| `demo_pipeline.ipynb` | End-to-end demo |
| `outputs/example_output.json` | Sample result for validation |

---

## 7. Example Full Pipeline Output

```json
{
  "input_image": "meal_001.jpg",
  "segments": [
    {
      "mask_id": 1,
      "dish_name": "grilled chicken",
      "volume_ml": 150.0,
      "nutrition": {"protein_g": 40, "fat_g": 6},
      "confidence": 0.9
    },
    {
      "mask_id": 2,
      "dish_name": "boiled rice",
      "volume_ml": 120.0,
      "nutrition": {"carbohydrates_g": 35, "protein_g": 3},
      "confidence": 0.87
    }
  ],
  "total": {
    "carbohydrates_g": 35,
    "protein_g": 43,
    "fat_g": 6
  },
  "overall_confidence": 0.88
}
```

---

## 8. Implementation Order

1. Build segmentation + volume estimation module  
2. Implement LLM-based classification (JSON schema enforced)  
3. Integrate embedding-based nutrition search  
4. Implement nutrient aggregation logic  
5. Connect modules with LangGraph  
6. Add CLI or Jupyter interface for testing  

---

## 9. File Structure

MVP-diabetets-project/
│
├── src/
│   ├── __init__.py
│   ├── segmenter.py             # Step 1: segmentation + volume estimation
│   ├── classifier.py            # Step 2: LLM-based food classification
│   ├── nutrition_search.py      # Step 3: embedding + vector DB retrieval
│   ├── aggregator.py            # Step 4: nutrient aggregation logic
│   ├── graph.py                 # LangGraph orchestration pipeline
│   └── utils.py                 # Shared functions (I/O, JSON schema, etc.)
│
├── data/
│   ├── raw/                     # Sample meal images
│   ├── processed/               # Segments, depth maps, etc.
│   ├── embeddings/              # Vector DB or precomputed embeddings
│   └── db/                      # Nutrition CSVs (Swedish Food Agency, USDA, etc.)
│
├── notebooks/
│   ├── demo_pipeline.ipynb      # Run pipeline end-to-end in notebook
│   └── evaluation.ipynb         # Evaluation + metrics testing
│
├── config/
│   ├── settings.yaml            # Global paths, model parameters, API keys
│   ├── prompt_templates/        # JSON or YAML LLM prompt schemas
│   └── densities.yaml           # Food densities (for nutrient aggregation)
│
├── outputs/
│   ├── logs/                    # Log files from runs
│   ├── examples/                # JSON example outputs
│   └── results.csv              # Aggregate evaluation results
│
├── tests/
│   ├── test_segmenter.py
│   ├── test_classifier.py
│   └── test_aggregator.py
│
├── requirements.txt
├── README.md
└── mvp_spec.md

---

## 10. Future Extensions

- Depth-based 3D volume estimation  
- Personalized nutrition tracking  
- Glucose-level prediction  
- Dietary recommendations based on history  

---

**Author:** Fredrick Carlsåker  
**Version:** 1.0  
**Date:** 2025-10-19  


# Implementation Guide

This guide explains what has been built and what needs to be implemented to complete the MVP.

## ✅ What's Been Built

### 1. Complete Project Structure
- All directories and placeholder files created
- Organized modular architecture
- Clear separation of concerns

### 2. Configuration System
- `config/settings.yaml` - Main configuration
- `config/densities.yaml` - Food density lookup
- `config/prompt_templates/` - LLM prompts
- Environment variable support via `.env`

### 3. Utility Framework
- `src/utils.py` - Complete utility library with:
  - Config management
  - Logging setup
  - Image I/O functions
  - JSON validation
  - Density lookup
  - Timing utilities
  - Schema definitions

### 4. Model Wrapper Interfaces
All model wrappers are structured with clear interfaces:

- **SegmentationModel** (`src/models/segmentation_model.py`)
  - Methods: `load_model()`, `segment()`, `extract_crops()`, `visualize_segments()`
  
- **DepthEstimationModel** (`src/models/depth_model.py`)
  - Methods: `load_model()`, `estimate_depth()`, `normalize_depth()`, `visualize_depth()`
  
- **FoodClassificationModel** (`src/models/classification_model.py`)
  - Methods: `load_model()`, `classify()`, `classify_batch()`
  
- **EmbeddingModel** (`src/models/embedding_model.py`)
  - Methods: `load_model()`, `embed()`, `embed_batch()`, `find_most_similar()`

### 5. Pipeline Components
All components are structured and ready for model integration:

- **FoodSegmenter** (`src/segmenter.py`)
  - Volume estimation logic implemented (3 methods)
  - Crop extraction ready
  - Visualization support

- **FoodClassifier** (`src/classifier.py`)
  - Prompt template system
  - Batch classification support
  - Result validation

- **NutritionSearch** (`src/nutrition_search.py`)
  - Vector search implementation
  - Exact string matching fallback
  - Database loading and caching
  - Embedding generation system

- **NutrientAggregator** (`src/aggregator.py`)
  - Complete aggregation logic
  - Macronutrient ratio calculation
  - Glycemic load estimation
  - CSV export functionality

### 6. Pipeline Orchestration
- **NutritionPipeline** (`src/graph.py`)
  - Complete orchestration framework
  - Node-based execution
  - State management
  - Error handling
  - Individual node execution support

### 7. CLI Interface
- **main.py** - Full command-line interface with:
  - Complete pipeline execution
  - Individual node execution
  - State saving/loading
  - Verbose logging option

### 8. Testing Framework
- Test structure in `tests/`
- Example tests in `test_segmenter.py`
- Pytest configuration ready

### 9. Documentation
- Comprehensive README.md
- Inline code documentation
- Jupyter notebook template
- This implementation guide

---

## ⏳ What Needs to Be Implemented

### Phase 1: Core Models (Critical Path)

#### 1.1 Segmentation Model
**File**: `src/models/segmentation_model.py`

```python
def load_model(self):
    """Implement Detectron2 Mask2Former loading"""
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask2former_R50_bs16_50ep.yaml"
    ))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask2former_R50_bs16_50ep.yaml"
    )
    cfg.MODEL.DEVICE = self.device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
    
    self.predictor = DefaultPredictor(cfg)

def segment(self, image):
    """Implement actual segmentation"""
    outputs = self.predictor(image)
    instances = outputs["instances"].to("cpu")
    
    results = []
    for i in range(len(instances)):
        results.append({
            "mask": instances.pred_masks[i].numpy(),
            "bbox": instances.pred_boxes[i].tensor.numpy()[0],
            "score": float(instances.scores[i]),
            "class_id": int(instances.pred_classes[i]),
            "class_name": self._get_class_name(int(instances.pred_classes[i]))
        })
    
    return results
```

**Estimated Time**: 2-3 hours
**Dependencies**: Detectron2, CUDA (optional)

#### 1.2 Depth Estimation Model
**File**: `src/models/depth_model.py`

```python
def load_model(self):
    """Implement MiDaS loading"""
    import torch
    
    self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
    self.model.to(self.device)
    self.model.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if self.model_type == "DPT_Large":
        self.transform = midas_transforms.dpt_transform
    else:
        self.transform = midas_transforms.small_transform

def estimate_depth(self, image):
    """Implement depth estimation"""
    import torch
    
    input_batch = self.transform(image).to(self.device)
    
    with torch.no_grad():
        prediction = self.model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    return prediction.cpu().numpy()
```

**Estimated Time**: 1-2 hours
**Dependencies**: torch, torchvision

#### 1.3 Classification Model
**File**: `src/models/classification_model.py`

```python
def load_model(self):
    """Implement OpenAI client"""
    from openai import OpenAI
    
    api_key = self.config.get("api_keys.openai")
    self.client = OpenAI(api_key=api_key)

def classify(self, image, additional_context=None):
    """Implement GPT-4V classification"""
    from ..utils import image_to_base64
    
    image_base64 = image_to_base64(image)
    
    prompt = self.prompt_template
    if additional_context:
        prompt += f"\n\nAdditional context: {additional_context}"
    
    response = self.client.chat.completions.create(
        model=self.model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=self.max_tokens,
        temperature=self.temperature,
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    return self._validate_classification_result(result)
```

**Estimated Time**: 1-2 hours
**Dependencies**: openai>=1.0.0
**Cost**: ~$0.01-0.05 per image (GPT-4V API)

#### 1.4 Embedding Model
**File**: `src/models/embedding_model.py`

```python
def load_model(self):
    """Implement OpenAI embeddings client"""
    from openai import OpenAI
    
    api_key = self.config.get("api_keys.openai")
    self.client = OpenAI(api_key=api_key)

def embed(self, text):
    """Implement embedding generation"""
    is_single = isinstance(text, str)
    texts = [text] if is_single else text
    
    response = self.client.embeddings.create(
        model=self.model_name,
        input=texts,
        dimensions=self.dimensions
    )
    
    embeddings = np.array([item.embedding for item in response.data])
    return embeddings[0] if is_single else embeddings
```

**Estimated Time**: 1 hour
**Dependencies**: openai>=1.0.0
**Cost**: ~$0.0001 per 1000 tokens

### Phase 2: Data Preparation

#### 2.1 Nutrition Databases
**Location**: `data/db/`

Download and prepare:

1. **Swedish Food Agency Database**
   - Source: [Livsmedelsverket](https://www.livsmedelsverket.se/)
   - Convert to CSV format

2. **USDA FoodData Central**
   - Source: [FoodData Central](https://fdc.nal.usda.gov/)
   - Download: [Foundation Foods](https://fdc.nal.usda.gov/download-datasets.html)

3. **Format Requirements**:
```csv
food_name,carbohydrates_per_100g,protein_per_100g,fat_per_100g,energy_kcal_per_100g
"Boiled White Rice",28.0,2.7,0.3,130
"Grilled Chicken Breast",0.0,31.0,3.6,165
```

**Estimated Time**: 4-6 hours
**Tools**: Python, Pandas for data processing

#### 2.2 Generate Embeddings
**Script**: Create `scripts/generate_embeddings.py`

```python
from src.models.embedding_model import EmbeddingModel
from src.utils import Config
import pandas as pd
import numpy as np

config = Config()
embedding_model = EmbeddingModel(config)

# Load nutrition database
db = pd.read_csv("data/db/combined_nutrition_db.csv")

# Generate embeddings
food_names = db["food_name"].tolist()
embeddings = embedding_model.embed_batch(food_names, show_progress=True)

# Save
np.save("data/embeddings/nutrition_db_embeddings.npy", embeddings)
```

**Estimated Time**: 2 hours (including API time)
**Cost**: ~$0.10-0.50 depending on database size

### Phase 3: Testing & Validation

#### 3.1 Unit Tests
Complete test implementations in `tests/`:
- `test_segmenter.py` - Test segmentation logic
- `test_classifier.py` - Test classification
- `test_nutrition_search.py` - Test database search
- `test_aggregator.py` - Test nutrient calculations

**Estimated Time**: 4-6 hours

#### 3.2 Integration Tests
Test complete pipeline with sample images

**Estimated Time**: 2-3 hours

### Phase 4: Fine-Tuning (Optional but Recommended)

#### 4.1 Collect Swedish Food Dataset
- Capture 150-300 Swedish meal images
- Annotate with Segment Anything Model (SAM)
- Create training dataset

**Estimated Time**: 20-30 hours

#### 4.2 Fine-Tune Mask2Former
Fine-tune segmentation on food-specific dataset

**Estimated Time**: 10-15 hours (+ training time)
**Requirements**: GPU with 16GB+ VRAM

---

## 📋 Implementation Checklist

### Week 1: Core Models
- [ ] Implement SegmentationModel
- [ ] Implement DepthEstimationModel  
- [ ] Implement FoodClassificationModel
- [ ] Implement EmbeddingModel
- [ ] Test each model independently

### Week 2: Data & Integration
- [ ] Download and process nutrition databases
- [ ] Generate embeddings for nutrition DB
- [ ] Test end-to-end pipeline with sample images
- [ ] Fix bugs and edge cases

### Week 3: Testing & Validation
- [ ] Complete unit tests for all modules
- [ ] Integration tests with real meal images
- [ ] Validate accuracy on test set
- [ ] Performance optimization
- [ ] Documentation updates

### Week 4+: Optional Enhancements
- [ ] Collect Swedish food dataset
- [ ] Fine-tune models on food data
- [ ] Improve volume estimation with calibration
- [ ] Add user feedback mechanism
- [ ] Deploy as web service

---

## 🎯 Quick Start Implementation

### Minimal Working Version (1-2 days)

To get a working pipeline ASAP, implement in this order:

1. **Classification Model Only** (2 hours)
   - Skip segmentation, use whole image
   - Implement GPT-4V classification
   - Manually provide volume estimates

2. **Nutrition Search** (1 hour)
   - Use exact string matching (skip embeddings)
   - Start with small manual nutrition database

3. **Basic Aggregation** (30 mins)
   - Calculate totals from manual inputs
   - Generate simple JSON output

4. **Test** (1 hour)
   - Run on 3-5 sample images
   - Verify output format
   - Document limitations

### Code Example for Minimal MVP:

```python
# minimal_mvp.py
from src.models.classification_model import FoodClassificationModel
from src.aggregator import NutrientAggregator
from src.utils import Config, load_image
import json

config = Config()
classifier = FoodClassificationModel(config)
aggregator = NutrientAggregator(config)

# Classify entire image
image = load_image("meal.jpg")
classification = classifier.classify(image)

# Manual nutrition lookup (simplified)
nutrition_db = {
    "chicken": {"carbs": 0, "protein": 31, "fat": 3.6},
    "rice": {"carbs": 28, "protein": 2.7, "fat": 0.3},
    # ... add more
}

# Manual volume estimate
volume_ml = 300  # You estimate this

# Calculate nutrients
dish = classification["dish_name"]
for food, nutrients in nutrition_db.items():
    if food in dish.lower():
        # Rough calculation
        mass_g = volume_ml * 0.8  # assume density
        print(f"Estimated {food}:")
        print(f"  Carbs: {nutrients['carbs'] * mass_g / 100:.1f}g")
        print(f"  Protein: {nutrients['protein'] * mass_g / 100:.1f}g")
        print(f"  Fat: {nutrients['fat'] * mass_g / 100:.1f}g")
```

---

## 💡 Implementation Tips

### 1. Start Simple, Then Iterate
- Get basic functionality working first
- Add complexity incrementally
- Test each component thoroughly

### 2. Use Pre-trained Models Initially
- Don't fine-tune until you validate the approach
- Pre-trained COCO models work reasonably well for food
- OpenAI models provide good baseline accuracy

### 3. Handle Errors Gracefully
```python
try:
    results = segmenter.process(image_path)
except Exception as e:
    logger.error(f"Segmentation failed: {e}")
    # Fallback to manual segmentation or skip
    results = []
```

### 4. Cache Expensive Operations
```python
# Cache embeddings
embeddings_file = "data/embeddings/cache.npy"
if Path(embeddings_file).exists():
    embeddings = np.load(embeddings_file)
else:
    embeddings = model.embed_batch(texts)
    np.save(embeddings_file, embeddings)
```

### 5. Log Everything
```python
from loguru import logger

logger.info(f"Processing image: {image_path}")
logger.debug(f"Segmentation found {len(segments)} segments")
logger.warning(f"Low confidence: {confidence:.2f}")
logger.error(f"Failed to classify segment {id}")
```

---

## 🔍 Testing Strategy

### Unit Tests
Test individual functions with known inputs:

```python
def test_volume_estimation():
    """Test volume calculation"""
    mask = np.zeros((100, 100), dtype=bool)
    mask[25:75, 25:75] = True  # 50x50 square
    
    volume = segmenter._estimate_volume_area_based(mask)
    assert volume > 0
    assert volume < 1000  # Sanity check
```

### Integration Tests
Test complete pipeline with sample data:

```python
def test_full_pipeline():
    """Test end-to-end pipeline"""
    pipeline = create_pipeline()
    results = pipeline.run("test_images/meal.jpg")
    
    assert "total" in results
    assert results["total"]["carbohydrates_g"] >= 0
    assert results["overall_confidence"] > 0
```

### Visual Validation
Always visually inspect outputs:
- Segmentation masks overlaid on images
- Depth maps visualization
- Compare final nutrition to expected values

---

## 📊 Expected Accuracy

Based on literature and similar systems:

| Component | Expected Accuracy |
|-----------|------------------|
| Segmentation (mIoU) | 60-75% (pre-trained), 80-90% (fine-tuned) |
| Classification | 70-85% (GPT-4V), 85-95% (fine-tuned) |
| Volume Estimation (RMSE) | ±20-30% (depth-based), ±30-50% (plate-reference) |
| Nutrition Matching | 80-90% (with good DB) |
| Overall System | 50-70% within ±20% of true values |

### Factors Affecting Accuracy:
- Image quality and lighting
- Food presentation and overlap
- Database coverage
- Portion size estimation
- Mixed/complex dishes

---

## 🐛 Common Issues & Solutions

### Issue 1: Detectron2 Installation Fails
**Solution**: Use pre-built wheels or Docker
```bash
# Try pre-built wheel
pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

### Issue 2: CUDA Out of Memory
**Solution**: Reduce batch size or use CPU
```yaml
# config/settings.yaml
models:
  segmentation:
    device: "cpu"  # or reduce image size
```

### Issue 3: OpenAI Rate Limits
**Solution**: Add retry logic with exponential backoff
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def classify_with_retry(image):
    return classifier.classify(image)
```

### Issue 4: Poor Segmentation on Food
**Solution**: 
- Adjust confidence threshold
- Use SAM (Segment Anything) for better food segmentation
- Consider fine-tuning on food dataset

### Issue 5: Nutrition Database Mismatches
**Solution**:
- Improve embedding search with synonyms
- Add manual mapping for common foods
- Allow user confirmation/correction

---

## 📈 Performance Optimization

### Speed Improvements:

1. **Batch Processing**
```python
# Process multiple images at once
images = [load_image(p) for p in image_paths]
results = classifier.classify_batch(images)
```

2. **Async API Calls**
```python
import asyncio

async def classify_async(images):
    tasks = [classifier.classify(img) for img in images]
    return await asyncio.gather(*tasks)
```

3. **Model Quantization**
```python
# Quantize models for faster inference
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

4. **GPU Optimization**
```python
# Use mixed precision
with torch.cuda.amp.autocast():
    output = model(input)
```

---

## 🚀 Deployment Considerations

### For Production Use:

1. **API Service**
   - FastAPI for REST endpoints
   - Docker containerization
   - Load balancing for scale

2. **Caching Strategy**
   - Redis for session data
   - S3 for images
   - Pre-computed embeddings

3. **Monitoring**
   - Prometheus metrics
   - Error tracking (Sentry)
   - Performance logging

4. **User Feedback Loop**
   - Allow corrections
   - Collect ground truth
   - Continuous improvement

---

## 📚 Resources

### Documentation
- [Detectron2 Docs](https://detectron2.readthedocs.io/)
- [MiDaS GitHub](https://github.com/isl-org/MiDaS)
- [OpenAI Vision API](https://platform.openai.com/docs/guides/vision)
- [LangGraph Docs](https://python.langchain.com/docs/langgraph)

### Datasets
- [UEC-FoodPix Complete](http://foodcam.mobi/dataset.html)
- [UNIMIB2016](http://www.ivl.disco.unimib.it/activities/food-recognition/)
- [FoodSeg103](https://github.com/LARC-CMU-SMU/FoodSeg103)
- [USDA FoodData Central](https://fdc.nal.usda.gov/)

### Papers
- "Food Recognition and Volume Estimation" (various)
- "Mask2Former for Instance Segmentation"
- "Monocular Depth Estimation for Nutrition Analysis"

---

## 📞 Getting Help

If you encounter issues:

1. Check the logs in `outputs/logs/`
2. Run tests: `pytest tests/ -v`
3. Verify configuration in `config/settings.yaml`
4. Review error messages carefully
5. Check API quotas and rate limits

---

## 🎓 Learning Path

To fully understand and extend this system:

1. **Week 1-2**: Computer Vision Basics
   - Instance segmentation
   - Depth estimation
   - Image preprocessing

2. **Week 3-4**: Machine Learning
   - Transfer learning
   - Fine-tuning strategies
   - Model evaluation

3. **Week 5-6**: Production ML
   - MLOps practices
   - Model monitoring
   - A/B testing

4. **Ongoing**: Domain Knowledge
   - Nutrition science
   - Food databases
   - Swedish dietary patterns

---

## ✅ Success Criteria

Your MVP is successful when:

- [ ] Pipeline runs end-to-end without errors
- [ ] Produces valid JSON output
- [ ] Estimates within ±30% of true values (on average)
- [ ] Processes image in < 60 seconds
- [ ] Handles 10+ different meal types
- [ ] Confidence scores correlate with accuracy
- [ ] Documentation is complete
- [ ] Tests pass consistently

---

**Good luck with your implementation! 🚀**

For questions or issues, refer to the README.md and inline code documentation.

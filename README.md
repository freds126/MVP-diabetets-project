# MVP Diabetes Project - Nutrition Pipeline

AI-powered pipeline for meal recognition and nutritional estimation from images.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd MVP-diabetes-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env
```

### 3. Run the Pipeline

```bash
# Analyze a meal image
python main.py path/to/meal_image.jpg

# With custom output directory
python main.py path/to/meal_image.jpg -o results/my_meal

# Verbose mode
python main.py path/to/meal_image.jpg -v
```

## 📁 Project Structure

```
MVP-diabetes-project/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── segmenter.py             # Segmentation & volume estimation
│   ├── classifier.py            # LLM-based classification
│   ├── nutrition_search.py      # Embedding-based DB search
│   ├── aggregator.py            # Nutrient aggregation
│   ├── graph.py                 # Pipeline orchestration
│   ├── utils.py                 # Utility functions
│   └── models/                  # Model wrappers
│       ├── segmentation_model.py
│       ├── depth_model.py
│       ├── classification_model.py
│       └── embedding_model.py
├── data/
│   ├── raw/                     # Input images
│   ├── processed/               # Processed outputs
│   ├── embeddings/              # Cached embeddings
│   └── db/                      # Nutrition databases (CSV)
├── config/
│   ├── settings.yaml            # Main configuration
│   ├── densities.yaml           # Food density lookup
│   └── prompt_templates/        # LLM prompts
├── outputs/                     # Pipeline outputs
├── tests/                       # Unit tests
├── notebooks/                   # Jupyter notebooks
├── main.py                      # CLI interface
└── requirements.txt
```

## 🔧 Configuration

Edit `config/settings.yaml` to customize:

- Model selection (OpenAI vs local models)
- Volume estimation method
- Nutrition database sources
- Logging settings
- Pipeline parameters

## 🎯 Usage Examples

### Basic Usage

```python
from src import create_pipeline

# Create pipeline
pipeline = create_pipeline()

# Analyze meal
results = pipeline.run("meal_photo.jpg")

# Access results
print(f"Total calories: {results['total']['energy_kcal']:.0f} kcal")
print(f"Protein: {results['total']['protein_g']:.1f}g")
```

### Running Individual Nodes

```bash
# Run only segmentation
python main.py meal.jpg --node segment

# Resume from previous state
python main.py meal.jpg --node classify --state outputs/run_xxx/pipeline_state.json
```

### Using in Notebooks

```python
from src import SimplePipeline

pipeline = SimplePipeline()
results = pipeline.analyze_meal("meal.jpg")
```

## 📊 Output Format

The pipeline produces a JSON file with:

```json
{
  "input_image": "meal.jpg",
  "timestamp": "2025-10-19T10:30:00",
  "segments": [
    {
      "mask_id": 0,
      "dish_name": "grilled chicken breast",
      "volume_ml": 150.0,
      "mass_g": 142.5,
      "nutrition": {
        "carbohydrates_g": 0.0,
        "protein_g": 42.0,
        "fat_g": 3.5,
        "energy_kcal": 195.0
      },
      "confidence": 0.92
    }
  ],
  "total": {
    "carbohydrates_g": 45.2,
    "protein_g": 58.3,
    "fat_g": 12.1,
    "energy_kcal": 520.0,
    "total_mass_g": 420.5
  },
  "overall_confidence": 0.88,
  "macronutrient_ratios": {
    "carbohydrates_percent": 34.8,
    "protein_percent": 44.9,
    "fat_percent": 20.3
  },
  "glycemic_estimate": {
    "total_carbohydrates_g": 45.2,
    "estimated_glycemic_load": 27.1,
    "glycemic_impact": "medium"
  }
}
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_segmenter.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📝 Adding Nutrition Databases

The pipeline expects CSV files in `data/db/` with this format:

```csv
food_name,carbohydrates_per_100g,protein_per_100g,fat_per_100g,energy_kcal_per_100g
"Boiled Rice",28.0,2.7,0.3,130
"Grilled Chicken Breast",0.0,31.0,3.6,165
"Steamed Broccoli",7.0,2.8,0.4,35
```

Configure database sources in `config/settings.yaml`:

```yaml
nutrition_db:
  sources:
    - name: "swedish_food_agency"
      path: "data/db/swedish_food_db.csv"
      priority: 1
    - name: "usda"
      path: "data/db/usda_food_db.csv"
      priority: 2
```

## 🔌 Model Implementation Status

### ✅ Completed
- Project structure and scaffolding
- Configuration management
- Pipeline orchestration framework
- Utility functions
- Model wrappers (interfaces)
- CLI interface
- Test structure

### ⏳ To Be Implemented
- [ ] Segmentation model (Mask2Former)
- [ ] Depth estimation (MiDaS)
- [ ] Classification model (GPT-4V or Llava)
- [ ] Embedding model integration
- [ ] Vector database setup (FAISS/LanceDB)
- [ ] Fine-tuning on food datasets

## 🎓 Next Steps

### 1. Implement Segmentation Model

```python
# In src/models/segmentation_model.py
def load_model(self):
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    
    cfg = get_cfg()
    cfg.merge_from_file(self.config.get("models.segmentation.config"))
    cfg.MODEL.WEIGHTS = self.config.get("models.segmentation.weights")
    cfg.MODEL.DEVICE = self.device
    
    self.predictor = DefaultPredictor(cfg)
```

### 2. Implement Classification Model

```python
# In src/models/classification_model.py
def load_model(self):
    from openai import OpenAI
    
    api_key = self.config.get("api_keys.openai")
    self.client = OpenAI(api_key=api_key)
```

### 3. Setup Nutrition Databases

- Download Swedish Food Agency database
- Download USDA FoodData Central
- Process and convert to standard CSV format
- Generate embeddings for vector search

### 4. Fine-tune Models (Optional)

- Collect Swedish meal images
- Annotate with SAM (Segment Anything)
- Fine-tune Mask2Former on food data

## 📚 Documentation

- **MVP Specification**: See `mvp_spec.md` for detailed requirements
- **API Documentation**: Generate with `pdoc src/`
- **Configuration Guide**: See comments in `config/settings.yaml`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Fredrick Carlsåker**

## 🙏 Acknowledgments

- Detectron2 for segmentation models
- OpenAI for GPT-4V and embeddings
- Intel for MiDaS depth estimation
- Swedish Food Agency for nutrition data

---

**Status**: 🏗️ Under Development - Model implementations pending

For questions or issues, please open an issue on GitHub.

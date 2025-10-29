# Complete Workflow Guide

This guide walks you through the entire process of training, evaluating, and running evolutionary simulations with The Selfish LoRA.

## 📋 Overview

The complete workflow consists of 5 main steps:

1. **Create Dataset** - Generate training/validation/test data
2. **Fine-Tune Model** - Run hyperparameter search to find best models
3. **Evaluate Models** - Test all models and calculate evolution scores
4. **Prepare Parents** - Select best models for multi-parent initialization
5. **Run Simulation** - Execute evolutionary simulation with warm-start

---

## 🔄 Step-by-Step Process

### **Step 1: Create Dataset**

Generate training data for fine-tuning. The default creates a small, perception-only dataset for warm-start.

```bash
cd fine_tuning
python create_dataset.py
```

**Options:**
- `--include-history` - Add action history to prompts (creates larger dataset)
- `--train-size 100` - Set training set size
- `--val-size 15` - Set validation set size
- `--test-size 25` - Set test set size

**Output:**
- `data/train_data.json` - Training examples
- `data/validation_data.json` - Validation examples
- `data/test_data.json` - Test examples

**Recommended:** Use default (no history) for warm-start approach.

---

### **Step 2: Fine-Tune Model**

Run hyperparameter search to find multiple good models with diverse behaviors.

```bash
python fine_tune_model.py --hyper --trials 10
```

**Parameters:**
- `--hyper` - Enable hyperparameter search
- `--trials 10` - Number of trials to run (10-20 recommended)

**What happens:**
- Optuna searches learning rate, weight decay, batch size, warmup ratio
- Each trial trains for 3 epochs with early stopping
- Only best checkpoint per trial is saved (saves disk space)
- Trials with poor performance are pruned early

**Output:**
```
output/
├── TRAINING_REPORT_FINAL.txt    # All runs compared
├── run-0/
│   └── checkpoint-X/
│       ├── adapter_model.bin
│       ├── adapter_config.json
│       ├── training_report_run-0.txt
│       └── training_plots.png
├── run-1/
└── ...
```

**Time:** ~5-10 minutes per trial (depends on hardware)

---

### **Step 3: Evaluate Models**

Test all fine-tuned models on the test set and calculate evolution scores.

```bash
python evaluate_model.py
```

**What it does:**
- Loads base model and all fine-tuned runs
- Evaluates on test set (unseen data)
- Calculates evolution metrics:
  - **Valid Rate** - Percentage of valid actions
  - **Diversity** - Number of unique actions used
  - **Balance** - How evenly actions are distributed
  - **Evolution Score** - Composite metric (50% valid + 30% diversity + 20% balance)

**Output:**
```
output/EVALUATION_REPORT_FINAL.txt
```

**Example report:**
```
Fine-Tuned Runs Performance (sorted by Evolution Score):
======================================================================
Run          Evol     Valid    Div      Bal      Acc     
             Score    %        %        %        %       
----------------------------------------------------------------------
run-7        88.4     100.0    80.0     71.8     16.0    
run-0        85.7     100.0    80.0     58.3     12.0    
run-1        80.7     100.0    60.0     63.3     12.0    
```

**Key Insight:** Higher evolution score = better for warm-start (not necessarily higher accuracy!)

---

### **Step 4: Prepare Parents**

Select the best models for multi-parent initialization to maximize genetic diversity.

```bash
python prepare_parents.py --top 3
```

**Options:**
- `--top 3` - Select top 3 models by evolution score
- `--runs run-0 run-3 run-7` - Manually specify runs

**What it does:**
- Reads `EVALUATION_REPORT_FINAL.txt`
- Selects best models by evolution score
- Copies them to `../src/weights/`

**Output:**
```
src/weights/
├── run-0/
│   ├── adapter_model.bin
│   └── adapter_config.json
├── run-3/
└── run-7/
```

**Recommended:** Use 3-5 parent models for best balance of diversity and convergence.

---

### **Step 5: Run Simulation**

Execute evolutionary simulation with multi-parent warm-start.

```bash
cd ..
python main.py --population 20 --generations 20
```

**Key Parameters:**
- `--population 20` - Population size (20-50 recommended)
- `--generations 20` - Number of generations (20-50 recommended)
- `--initial-noise 0.01` - Noise added to parent weights (1%)
- `--sparse-mutation-rate 0.2` - Fraction of weights mutated (20%)
- `--weight-regularization 0.0005` - Pull toward initial weights (0.05%)
- `--fixed-environment` - Use same environment each generation (for debugging)

**What happens:**
```
Found 3 fine-tuned runs. Loading multiple parent models...
  Loaded run-0: 56 weight tensors
  Loaded run-3: 56 weight tensors
  Loaded run-7: 56 weight tensors

Initialized 20 agents from 3 parent models + 1.0% noise
  run-0: 7 agents
  run-3: 7 agents
  run-7: 6 agents

Generation 1/20: Avg Fitness = 15.3, Max = 42
Generation 2/20: Avg Fitness = 18.7, Max = 51
...
```

**Output:**
```
src/output/
├── performance_metrics.json
├── genetic_diversity.json
├── behavioral_metrics.json
├── champion_weights/
│   └── generation_20/
└── visualizations/
    ├── fitness_over_time.png
    ├── genetic_diversity.png
    └── champion_path_gen_20.png
```

**Time:** ~30-60 minutes for 20 generations with 20 agents

---

## 🎯 Quick Start (Full Pipeline)

Run the entire pipeline in one go:

```bash
# 1. Create dataset
cd fine_tuning
python create_dataset.py

# 2. Fine-tune (10 trials, ~1 hour)
python fine_tune_model.py --hyper --trials 10

# 3. Evaluate
python evaluate_model.py

# 4. Prepare top 3 parents
python prepare_parents.py --top 3

# 5. Run simulation
cd ..
python main.py --population 20 --generations 20
```

---

## 📊 Understanding the Metrics

### **Training Metrics (Validation Set)**
- **Eval Loss** - Lower is better (but not the only metric!)
- **Valid Action Rate** - Percentage of valid outputs
- **Correct Action Rate** - Percentage matching expected action
- **Action Diversity** - Number of unique actions (0-5)

### **Evaluation Metrics (Test Set)**
- **Evolution Score** - **PRIMARY METRIC** for warm-start quality
- **Valid Rate** - Must be high (ideally 100%)
- **Diversity** - Higher is better (80-100% ideal)
- **Balance** - Higher is better (even distribution)
- **Accuracy** - Less important for evolution

### **Simulation Metrics**
- **Fitness** - Resources gathered per generation
- **Genetic Diversity** - Distance between agent weights
- **Behavioral Diversity** - Variety of actions taken

---

## 🔧 Troubleshooting

### **Problem: Low diversity in fine-tuned models**

**Symptoms:** All models predict same 1-2 actions

**Solutions:**
- Reduce label smoothing (currently 0.05)
- Increase learning rate range
- Add more diverse training examples
- Use `--include-history` flag in dataset creation

### **Problem: Poor evolution performance**

**Symptoms:** Fitness doesn't improve over generations

**Solutions:**
- Increase mutation rate: `--sparse-mutation-rate 0.3`
- Reduce regularization: `--weight-regularization 0.0001`
- Use more parent models: `--top 5`
- Increase population size: `--population 50`

### **Problem: Overfitting during fine-tuning**

**Symptoms:** Training accuracy high, test accuracy low

**Solutions:**
- Reduce epochs (currently 3)
- Increase weight decay
- Use smaller learning rate
- Add more training data

---

## 📈 Optimization Tips

### **For Better Fine-Tuning:**
1. Run more trials (20-30) to find diverse models
2. Check `TRAINING_REPORT_FINAL.txt` for diversity metrics
3. Select models with high diversity, not just low loss

### **For Better Evolution:**
1. Use 3-5 parent models (sweet spot for diversity)
2. Start with higher mutation rate (0.2-0.3)
3. Monitor genetic diversity - should stay high
4. Use fixed environment for debugging, varied for final runs

### **For Faster Iteration:**
1. Use smaller population (10-15) for testing
2. Reduce generations (10) for quick experiments
3. Use `--fixed-environment` to reduce variance
4. Profile with fewer trials (5) during development

---

## 🎓 Best Practices

### **Dataset Creation:**
- ✅ Start with small, perception-only dataset
- ✅ Balance action distribution (stratified split)
- ❌ Don't include random history (causes biases)

### **Fine-Tuning:**
- ✅ Use hyperparameter search (10+ trials)
- ✅ Optimize for diversity, not just accuracy
- ✅ Save only best checkpoint per run
- ❌ Don't overtrain (3 epochs max)

### **Evaluation:**
- ✅ Always run evaluation before preparing parents
- ✅ Sort by evolution score, not accuracy
- ✅ Check action distributions for biases

### **Simulation:**
- ✅ Use multi-parent initialization (3-5 models)
- ✅ Start with higher mutation for exploration
- ✅ Monitor genetic diversity over time
- ❌ Don't use single parent (low diversity)

---

## 📚 Related Documentation

- [System Architecture](system_architecture.md) - Technical overview
- [Project Context](../.windsurf/project_context.md) - Goals and conventions
- [Fine-Tuning Details](../fine_tuning/README.md) - Deep dive into training

---

## 🚀 Next Steps

After completing the workflow:

1. **Analyze Results** - Check fitness curves, genetic diversity, behavioral patterns
2. **Visualize** - Look at champion paths, heatmaps, action distributions
3. **Iterate** - Adjust hyperparameters based on results
4. **Scale Up** - Increase population and generations for final runs
5. **Document** - Record interesting findings and parameter combinations

Happy evolving! 🧬

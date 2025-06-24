# Unit Testing Documentation untuk Komodo Mlipir Algorithm

## Overview

Unit test yang telah dibuat mencakup testing komprehensif untuk semua aspek dari Komodo Mlipir Algorithm (KMA), termasuk:

- **Initialization Tests**: Validasi parameter dan inisialisasi yang benar
- **Method Tests**: Testing untuk setiap method internal
- **Optimization Tests**: Testing proses optimasi dengan berbagai fungsi objektif
- **Adaptive Schema Tests**: Testing fitur adaptive population
- **Edge Case Tests**: Testing skenario khusus dan edge cases
- **Integration Tests**: Testing skenario real-world
- **Performance Tests**: Testing performa untuk problem besar

## Struktur Test

### 1. TestKomodoMlipirAlgorithmInitialization
Test untuk memastikan:
- Parameter default diset dengan benar
- Validasi parameter berfungsi dengan baik
- Populasi diinisialisasi dalam search space
- Error handling untuk input invalid

### 2. TestKomodoMlipirAlgorithmMethods
Test untuk method internal:
- `_calculate_fitness`: Perhitungan fitness
- `_clip_to_bounds`: Pembatasan ke search space
- `_sort_by_fitness`: Pengurutan populasi
- `_divide_population`: Pembagian populasi
- `_move_big_males`, `_move_female`, `_move_small_males`: Pergerakan populasi
- Dan method lainnya

### 3. TestKomodoMlipirAlgorithmOptimization
Test optimasi dengan berbagai fungsi:
- Sphere function (simple convex)
- Rosenbrock function (non-convex valley)
- Multi-dimensional problems
- Convergence testing

### 4. TestKomodoMlipirAlgorithmAdaptiveSchema
Test untuk fitur adaptive population:
- Population shrinking saat improvement
- Population expansion saat stagnant
- Respect untuk batas min/max population

### 5. TestKomodoMlipirAlgorithmEdgeCases
Test untuk edge cases:
- Single dimension optimization
- Asymmetric search spaces
- Functions dengan multiple local optima
- Population diversity

### 6. TestKomodoMlipirAlgorithmIntegration
Test integrasi:
- Constrained optimization via penalty
- Reproducibility dengan random state
- Verbose output testing

## Cara Menjalankan Tests

### Setup Environment

1. Install dependencies:
```bash
pip install -r requirements-test.txt
```

2. Pastikan file KMA ada di path yang benar:
```
project/
├── kma.py              # File implementasi KMA
├── test_kma.py         # File unit tests
├── pytest.ini          # Konfigurasi pytest
├── run_tests.py        # Test runner script
└── requirements-test.txt
```

### Menjalankan Tests

#### 1. Basic Testing
```bash
# Run all tests except slow ones
pytest test_kma.py

# Atau menggunakan script
python run_tests.py
```

#### 2. Run All Tests (termasuk slow tests)
```bash
pytest test_kma.py -m ""

# Atau
python run_tests.py --full
```

#### 3. Run Only Unit Tests
```bash
pytest test_kma.py -m "not slow and not integration"

# Atau
python run_tests.py --unit
```

#### 4. Run dengan Coverage Report
```bash
pytest test_kma.py --cov=kma --cov-report=html

# Atau
python run_tests.py --coverage
```

#### 5. Run Quick Tests Only
```bash
pytest test_kma.py -k "not Performance"

# Atau
python run_tests.py --quick
```

#### 6. Run Specific Test Class
```bash
pytest test_kma.py::TestKomodoMlipirAlgorithmInitialization -v
```

#### 7. Run Specific Test Method
```bash
pytest test_kma.py::TestKomodoMlipirAlgorithmMethods::test_calculate_fitness -v
```

## Output Examples

### Successful Test Run
```
============================= test session starts ==============================
collected 47 items

test_kma.py::TestKomodoMlipirAlgorithmInitialization::test_default_initialization PASSED
test_kma.py::TestKomodoMlipirAlgorithmInitialization::test_custom_initialization PASSED
...
test_kma.py::TestKomodoMlipirAlgorithmIntegration::test_reproducibility PASSED

==================== 47 passed in 12.34s ====================
```

### Coverage Report
```
----------- coverage: platform linux, python 3.9.0 -----------
Name     Stmts   Miss  Cover   Missing
--------------------------------------
kma.py     245      5    98%   123-127
--------------------------------------
TOTAL      245      5    98%
```

## Debugging Failed Tests

Jika ada test yang gagal:

1. **Run dengan verbose output**:
```bash
pytest test_kma.py -vv --tb=long
```

2. **Run specific failing test**:
```bash
pytest test_kma.py::TestClassName::test_method_name -vv
```

3. **Use pytest debugger**:
```bash
pytest test_kma.py --pdb
```

4. **Print debug info**:
```bash
pytest test_kma.py -s  # Don't capture stdout
```

## Writing New Tests

Template untuk menambah test baru:

```python
class TestNewFeature:
    """Test untuk fitur baru."""
    
    @pytest.fixture
    def setup_kma(self):
        """Setup KMA instance."""
        def test_function(x):
            return sum(x)
        
        return KomodoMlipirAlgorithm(
            fitness_function=test_function,
            search_space=[(-1, 1)]
        )
    
    def test_new_feature(self, setup_kma):
        """Test fitur baru."""
        kma = setup_kma
        
        # Test implementation
        result = kma.some_new_method()
        
        # Assertions
        assert result == expected_value
```

## Best Practices

1. **Test Independence**: Setiap test harus independent
2. **Clear Test Names**: Nama test harus deskriptif
3. **Arrange-Act-Assert**: Struktur test yang jelas
4. **Use Fixtures**: Untuk setup yang reusable
5. **Mock External Dependencies**: Untuk isolated testing
6. **Test Edge Cases**: Jangan hanya happy path
7. **Performance Tests**: Tandai dengan `@pytest.mark.slow`

## Continuous Integration

Untuk CI/CD, gunakan command:

```yaml
# GitHub Actions example
- name: Run tests
  run: |
    pip install -r requirements-test.txt
    pytest test_kma.py --cov=kma --cov-fail-under=90
```

## Troubleshooting

### Common Issues

1. **Import Error**:
   - Pastikan `kma.py` ada di path yang sama
   - Atau adjust PYTHONPATH

2. **Random Test Failures**:
   - Check random seed consistency
   - Increase tolerance untuk floating point comparisons

3. **Slow Tests**:
   - Skip dengan `-m "not slow"`
   - Atau optimize test parameters

4. **Memory Issues**:
   - Reduce population size di performance tests
   - Run tests in batches
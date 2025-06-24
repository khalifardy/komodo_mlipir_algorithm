"""
Unit tests untuk Komodo Mlipir Algorithm (KMA).

Menggunakan pytest untuk testing framework.
Run dengan: pytest test_kma.py -v
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import math
from typing import List, Tuple

# Import kelas yang akan ditest
# Asumsi: KomodoMlipirAlgorithm ada di file kma.py
from optimizer.KomodoMlipirAlgorithm import KomodoMlipirAlgorithm,KMA


class TestKomodoMlipirAlgorithmInitialization:
    """Test untuk inisialisasi dan validasi parameter."""
    
    def test_default_initialization(self):
        """Test inisialisasi dengan parameter default."""
        def dummy_fitness(x):
            return sum(x)
        
        kma = KomodoMlipirAlgorithm(
            fitness_function=dummy_fitness,
            search_space=[(-10, 10), (-10, 10)]
        )
        
        assert kma.population_size == 5
        assert kma.male_proportion == 0.5
        assert kma.mlipir_rate == 0.5
        assert kma.max_iterations == 1000
        assert kma.random_state == 42
        assert kma.parthenogenesis_radius == 0.1
        assert kma.stop_criteria == 0.01
        assert kma.stop is False
        assert kma.n_big_males == 2  # floor((1-0.5)*5) = 2
        
    def test_custom_initialization(self):
        """Test inisialisasi dengan parameter custom."""
        def dummy_fitness(x):
            return sum(x)
        
        kma = KomodoMlipirAlgorithm(
            population_size=20,
            male_proportion=0.3,
            mlipir_rate=0.7,
            fitness_function=dummy_fitness,
            search_space=[(-5, 5)],
            max_iterations=500,
            random_state=123,
            parthenogenesis_radius=0.2,
            stop_criteria=0.001,
            stop=True
        )
        
        assert kma.population_size == 20
        assert kma.male_proportion == 0.3
        assert kma.mlipir_rate == 0.7
        assert kma.max_iterations == 500
        assert kma.random_state == 123
        assert kma.parthenogenesis_radius == 0.2
        assert kma.stop_criteria == 0.001
        assert kma.stop is True
        assert kma.n_big_males == 14  # floor((1-0.3)*20) = 14
        
    def test_population_initialization_shape(self):
        """Test bentuk populasi yang diinisialisasi."""
        def dummy_fitness(x):
            return sum(x)
        
        dimensions = 3
        pop_size = 10
        
        kma = KomodoMlipirAlgorithm(
            population_size=pop_size,
            fitness_function=dummy_fitness,
            search_space=[(-1, 1)] * dimensions
        )
        
        assert kma.population.shape == (pop_size, dimensions)
        assert kma.fitness_values.shape == (pop_size,)
        
    def test_population_initialization_bounds(self):
        """Test apakah populasi diinisialisasi dalam search space."""
        def dummy_fitness(x):
            return sum(x)
        
        search_space = [(-5, 5), (0, 10), (-2, 2)]
        
        kma = KomodoMlipirAlgorithm(
            population_size=100,
            fitness_function=dummy_fitness,
            search_space=search_space
        )
        
        for i, (lower, upper) in enumerate(search_space):
            assert np.all(kma.population[:, i] >= lower)
            assert np.all(kma.population[:, i] <= upper)
    
    def test_population_size_validation(self):
        """Test validasi ukuran populasi minimum."""
        def dummy_fitness(x):
            return sum(x)
        
        with pytest.raises(ValueError, match="Population size must be at least"):
            KomodoMlipirAlgorithm(
                population_size=2,
                fitness_function=dummy_fitness,
                search_space=[(-1, 1)]
            )
    
    def test_male_proportion_validation(self):
        """Test validasi proporsi jantan."""
        def dummy_fitness(x):
            return sum(x)
        
        with pytest.raises(ValueError, match="Male proportion must be between"):
            KomodoMlipirAlgorithm(
                male_proportion=0.05,
                fitness_function=dummy_fitness,
                search_space=[(-1, 1)]
            )
        
        with pytest.raises(ValueError, match="Male proportion must be between"):
            KomodoMlipirAlgorithm(
                male_proportion=1.5,
                fitness_function=dummy_fitness,
                search_space=[(-1, 1)]
            )
    
    def test_mlipir_rate_validation(self):
        """Test validasi mlipir rate."""
        def dummy_fitness(x):
            return sum(x)
        
        with pytest.raises(ValueError, match="Mlipir rate must be between"):
            KomodoMlipirAlgorithm(
                mlipir_rate=-0.1,
                fitness_function=dummy_fitness,
                search_space=[(-1, 1)]
            )
        
        with pytest.raises(ValueError, match="Mlipir rate must be between"):
            KomodoMlipirAlgorithm(
                mlipir_rate=1.1,
                fitness_function=dummy_fitness,
                search_space=[(-1, 1)]
            )
    
    def test_fitness_function_validation(self):
        """Test validasi fitness function."""
        with pytest.raises(ValueError, match="Fitness function must be provided"):
            KomodoMlipirAlgorithm(
                fitness_function=None,
                search_space=[(-1, 1)]
            )
    
    def test_search_space_validation(self):
        """Test validasi search space."""
        def dummy_fitness(x):
            return sum(x)
        
        with pytest.raises(ValueError, match="Search space must be defined"):
            KomodoMlipirAlgorithm(
                fitness_function=dummy_fitness,
                search_space=None
            )
        
        with pytest.raises(ValueError, match="Search space must be defined"):
            KomodoMlipirAlgorithm(
                fitness_function=dummy_fitness,
                search_space=[]
            )
    
    def test_max_iterations_validation(self):
        """Test validasi maksimum iterasi."""
        def dummy_fitness(x):
            return sum(x)
        
        with pytest.raises(ValueError, match="Maximum iterations must be at least 1"):
            KomodoMlipirAlgorithm(
                fitness_function=dummy_fitness,
                search_space=[(-1, 1)],
                max_iterations=0
            )
    
    def test_alias_compatibility(self):
        """Test bahwa alias KMA masih berfungsi."""
        def dummy_fitness(x):
            return sum(x)
        
        kma = KMA(
            fitness_function=dummy_fitness,
            search_space=[(-1, 1)]
        )
        
        assert isinstance(kma, KomodoMlipirAlgorithm)


class TestKomodoMlipirAlgorithmMethods:
    """Test untuk method-method internal."""
    
    @pytest.fixture
    def setup_kma(self):
        """Setup KMA instance untuk testing."""
        def sphere_function(x):
            return -sum(xi**2 for xi in x)
        
        kma = KomodoMlipirAlgorithm(
            population_size=10,
            male_proportion=0.5,
            fitness_function=sphere_function,
            search_space=[(-5, 5), (-5, 5)],
            random_state=42
        )
        return kma
    
    def test_calculate_fitness(self, setup_kma):
        """Test perhitungan fitness."""
        kma = setup_kma
        
        # Test dengan individu tunggal
        individual = np.array([1.0, 2.0])
        fitness = kma._calculate_fitness(np.array([individual]))
        expected = -(1.0**2 + 2.0**2)  # -5.0
        assert np.isclose(fitness[0], expected)
        
        # Test dengan multiple individuals
        individuals = np.array([
            [1.0, 2.0],
            [0.0, 0.0],
            [-1.0, -1.0]
        ])
        fitness = kma._calculate_fitness(individuals)
        assert len(fitness) == 3
        assert np.isclose(fitness[0], -5.0)
        assert np.isclose(fitness[1], 0.0)
        assert np.isclose(fitness[2], -2.0)
    
    def test_clip_to_bounds(self, setup_kma):
        """Test clipping individu ke search space."""
        kma = setup_kma
        
        # Test individu yang keluar bounds
        individual = np.array([6.0, -6.0])
        clipped = kma._clip_to_bounds(individual)
        assert clipped[0] == 5.0  # Upper bound
        assert clipped[1] == -5.0  # Lower bound
        
        # Test individu yang dalam bounds
        individual = np.array([2.0, -3.0])
        clipped = kma._clip_to_bounds(individual)
        assert np.array_equal(clipped, individual)
    
    def test_sort_by_fitness(self, setup_kma):
        """Test pengurutan populasi berdasarkan fitness."""
        kma = setup_kma
        
        population = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [0.5, 0.5]
        ])
        fitness_values = np.array([-2.0, -8.0, -0.5])
        
        sorted_pop, sorted_fit = kma._sort_by_fitness(population, fitness_values)
        
        # Fitness terbaik (tertinggi) di index 0
        assert sorted_fit[0] == -0.5
        assert sorted_fit[1] == -2.0
        assert sorted_fit[2] == -8.0
        
        assert np.array_equal(sorted_pop[0], [0.5, 0.5])
        assert np.array_equal(sorted_pop[1], [1.0, 1.0])
        assert np.array_equal(sorted_pop[2], [2.0, 2.0])
    
    def test_divide_population(self, setup_kma):
        """Test pembagian populasi."""
        kma = setup_kma
        
        # Create sorted population of size 10
        sorted_population = np.arange(20).reshape(10, 2)
        
        big_males, female, small_males = kma._divide_population(sorted_population)
        
        # n_big_males = floor((1-0.5)*10) = 5
        assert len(big_males) == 5
        assert np.array_equal(female, sorted_population[5])
        assert len(small_males) == 4
        
        # Check ordering
        assert np.array_equal(big_males, sorted_population[:5])
        assert np.array_equal(small_males, sorted_population[6:])
    
    def test_calculate_male_interaction(self, setup_kma):
        """Test perhitungan interaksi antar jantan."""
        kma = setup_kma
        
        male_i = np.array([1.0, 1.0])
        male_j = np.array([2.0, 2.0])
        fitness_i = -2.0
        fitness_j = -8.0
        
        # Mock random values
        with patch.object(kma.rng, 'standard_normal', side_effect=[0.5, 0.3]):
            interaction = kma._calculate_male_interaction(
                male_i, male_j, fitness_i, fitness_j
            )
            
            # fitness_j < fitness_i and r2 < 0.5
            expected = 0.5 * (male_j - male_i)
            assert np.allclose(interaction, expected)
    
    def test_perform_mating(self, setup_kma):
        """Test proses mating."""
        kma = setup_kma
        
        male = np.array([1.0, 1.0])
        female = np.array([2.0, 2.0])
        
        # Mock random value for crossover
        with patch.object(kma.rng, 'standard_normal', return_value=0.3):
            offspring, fitness = kma._perform_mating(male, female)
            
            assert len(offspring) == 1
            assert len(fitness) == 1
            assert offspring[0].shape == male.shape
    
    def test_perform_parthenogenesis(self, setup_kma):
        """Test proses parthenogenesis."""
        kma = setup_kma
        
        female = np.array([1.0, 1.0])
        
        with patch.object(kma.rng, 'standard_normal', return_value=0.5):
            offspring, fitness = kma._perform_parthenogenesis(female)
            
            assert len(offspring) == 1
            assert len(fitness) == 1
            assert offspring[0].shape == female.shape
            
            # Check bounds
            assert np.all(offspring[0] >= -5.0)
            assert np.all(offspring[0] <= 5.0)
    
    def test_generate_new_individuals(self, setup_kma):
        """Test generasi individu baru."""
        kma = setup_kma
        
        best_individual = np.array([0.0, 0.0])
        n_individuals = 5
        
        new_individuals = kma._generate_new_individuals(
            best_individual, n_individuals
        )
        
        assert len(new_individuals) == n_individuals
        assert new_individuals.shape == (n_individuals, 2)
        
        # Check bounds
        for individual in new_individuals:
            assert np.all(individual >= -5.0)
            assert np.all(individual <= 5.0)
    
    def test_update_best_solution(self, setup_kma):
        """Test update solusi terbaik."""
        kma = setup_kma
        
        # Set population dan fitness
        kma.population = np.array([
            [1.0, 1.0],
            [0.1, 0.1],  # Best
            [2.0, 2.0]
        ])
        kma.fitness_values = np.array([-2.0, -0.02, -8.0])
        
        kma._update_best_solution()
        
        assert kma.best_fitness == -0.02
        assert np.array_equal(kma.best_solution, [0.1, 0.1])
        assert len(kma.history["best_fitness"]) == 1
        assert len(kma.history["best_solution"]) == 1
    
    def test_check_convergence(self, setup_kma):
        """Test pengecekan konvergensi."""
        kma = setup_kma
        
        # Not enough history
        assert kma._check_convergence() is False
        
        # Add converged history
        kma.history["best_fitness"] = [-1.0] * 10
        assert kma._check_convergence() is True
        
        # Add non-converged history
        kma.history["best_fitness"] = list(range(10))
        assert kma._check_convergence() is False


class TestKomodoMlipirAlgorithmOptimization:
    """Test untuk proses optimasi utama."""
    
    def test_sphere_function_optimization(self):
        """Test optimasi fungsi sphere sederhana."""
        def sphere_function(x):
            return -sum(xi**2 for xi in x)
        
        kma = KomodoMlipirAlgorithm(
            population_size=20,
            male_proportion=0.5,
            fitness_function=sphere_function,
            search_space=[(-5, 5), (-5, 5)],
            max_iterations=50,
            random_state=42
        )
        
        kma.fit(verbose=False)
        
        results = kma.get_results()
        
        # Check results structure
        assert "best_solution" in results
        assert "best_fitness" in results
        assert "history" in results
        assert "n_iterations" in results
        
        # Optimum should be near [0, 0] with fitness near 0
        assert results["best_fitness"] > -1.0  # Should be close to 0
        assert np.allclose(results["best_solution"], [0, 0], atol=1.0)
        assert results["n_iterations"] == 50
    
    def test_rosenbrock_function_optimization(self):
        """Test optimasi fungsi Rosenbrock."""
        def rosenbrock(x):
            return -(100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2)
        
        kma = KomodoMlipirAlgorithm(
            population_size=30,
            male_proportion=0.4,
            fitness_function=rosenbrock,
            search_space=[(-2, 2), (-2, 2)],
            max_iterations=100,
            random_state=42
        )
        
        kma.fit(verbose=False)
        
        results = kma.get_results()
        
        # Rosenbrock optimum at [1, 1] with fitness 0
        assert results["best_fitness"] > -10.0
        assert len(results["history"]["best_fitness"]) == 100
    
    def test_fit_with_adaptive_schema(self):
        """Test optimasi dengan adaptive schema."""
        def simple_function(x):
            return -x[0]**2
        
        kma = KomodoMlipirAlgorithm(
            population_size=10,
            fitness_function=simple_function,
            search_space=[(-10, 10)],
            max_iterations=20,
            random_state=42
        )
        
        kma.fit(adaptive_schema=True, min_population=5, max_population=20, verbose=False)
        
        # Population size might have changed
        assert 5 <= len(kma.population) <= 20
    
    def test_fit_with_convergence_stop(self):
        """Test optimasi dengan kriteria konvergensi."""
        def constant_function(x):
            return 1.0  # Always returns same value
        
        kma = KomodoMlipirAlgorithm(
            population_size=10,
            fitness_function=constant_function,
            search_space=[(-1, 1)],
            max_iterations=100,
            stop_criteria=0.001,
            stop=True,
            random_state=42
        )
        
        kma.fit(verbose=False)
        
        results = kma.get_results()
        
        # Should converge early
        assert results["n_iterations"] < 100
        assert results["best_fitness"] == 1.0
    
    def test_multidimensional_optimization(self):
        """Test optimasi dengan dimensi tinggi."""
        dimensions = 10
        
        def sphere_nd(x):
            return -sum(xi**2 for xi in x)
        
        kma = KomodoMlipirAlgorithm(
            population_size=50,
            fitness_function=sphere_nd,
            search_space=[(-10, 10)] * dimensions,
            max_iterations=100,
            random_state=42
        )
        
        kma.fit(verbose=False)
        
        results = kma.get_results()
        
        assert len(results["best_solution"]) == dimensions
        assert results["best_fitness"] > -100.0  # Should find reasonably good solution


class TestKomodoMlipirAlgorithmAdaptiveSchema:
    """Test untuk fitur adaptive schema."""
    
    def test_adaptive_schema_population_shrink(self):
        """Test pengurangan populasi saat ada improvement."""
        def dummy_fitness(x):
            return sum(x)
        
        kma = KomodoMlipirAlgorithm(
            population_size=25,
            fitness_function=dummy_fitness,
            search_space=[(-1, 1)],
            random_state=42
        )
        
        # Simulate improving fitness history
        kma.history["best_fitness"] = [1.0, 2.0, 3.0]
        kma.fitness_values = np.random.rand(25)
        
        initial_size = len(kma.population)
        kma._apply_adaptive_schema(min_population=20, max_population=30)
        
        # Population should shrink by 5
        assert len(kma.population) == initial_size - 5
        assert kma.population_size == initial_size - 5
    
    def test_adaptive_schema_population_expand(self):
        """Test penambahan populasi saat stagnant."""
        def dummy_fitness(x):
            return sum(x)
        
        kma = KomodoMlipirAlgorithm(
            population_size=15,
            fitness_function=dummy_fitness,
            search_space=[(-1, 1)],
            random_state=42
        )
        
        # Simulate stagnant fitness history
        kma.history["best_fitness"] = [1.0, 1.0, 1.0]
        kma.fitness_values = np.random.rand(15)
        
        initial_size = len(kma.population)
        kma._apply_adaptive_schema(min_population=10, max_population=25)
        
        # Population should expand by 5
        assert len(kma.population) == initial_size + 5
        assert kma.population_size == initial_size + 5
    
    def test_adaptive_schema_min_population_limit(self):
        """Test batas minimum populasi."""
        def dummy_fitness(x):
            return sum(x)
        
        kma = KomodoMlipirAlgorithm(
            population_size=23,  # 23 - 5 = 18 < 20 (min)
            fitness_function=dummy_fitness,
            search_space=[(-1, 1)],
            random_state=42
        )
        
        # Simulate improving fitness
        kma.history["best_fitness"] = [1.0, 2.0, 3.0]
        kma.fitness_values = np.random.rand(23)
        
        initial_size = len(kma.population)
        kma._apply_adaptive_schema(min_population=20, max_population=30)
        
        # Population should not shrink below minimum
        assert len(kma.population) == initial_size  # No change
    
    def test_adaptive_schema_max_population_limit(self):
        """Test batas maksimum populasi."""
        def dummy_fitness(x):
            return sum(x)
        
        kma = KomodoMlipirAlgorithm(
            population_size=28,  # 28 + 5 = 33 > 30 (max)
            fitness_function=dummy_fitness,
            search_space=[(-1, 1)],
            random_state=42
        )
        
        # Simulate stagnant fitness
        kma.history["best_fitness"] = [1.0, 1.0, 1.0]
        kma.fitness_values = np.random.rand(28)
        
        initial_size = len(kma.population)
        kma._apply_adaptive_schema(min_population=20, max_population=30)
        
        # Population should not expand beyond maximum
        assert len(kma.population) == initial_size  # No change


class TestKomodoMlipirAlgorithmEdgeCases:
    """Test untuk edge cases dan skenario khusus."""
    
    def test_single_dimension_optimization(self):
        """Test optimasi 1 dimensi."""
        def parabola(x):
            return -(x[0] - 3)**2 + 10
        
        kma = KomodoMlipirAlgorithm(
            population_size=15,
            fitness_function=parabola,
            search_space=[(0, 6)],
            max_iterations=50,
            random_state=42
        )
        
        kma.fit(verbose=False)
        
        results = kma.get_results()
        
        # Optimum at x=3
        assert abs(results["best_solution"][0] - 3.0) < 0.5
        assert results["best_fitness"] > 9.0
    
    def test_negative_fitness_values(self):
        """Test dengan nilai fitness negatif."""
        def negative_sphere(x):
            return sum(xi**2 for xi in x)  # Positive sphere (we minimize)
        
        kma = KomodoMlipirAlgorithm(
            population_size=20,
            fitness_function=negative_sphere,
            search_space=[(-5, 5), (-5, 5)],
            max_iterations=50,
            random_state=42
        )
        
        kma.fit(verbose=False)
        
        # Since we're maximizing positive sphere, worst solution is far from origin
        results = kma.get_results()
        assert results["best_fitness"] > 10.0
    
    def test_asymmetric_search_space(self):
        """Test dengan search space tidak simetris."""
        def weighted_sum(x):
            return x[0] + 2*x[1] + 3*x[2]
        
        kma = KomodoMlipirAlgorithm(
            population_size=25,
            fitness_function=weighted_sum,
            search_space=[(0, 10), (-5, 5), (-20, -10)],
            max_iterations=30,
            random_state=42
        )
        
        kma.fit(verbose=False)
        
        results = kma.get_results()
        
        # Optimum should be at upper bounds for positive weights
        # and lower bounds for negative weights
        assert results["best_solution"][0] > 8.0  # Near 10
        assert results["best_solution"][1] > 3.0   # Near 5
        assert results["best_solution"][2] < -18.0 # Near -20
    
    def test_discrete_like_function(self):
        """Test dengan fungsi yang memiliki banyak local optima."""
        def rastrigin(x):
            n = len(x)
            return -(10 * n + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x))
        
        kma = KomodoMlipirAlgorithm(
            population_size=50,
            fitness_function=rastrigin,
            search_space=[(-5.12, 5.12)] * 2,
            max_iterations=100,
            random_state=42
        )
        
        kma.fit(verbose=False)
        
        results = kma.get_results()
        
        # Rastrigin has global optimum at origin
        assert results["best_fitness"] > -50.0  # Should find reasonably good solution
    
    def test_population_diversity_maintained(self):
        """Test bahwa diversitas populasi terjaga."""
        def simple_function(x):
            return -x[0]**2
        
        kma = KomodoMlipirAlgorithm(
            population_size=20,
            fitness_function=simple_function,
            search_space=[(-10, 10)],
            max_iterations=10,
            random_state=42
        )
        
        kma.fit(verbose=False)
        
        # Check population diversity
        unique_solutions = np.unique(kma.population, axis=0)
        
        # Should maintain some diversity
        assert len(unique_solutions) > len(kma.population) * 0.5


class TestKomodoMlipirAlgorithmIntegration:
    """Test integrasi dan skenario real-world."""
    
    def test_constrained_optimization_via_penalty(self):
        """Test optimasi dengan constraint melalui penalty function."""
        def constrained_function(x):
            # Objective: maximize x[0] + x[1]
            # Constraint: x[0]^2 + x[1]^2 <= 1 (unit circle)
            objective = x[0] + x[1]
            
            # Penalty for constraint violation
            constraint_violation = max(0, x[0]**2 + x[1]**2 - 1)
            penalty = 100 * constraint_violation
            
            return objective - penalty
        
        kma = KomodoMlipirAlgorithm(
            population_size=30,
            fitness_function=constrained_function,
            search_space=[(-2, 2), (-2, 2)],
            max_iterations=100,
            random_state=42
        )
        
        kma.fit(verbose=False)
        
        results = kma.get_results()
        
        # Optimum should be at x = [1/sqrt(2), 1/sqrt(2)]
        x_opt = results["best_solution"]
        
        # Check constraint satisfaction
        assert x_opt[0]**2 + x_opt[1]**2 <= 1.1  # Small tolerance
        
        # Check objective value
        assert results["best_fitness"] > 1.0
    
    def test_reproducibility_with_random_state(self):
        """Test reproduktibilitas dengan random state."""
        def test_function(x):
            return -sum(xi**2 for xi in x)
        
        results = []
        
        for _ in range(3):
            kma = KomodoMlipirAlgorithm(
                population_size=10,
                fitness_function=test_function,
                search_space=[(-5, 5)] * 2,
                max_iterations=20,
                random_state=42
            )
            
            kma.fit(verbose=False)
            results.append(kma.get_results())
        
        # All runs should produce identical results
        for i in range(1, len(results)):
            assert results[i]["best_fitness"] == results[0]["best_fitness"]
            assert np.array_equal(
                results[i]["best_solution"], 
                results[0]["best_solution"]
            )
    
    def test_different_random_states_produce_different_results(self):
        """Test bahwa random state berbeda menghasilkan hasil berbeda."""
        def test_function(x):
            return -sum(xi**2 for xi in x)
        
        results = []
        
        for seed in [42, 123, 456]:
            kma = KomodoMlipirAlgorithm(
                population_size=10,
                fitness_function=test_function,
                search_space=[(-5, 5)] * 2,
                max_iterations=10,
                random_state=seed
            )
            
            kma.fit(verbose=False)
            results.append(kma.get_results())
        
        # Different seeds should produce different results
        solutions = [r["best_solution"] for r in results]
        
        # Check that not all solutions are identical
        all_different = True
        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                if np.array_equal(solutions[i], solutions[j]):
                    all_different = False
                    break
        
        assert all_different or len(set(r["best_fitness"] for r in results)) > 1
    
    def test_verbose_output(self, capsys):
        """Test output verbose."""
        def simple_function(x):
            return -x[0]**2
        
        kma = KomodoMlipirAlgorithm(
            population_size=5,
            fitness_function=simple_function,
            search_space=[(-1, 1)],
            max_iterations=2,
            random_state=42
        )
        
        kma.fit(verbose=True)
        
        captured = capsys.readouterr()
        
        # Check verbose output contains expected strings
        assert "Iteration" in captured.out
        assert "Population size" in captured.out
        assert "Moving big males" in captured.out
        assert "Moving female" in captured.out
        assert "Moving small males" in captured.out
        assert "Best fitness" in captured.out


# Performance tests (optional, can be skipped for faster testing)
@pytest.mark.slow
class TestKomodoMlipirAlgorithmPerformance:
    """Test performa untuk dataset besar."""
    
    def test_high_dimensional_optimization(self):
        """Test optimasi dimensi tinggi (50D)."""
        dimensions = 50
        
        def sphere_50d(x):
            return -sum(xi**2 for xi in x)
        
        kma = KomodoMlipirAlgorithm(
            population_size=100,
            fitness_function=sphere_50d,
            search_space=[(-10, 10)] * dimensions,
            max_iterations=200,
            random_state=42
        )
        
        import time
        start_time = time.time()
        kma.fit(verbose=False)
        end_time = time.time()
        
        results = kma.get_results()
        
        # Should complete in reasonable time
        assert end_time - start_time < 60  # Less than 60 seconds
        
        # Should find reasonable solution
        assert results["best_fitness"] > -1000.0
    
    def test_large_population_optimization(self):
        """Test dengan populasi besar."""
        def simple_function(x):
            return -sum(xi**2 for xi in x)
        
        kma = KomodoMlipirAlgorithm(
            population_size=500,
            fitness_function=simple_function,
            search_space=[(-5, 5)] * 5,
            max_iterations=50,
            random_state=42
        )
        
        import time
        start_time = time.time()
        kma.fit(verbose=False)
        end_time = time.time()
        
        # Should handle large population efficiently
        assert end_time - start_time < 30  # Less than 30 seconds
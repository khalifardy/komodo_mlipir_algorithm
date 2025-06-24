import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class BenchmarkFunctions:
    """
    Koleksi fungsi benchmark untuk optimisasi
    """
    
    @staticmethod
    def sphere(x):
        """Sphere Function - Unimodal"""
        return np.sum(x**2)
    
    @staticmethod
    def rosenbrock(x):
        """Rosenbrock Function - Unimodal dengan valley"""
        return np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    
    @staticmethod
    def ackley(x):
        """Ackley Function - Multimodal"""
        n = len(x)
        a, b, c = 20, 0.2, 2*np.pi
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c*x))
        return -a * np.exp(-b * np.sqrt(sum1/n)) - np.exp(sum2/n) + a + np.e
    
    @staticmethod
    def griewank(x):
        """Griewank Function - Multimodal"""
        sum_part = np.sum(x**2) / 4000
        prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
        return sum_part - prod_part + 1
    
    @staticmethod
    def rastrigin(x):
        """Rastrigin Function - Multimodal dengan banyak local minima"""
        n = len(x)
        return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    @staticmethod
    def schwefel(x):
        """Schwefel Function - Multimodal"""
        n = len(x)
        return 418.9829*n - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    @staticmethod
    def easom(x):
        """Easom Function - Unimodal dengan global minimum yang sulit ditemukan"""
        if len(x) != 2:
            raise ValueError("Easom function hanya untuk 2 dimensi")
        return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0]-np.pi)**2 - (x[1]-np.pi)**2)
    
    @staticmethod
    def michalewicz(x, m=10):
        """Michalewicz Function - Multimodal"""
        result = 0
        for i, xi in enumerate(x):
            result += np.sin(xi) * (np.sin((i+1) * xi**2 / np.pi))**(2*m)
        return -result

def visualize_function_2d(func, bounds, title, num_points=100):
    """
    Visualisasi fungsi 2D
    """
    x = np.linspace(bounds[0], bounds[1], num_points)
    y = np.linspace(bounds[0], bounds[1], num_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(num_points):
        for j in range(num_points):
            Z[i,j] = func([X[i,j], Y[i,j]])
    
    fig = plt.figure(figsize=(12, 5))
    
    # Contour plot
    ax1 = fig.add_subplot(121)
    contour = ax1.contour(X, Y, Z, levels=20)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.set_title(f'{title} - Contour Plot')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax2.set_title(f'{title} - 3D Surface')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('f(x1, x2)')
    fig.colorbar(surf)
    
    plt.tight_layout()
    plt.show()

def benchmark_optimizer(optimizer_func, benchmark_func, bounds, dimensions, max_iterations=1000):
    """
    Menguji optimizer pada fungsi benchmark
    
    Args:
        optimizer_func: Fungsi optimizer yang akan diuji
        benchmark_func: Fungsi benchmark
        bounds: Tuple (min, max) untuk setiap dimensi
        dimensions: Jumlah dimensi
        max_iterations: Maksimum iterasi
    
    Returns:
        Dictionary dengan hasil benchmark
    """
    results = {
        'best_solution': None,
        'best_fitness': float('inf'),
        'convergence_history': [],
        'iterations': 0
    }
    
    # Implementasi optimizer sederhana (Random Search sebagai contoh)
    np.random.seed(42)
    
    for iteration in range(max_iterations):
        # Generate random solution
        candidate = np.random.uniform(bounds[0], bounds[1], dimensions)
        fitness = benchmark_func(candidate)
        
        if fitness < results['best_fitness']:
            results['best_fitness'] = fitness
            results['best_solution'] = candidate.copy()
        
        results['convergence_history'].append(results['best_fitness'])
        results['iterations'] = iteration + 1
        
        # Stopping criteria
        if results['best_fitness'] < 1e-10:
            break
    
    return results

# Contoh penggunaan
if __name__ == "__main__":
    bf = BenchmarkFunctions()
    
    # Test fungsi sphere
    x = np.array([1, 2, 3])
    print(f"Sphere([1,2,3]) = {bf.sphere(x)}")
    
    # Test fungsi ackley
    x = np.array([0, 0])
    print(f"Ackley([0,0]) = {bf.ackley(x)}")
    
    # Visualisasi fungsi Ackley 2D
    print("Membuat visualisasi fungsi Ackley...")
    # visualize_function_2d(bf.ackley, (-5, 5), "Ackley Function")
    
    # Benchmark optimizer sederhana
    print("\nMenjalankan benchmark pada fungsi Sphere...")
    results = benchmark_optimizer(None, bf.sphere, (-5, 5), 10, 1000)
    print(f"Best solution: {results['best_solution']}")
    print(f"Best fitness: {results['best_fitness']}")
    print(f"Iterations: {results['iterations']}")
    
    # Contoh karakteristik fungsi benchmark
    functions_info = {
        'Sphere': {
            'type': 'Unimodal',
            'global_min': 0,
            'at_point': '[0, 0, ..., 0]',
            'difficulty': 'Easy',
            'characteristics': 'Smooth, konveks'
        },
        'Rosenbrock': {
            'type': 'Unimodal',
            'global_min': 0,
            'at_point': '[1, 1, ..., 1]',
            'difficulty': 'Medium',
            'characteristics': 'Valley-shaped, konvergensi lambat'
        },
        'Ackley': {
            'type': 'Multimodal',
            'global_min': 0,
            'at_point': '[0, 0, ..., 0]',
            'difficulty': 'Hard',
            'characteristics': 'Banyak local minima'
        },
        'Rastrigin': {
            'type': 'Multimodal',
            'global_min': 0,
            'at_point': '[0, 0, ..., 0]',
            'difficulty': 'Very Hard',
            'characteristics': 'Sangat banyak local minima'
        }
    }
    
    print("\n=== Karakteristik Fungsi Benchmark ===")
    for name, info in functions_info.items():
        print(f"\n{name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
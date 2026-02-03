#@author: AI

import numpy as np
import matplotlib.pyplot as plt
import cooling_tower  # Your module
import unittest

class TestCoolingTower(unittest.TestCase):
    
    def test_cylinder(self):
        """Test a simple cylinder (r1=r2)"""
        r = 10.0
        h = 20.0
        # 1 segment
        radii = np.array([r, r])
        heights = np.array([h])
        
        calc_area = cooling_tower.total_surface_area(radii, heights)
        calc_vol = cooling_tower.total_volume(radii, heights)
        
        expected_area = 2 * np.pi * r * h
        expected_vol = np.pi * r**2 * h
        
        # Allow small floating point error
        self.assertAlmostEqual(calc_area, expected_area, places=5)
        self.assertAlmostEqual(calc_vol, expected_vol, places=5)
        print(f"Cylinder Test: Area={calc_area:.2f}, Vol={calc_vol:.2f} [PASSED]")

    def test_cone(self):
        """Test a full cone (r_top = 0)"""
        r_bot = 10.0
        r_top = 0.0
        h = 10.0
        radii = np.array([r_bot, r_top])
        heights = np.array([h])
        
        calc_area = cooling_tower.total_surface_area(radii, heights)
        calc_vol = cooling_tower.total_volume(radii, heights)
        
        slant = np.sqrt(r_bot**2 + h**2)
        expected_area = np.pi * r_bot * slant
        expected_vol = (np.pi * r_bot**2 * h) / 3.0
        
        self.assertAlmostEqual(calc_area, expected_area, places=5)
        self.assertAlmostEqual(calc_vol, expected_vol, places=5)
        print(f"Cone Test: Area={calc_area:.2f}, Vol={calc_vol:.2f} [PASSED]")

def plot_cooling_tower_profile(radii, heights, title="Cooling Tower Profile"):
    """
    Plots the 2D cross-section of the cooling tower.
    """
    # Construct Z coordinates
    # z0 = 0
    # z1 = h1
    # z2 = h1 + h2 ...
    z_coords = np.concatenate(([0], np.cumsum(heights)))
    
    # Calculate stats
    total_area = cooling_tower.total_surface_area(radii, heights)
    total_vol = cooling_tower.total_volume(radii, heights)
    
    plt.figure(figsize=(8, 10))
    
    # Left profile (-r) and Right profile (+r)
    plt.plot(radii, z_coords, 'b-o', label='Right Profile')
    plt.plot(-radii, z_coords, 'b-o', label='Left Profile')
    
    # Fill between to show the shape volume clearly
    plt.fill_betweenx(z_coords, -radii, radii, alpha=0.2, color='gray')
    
    # Annotate segments with their individual areas or just general info
    for i in range(len(heights)):
        mid_z = (z_coords[i] + z_coords[i+1]) / 2
        mid_r = (radii[i] + radii[i+1]) / 2
        
        # Calculate frustum data
        f_area = cooling_tower.frustum_surface_area(radii[i], radii[i+1], heights[i])
        f_vol = cooling_tower.frustum_volume(radii[i], radii[i+1], heights[i])
        
        plt.text(0, mid_z, f"Seg {i+1}\nA={f_area:.1f}\nV={f_vol:.1f}", 
                 ha='center', va='center', fontsize=8, color='red')

    plt.title(f"{title}\nTotal Area: {total_area:.2f} | Total Volume: {total_vol:.2f}")
    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    plt.grid(True)
    plt.axis('equal')
    
    filename = "cooling_tower_profile_test.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

if __name__ == "__main__":
    # 1. Run Unit Tests
    print("--- Running Unit Tests ---")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCoolingTower)
    unittest.TextTestRunner(verbosity=0).run(suite)
    
    # 2. Run Visualization for parameters from Group Project Spring 2026.pdf
    print("\n--- Generating Visualization for Group Project Spring 2026 ---")
    
    # Data from: @[Task6/Objectives Informations/Group Project Spring 2026.pdf]
    # Radii: [39.3  33.35 33.35 33.35 33.35 33.35 33.35 33.35 33.35 33.35 27.4 ]
    # Heights: [3.65 3.65 3.65 3.65 3.65 3.65 3.65 3.65 3.65 3.65]
    radii = np.array([39.3, 33.35, 33.35, 33.35, 33.35, 33.35, 33.35, 33.35, 33.35, 33.35, 27.4])
    heights = np.array([3.65, 3.65, 3.65, 3.65, 3.65, 3.65, 3.65, 3.65, 3.65, 3.65])
    
    print(f"Radii: {radii}")
    print(f"Heights: {heights}")
    
    plot_cooling_tower_profile(radii, heights, title="Cooling Tower Profile\n(Group Project Spring 2026)")


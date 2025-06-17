import random
import itertools
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import Voronoi, cKDTree


class BackgroundGen3D():
    def __init__(
        self, mode_bg_geom, mode_bg_noise, num_imgs, shape=[128, 128, 128], rand_int_scale=True,
        num_voronoi_min=5, num_voronoi_max=30, num_spheres_min=10, num_spheres_max=30, rad_min=20, 
        rad_max=50, max_attempts=1000, perlin_scales=[2, 4, 8, 16, 32, 64], perlin_min_std=0, 
        perlin_max_std=5, gaussian_min_std=0, gaussian_max_std=5, out_dir=None
    ):
        self.img_type = f'bg_geom_{mode_bg_geom}_bg_noise_{mode_bg_noise}'
        self.save_imgs = True if out_dir is not None else False
        self.out_dir = Path(out_dir) if out_dir is not None else None

        self.mode_bg_geom = mode_bg_geom
        self.mode_bg_noise = mode_bg_noise
        self.num_imgs = num_imgs

        self.shape = shape
        self.rand_int_scale = rand_int_scale

        self.num_voronoi_min = num_voronoi_min      # voronoi and worley
        self.num_voronoi_max = num_voronoi_max

        self.perlin_scales = perlin_scales          # perlin
        self.perlin_min_std = perlin_min_std
        self.perlin_max_std = perlin_max_std

        self.gaussian_min_std = gaussian_min_std    # gaussian
        self.gaussian_max_std = gaussian_max_std

        self.num_spheres_min = num_spheres_min      # shperes
        self.num_spheres_max = num_spheres_max
        self.rad_min = rad_min
        self.rad_max = rad_max
        self.max_attempts = max_attempts

 
    def __call__(self):
        # output_imgs = []
        output_img = None

        for idx in range(self.num_imgs):
            print(f'Generating img {idx}.')

            # init output img
            output_img = np.zeros(self.shape)

            # 1) generate bg geometry
            if self.mode_bg_geom == 'voronoi':      # polyhedrons
                bg_geom = self.generate_voronoi_regions()
            elif self.mode_bg_geom == 'spheres':    # spheres
                bg_geom = self.generate_spheres()
            elif self.mode_bg_geom == None:         # plain
                bg_geom = np.zeros(self.shape)
            else: 
                raise NotImplementedError
            
            # 2) sample for each bg geometry a texture
            geom_ids = np.unique(bg_geom)
            for geom_id in geom_ids:
                if self.mode_bg_noise == 'worley':
                    texture = self.generate_worley_noise()
                elif self.mode_bg_noise == 'perlin': 
                    texture = self.draw_perlin()
                elif self.mode_bg_noise == 'gaussian':
                    texture = self.draw_gaussian()
                elif self.mode_bg_noise == 'plain':
                    texture = self.generate_plain()
                elif self.mode_bg_noise == None:
                    texture = np.zeros(self.shape)
                else:
                    raise NotImplementedError
                
                if self.rand_int_scale:
                    texture *= np.random.rand() # introduce scale

                output_img[bg_geom == geom_id] = texture[bg_geom == geom_id]

            assert output_img.max() <= 1 and output_img.min() >= 0
            if self.save_imgs:
                np.save(self.out_dir / (self.img_type + f'_{idx}.npy'), output_img.astype(np.float16))

            # output_imgs.append(output_img)
        return output_img
    
    def generate_spheres(self):
        def is_overlapping(array, center, radius):
            x0, y0, z0 = center
            for x in range(max(0, x0 - radius), min(array.shape[0], x0 + radius + 1)):
                for y in range(max(0, y0 - radius), min(array.shape[1], y0 + radius + 1)):
                    for z in range(max(0, z0 - radius), min(array.shape[2], z0 + radius + 1)):
                        if (x - x0)**2 + (y - y0)**2 + (z - z0)**2 <= radius**2:
                            if array[x, y, z] != 0:
                                return True
            return False
        
        def add_sphere(array, center, radius, value):
            x0, y0, z0 = center
            for x in range(max(0, x0 - radius), min(array.shape[0], x0 + radius + 1)):
                for y in range(max(0, y0 - radius), min(array.shape[1], y0 + radius + 1)):
                    for z in range(max(0, z0 - radius), min(array.shape[2], z0 + radius + 1)):
                        if (x - x0)**2 + (y - y0)**2 + (z - z0)**2 <= radius**2:
                            array[x, y, z] = value

        array = np.zeros(self.shape, dtype=np.int32)
        num_spheres = random.randint(self.num_spheres_min, self.num_spheres_max)

        spheres_added = 0
        attempts = 0
        while spheres_added < num_spheres and attempts < self.max_attempts:
            radius = random.randint(self.rad_min, self.rad_max)
            center = (
                random.randint(radius, array.shape[0] - radius - 1),
                random.randint(radius, array.shape[1] - radius - 1),
                random.randint(radius, array.shape[2] - radius - 1)
            )
            if not is_overlapping(array, center, radius):
                add_sphere(array, center, radius, spheres_added+1)
                spheres_added += 1
            attempts += 1

        return array

    def generate_voronoi_regions(self):
        num_points = random.randint(self.num_voronoi_min, self.num_voronoi_max)
        points = np.random.rand(num_points, 3) * np.array(self.shape)
        
        x = np.arange(self.shape[0])
        y = np.arange(self.shape[1])
        z = np.arange(self.shape[2])
        grid = np.array(list(itertools.product(x, y, z))).reshape(self.shape[0], self.shape[1], self.shape[2], 3)

        flat_grid = grid.reshape(-1, 3)

        tree = cKDTree(points)
        _, regions = tree.query(flat_grid)

        voronoi_grid = regions.reshape(self.shape)
        return voronoi_grid
    
    def generate_worley_noise(self):
        num_points = random.randint(self.num_voronoi_min, self.num_voronoi_max)
        points = np.random.rand(num_points, 3) * np.array(self.shape)
        vor = Voronoi(points)
        
        data = np.zeros(self.shape)
        for i, j, k in itertools.product(range(self.shape[0]), range(self.shape[1]), range(self.shape[2])):
            min_dist = np.min(np.sum((vor.vertices - np.array([i, j, k]))**2, axis=1))
            data[i, j, k] = min_dist
        return (data - data.min()) / (data.max() - data.min())
    
    def draw_gaussian(self):
        std = np.random.uniform(self.gaussian_min_std, self.gaussian_max_std)
        out = np.random.normal(loc=0, scale=std, size=self.shape)
        return (out - out.min()) / (out.max() - out.min())
    
    def generate_plain(self):
        intensity = np.random.uniform(0, 1)
        return np.ones(self.shape) * intensity

    def draw_perlin(self):
        """
        Adapted from: https://github.com/adalca/neurite/blob/9167ad8ad4ef5cb9b22ee8c67d3c42c38eea4bda/neurite/tf/utils/augment.py#L7
        """
        out = np.zeros(self.shape)
        for scale in self.perlin_scales:
            std = np.random.uniform(self.perlin_min_std, self.perlin_max_std)
            gauss = np.random.normal(loc=0, scale=std, size=np.ceil(np.array(self.shape) / scale).astype(int))

            if scale == 1: 
                out += gauss
            else:
                out += F.interpolate(torch.tensor(gauss).squeeze()[None, None], size=[128, 128, 128], mode='trilinear').squeeze().numpy()
        return (out - out.min()) / (out.max() - out.min())


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = BackgroundGen3D(
        mode_bg_geom=None,  # TODO
        mode_bg_noise='perlin', # TODO
        num_imgs=20000,  # TODO
        out_dir=r'path/to/output/dir',  # TODO

        num_voronoi_min=10, num_voronoi_max=30,
        num_spheres_min=10, num_spheres_max=30, rad_min=20, rad_max=50, max_attempts=1000,
        perlin_scales=[2, 4, 8, 16, 32, 64], perlin_min_std=0, perlin_max_std=5,
    )()
    # img = imgs[-1]

    if img:
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for idx in range(16):
            ax = axes[idx // 4, idx % 4]
            ax.imshow(img[:, idx*7, :].squeeze(), cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('./bg_sample.png')
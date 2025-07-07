import numpy as np
import cairo
import cv2
import os
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import argparse

@dataclass
class Color:
    r: float
    g: float
    b: float
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """转换为RGB元组"""
        return (self.r, self.g, self.b)
    
    def to_bgr_tuple(self) -> Tuple[float, float, float]:
        """转换为BGR元组（适用于OpenCV）"""
        return (self.b, self.g, self.r)
    

def create_index_array(shape):
    """
    创建一个三维索引数组，其中：
    - 第一通道表示 x 坐标（第一维下标）
    - 第二通道表示 y 坐标（第二维下标）
    
    参数:
    shape: 元组 (W, H)，指定输出数组的宽度和高度
    
    返回:
    numpy.ndarray: 形状为 [W, H, 2] 的三维数组
    """
    W, H = shape
    x_indices = np.repeat(np.arange(W)[:, np.newaxis], H, axis=1)
    y_indices = np.repeat(np.arange(H)[np.newaxis, :], W, axis=0)
    
    # 合并两个索引数组为一个三维数组
    return np.stack([x_indices, y_indices], axis=2)


def point_to_line_distance(x0, y0, x1, y1, x2, y2):
    # 计算直线的一般式方程 Ax + By + C = 0
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    
    # 计算点到直线的距离
    numerator = abs(A * x0 + B * y0 + C)
    denominator = np.sqrt(A**2 + B**2)
    
    return numerator / denominator 


class ShapeGenerator:
    """形状生成器，用于创建三角形和圆形（归一化坐标）"""
    
    def __init__(self, max_shapes=5, color_variation=0.3):
        """
        初始化形状生成器（坐标归一化至[0,1]）
        
        参数:
            max_shapes: 每个图像最多包含的形状数量
            color_variation: 颜色变化范围
        """
        self.max_shapes = max_shapes
        self.colors = self._generate_colors()
    
    def _generate_colors(self) -> List[Color]:
        """生成随机颜色列表"""
        colors = []
        for _ in range(20):
            r = np.random.uniform(0.2, 0.9)
            g = np.random.uniform(0.2, 0.9)
            b = np.random.uniform(0.2, 0.9)
            colors.append(Color(r, g, b))
        return colors
    
    def generate_triangle(self, min_size=0.1, max_size=0.4) -> Dict:
        """
        生成归一化坐标的随机三角形（尺寸范围0.1-0.4倍图像大小）
        
        参数:
            min_size: 三角形最小归一化尺寸
            max_size: 三角形最大归一化尺寸
            
        返回:
            包含归一化顶点和颜色的字典
        """
        size = np.random.uniform(min_size, max_size)
        center_x = np.random.uniform(0, 1)
        center_y = np.random.uniform(0, 1)
        
        # 生成归一化顶点坐标（0-1范围）
        angle = np.random.uniform(0, 2 * np.pi)
        vertex1 = (
            center_x + size * np.cos(angle),
            center_y + size * np.sin(angle)
        )
        vertex2 = (
            center_x + size * np.cos(angle + 2 * np.pi / 3),
            center_y + size * np.sin(angle + 2 * np.pi / 3)
        )
        vertex3 = (
            center_x + size * np.cos(angle + 4 * np.pi / 3),
            center_y + size * np.sin(angle + 4 * np.pi / 3)
        )
        
        color = np.random.choice(self.colors)
        return {
            'type': 'triangle',
            'vertices': np.array([vertex1, vertex2, vertex3]),  # 归一化顶点
            'color': color,
            'center': np.array([center_x, center_y]),          # 归一化中心
            'size': size                             # 归一化尺寸
        }
    
    def generate_circle(self, min_radius=0.05, max_radius=0.3) -> Dict:
        """
        生成归一化坐标的随机圆形（半径范围0.05-0.3倍图像大小）
        
        参数:
            min_radius: 圆的最小归一化半径
            max_radius: 圆的最大归一化半径
            
        返回:
            包含归一化圆心、半径和颜色的字典
        """
        radius = np.random.uniform(min_radius, max_radius)
        center_x = np.random.uniform(-0.1, 1.1)
        center_y = np.random.uniform(-0.1, 1.1)
        
        color = np.random.choice(self.colors)
        return {
            'type': 'circle',
            'center': np.array([center_x, center_y]),    # 归一化圆心
            'radius': radius,                  # 归一化半径
            'color': color
        }
    
    def generate_random_shape(self) -> Dict:
        """随机生成三角形或圆形（均为归一化坐标）"""
        shape_type = np.random.choice(['triangle', 'circle'], p=[0.6, 0.4])
        if shape_type == 'triangle':
            return self.generate_triangle()
        else:
            return self.generate_circle()


class DatasetGenerator:
    """数据集生成器（支持任意图像尺寸的归一化坐标处理）"""
    
    def __init__(self, image_size=256, num_samples=1000, 
                 max_shapes_per_image=5, noise_level=0.1):
        """
        初始化数据集生成器
        
        参数:
            image_size: 基础图像尺寸（用于坐标转换）
            num_samples: 生成样本数量
            max_shapes_per_image: 每张图像最多形状数量
            noise_level: 噪声水平
        """
        self.image_size = image_size
        self.num_samples = num_samples
        self.max_shapes_per_image = max_shapes_per_image
        self.noise_level = noise_level
        self.shape_generator = ShapeGenerator(max_shapes_per_image)

        self.surface = None
        self.ctx = None

        # 常用的变量但是不能立即初始化
        self._points = None
    
    def generate_image_with_shapes(self, num_shapes=None, target_size=None) -> Tuple[np.ndarray, Dict]:
        """
        生成包含指定数量形状的图像及其标签（支持任意目标尺寸）
        
        参数:
            num_shapes: 图像中形状的数量，None表示随机数量
            target_size: 目标图像尺寸，None使用初始化尺寸
            
        返回:
            图像数组和标签字典（标签包含归一化参数）
        """
        # 确定目标尺寸
        target_size = target_size or self.image_size
        
        if num_shapes is None:
            num_shapes = np.random.randint(1, self.max_shapes_per_image + 1)
        
        # 生成形状（归一化坐标）
        shapes = []
        for _ in range(num_shapes):
            shape = self.shape_generator.generate_random_shape()
            shapes.append(shape)
        
        # 初始化图像和标签
        image = np.zeros((target_size, target_size, 3), dtype=np.float32)
        # 对角线长为 size*sqrt(2), 这里添加 *2保证初始值大于最大可能值
        distance_field = np.ones((target_size, target_size), dtype=np.float32) * target_size * 2 
        association_map = np.empty((target_size, target_size), dtype=object)
        for idx in np.ndindex(association_map.shape):
            association_map[idx] = {'dir':[]}
        
        self._points = create_index_array((target_size, target_size))
        # 绘制形状并计算标签（归一化坐标转换为像素坐标）
        for shape in shapes:
            if shape['type'] == 'triangle':
                self._draw_triangle(shape, image, distance_field, association_map, target_size)
            else:
                self._draw_circle(shape, image, distance_field, association_map, target_size)

        u , theta, omega = self._gen_g(association_map, target_size)
        u = u / target_size
        distance_field = distance_field / target_size
        # 添加噪声，一般不进行，在训练时动态添加
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, (target_size, target_size, 3))
            image = np.clip(image + noise, 0, 1)
        
        # g = (u, cos\theta, sin\theta, \omegea)
        labels = np.stack([u[...,0], u[...,1], np.cos(theta), np.sin(theta), omega[...,0], omega[...,1], omega[...,2]], axis=2)
        labels = labels.astype(float)
        # from [W, H, C] to [C, W, H]
        labels = np.transpose(labels, (2, 0, 1))

        return image, labels
    
    def generate_image_with_shapes_simple(self, num_shapes=None, target_size=None) -> Tuple[np.ndarray, Dict]:
        target_size = target_size or self.image_size
        
        if num_shapes is None:
            num_shapes = np.random.randint(1, self.max_shapes_per_image + 1)
        
        # 生成形状（归一化坐标）
        shapes = []
        for _ in range(num_shapes):
            shape = self.shape_generator.generate_random_shape()
            shapes.append(shape)
        
        # 对角线长为 size*sqrt(2), 这里添加 *2保证初始值大于最大可能值
        distance_field = np.ones((target_size, target_size), dtype=np.float32) * target_size * 2 
        
        self._points = create_index_array((target_size, target_size))
        # 绘制形状并计算标签（归一化坐标转换为像素坐标）
        for shape in shapes:
            if shape['type'] == 'triangle':
                self._draw_triangle_simple(shape, distance_field, target_size)
            else:
                self._draw_circle_simple(shape, distance_field, target_size)

        distance_field = distance_field / target_size
        # 添加噪声，一般不进行，在训练时动态添加
        
        labels = distance_field
        # from [W, H, C] to [C, W, H]
        # labels = np.transpose(labels, (2, 0, 1))

        return labels
    
    def _draw_triangle(self, shape, distance_field, association_map, size):
        """绘制三角形（归一化坐标转像素坐标）"""
        # 归一化坐标转像素坐标
        vertices = shape['vertices'] * size
        color = shape['color']
        
        self.ctx.set_source_rgb(*color.to_tuple())
        self.ctx.move_to(*vertices[0])
        self.ctx.line_to(*vertices[1])
        self.ctx.line_to(*vertices[2])
        self.ctx.close_path()
        self.ctx.fill()
        
        # 计算距离场和边界图（使用像素坐标）
        distance, edge, projection = self._distance_to_triangle_boundary(self._points, vertices)

        mask_inside = self._point_in_triangle(self._points, vertices)
        changed_mask = distance <= distance_field
        mask = mask_inside | changed_mask
        distance_field[mask] = distance[mask]
        def __update_association(d, v, p, inside):
            if v < 3: # 边
                (x1, y1), (x2, y2) = (vertices[v], vertices[(v+1)%3])
            else: # 顶点
                v = (v + 1) % 3
                (x1, y1), (x2, y2) = (vertices[v], vertices[(v+1)%3])
            x0, y0 = p
            if inside:
                d['dir'].clear()
            d['dir'].append(np.array([x1-x0,y1-y0]))
            d['dir'].append(np.array([x2-x0,y2-y0]))
            d['projection'] = p

        np.vectorize(__update_association, signature='(),(),(2),()->()')(association_map[mask], edge[mask], projection[mask], mask_inside[mask])
       
    def _draw_circle(self, shape, distance_field, association_map, size):
        """绘制圆形（归一化坐标转像素坐标）"""
        # 归一化坐标转像素坐标
        center = (int(round(shape['center'][0] * size)), int(round(shape['center'][1] * size)))
        radius = int(round(shape['radius'] * size))
        color = shape['color']
        
        self.ctx.set_source_rgb(*color.to_tuple())
        self.ctx.arc(center[0], center[1], radius, 0, 2 * np.pi)
        self.ctx.fill()
        
        # 计算距离场和边界图（使用像素坐标）
        center = shape['center'] * size
        radius = shape['radius'] * size

        d = self._points - center
        distance = np.linalg.norm(d, axis=2)
        n = d / distance[:, :, np.newaxis]
        projection = n * radius + center

        distance = distance - radius
        mask_inside = distance < 0
        distance = np.abs(distance)
        
        change_mask = distance <= distance_field
        mask = mask_inside | change_mask
        distance_field[mask] = distance[mask]
        def __update_association(d, p, n, inside):
            dx, dy = n
            if inside:
                d['dir'].clear()
            d['dir'].append(np.array([dy, -dx]))
            d['dir'].append(np.array([-dy, dx]))
            d['projection'] = p

        np.vectorize(__update_association, signature='(),(2),(2),()->()')(association_map[mask], projection[mask], n[mask], mask_inside[mask])
    
    def _draw_triangle_simple(self, shape, distance_field, size):
        """绘制三角形（归一化坐标转像素坐标）"""
        # 归一化坐标转像素坐标
        vertices = shape['vertices'] * size
        color = shape['color']
        
        self.ctx.set_source_rgb(*color.to_tuple())
        self.ctx.move_to(*vertices[0])
        self.ctx.line_to(*vertices[1])
        self.ctx.line_to(*vertices[2])
        self.ctx.close_path()
        self.ctx.fill()

        def simple_bound(points, triangle):
            A, B, C = triangle
            AB = B - A
            BC = C - B
            CA = A - C
            
            def distance_projection_and_type(p, seg_start, seg_vec):
                """计算点p到线段的距离、投影点及投影类型"""
                seg_len_sq = np.sum(seg_vec**2, axis=-1)
                
                # 处理零长度线段的情况（退化为顶点）
                if np.isscalar(seg_len_sq):
                    if seg_len_sq == 0:
                        dist = np.linalg.norm(p - seg_start, axis=-1)
                        return dist
                
                t = np.sum((p - seg_start) * seg_vec, axis=-1) / seg_len_sq
                
                t_clamped = np.clip(t, 0, 1)
                
                projection = seg_start + t_clamped[..., np.newaxis] * seg_vec
                
                distance = np.linalg.norm(p - projection, axis=-1)
                
                return distance
            
            # 计算点到三条边的距离、投影点及类型
            d_ab = distance_projection_and_type(points, A, AB)
            d_bc = distance_projection_and_type(points, B, BC)
            d_ca = distance_projection_and_type(points, C, CA)
            
            # 创建距离数组和索引数组
            distances = np.stack([d_ab, d_bc, d_ca], axis=-1)
            min_distances = np.min(distances, axis=-1)
            
            return min_distances
        
        # 计算距离场和边界图（使用像素坐标）
        distance = simple_bound(self._points, vertices)

        mask_inside = self._point_in_triangle(self._points, vertices)
        changed_mask = distance <= distance_field
        mask = mask_inside | changed_mask
        distance_field[mask] = distance[mask]
       
    def _draw_circle_simple(self, shape, distance_field, size):
        """绘制圆形（归一化坐标转像素坐标）"""
        # 归一化坐标转像素坐标
        center = (int(round(shape['center'][0] * size)), int(round(shape['center'][1] * size)))
        radius = int(round(shape['radius'] * size))
        color = shape['color']
        
        self.ctx.set_source_rgb(*color.to_tuple())
        self.ctx.arc(center[0], center[1], radius, 0, 2 * np.pi)
        self.ctx.fill()
        
        # 计算距离场和边界图（使用像素坐标）
        center = shape['center'] * size
        radius = shape['radius'] * size

        d = self._points - center
        distance = np.linalg.norm(d, axis=2)

        distance = distance - radius
        mask_inside = distance < 0
        distance = np.abs(distance)
        
        change_mask = distance <= distance_field
        mask = mask_inside | change_mask
        distance_field[mask] = distance[mask]

    def _point_in_triangle(self, points, triangle) -> np.ndarray:
        """判断点是否在三角形内部（像素坐标版）"""
        w, h, _ = points.shape
        points = points.reshape((w * h, 2))
        v0, v1, v2 = triangle

        area = abs((v0[0]*(v1[1]-v2[1]) + v1[0]*(v2[1]-v0[1]) + v2[0]*(v0[1]-v1[1])))
        
        def sub_area(p):
            return abs((v0[0]*(v1[1]-p[1]) + v1[0]*(p[1]-v0[1]) + p[0]*(v0[1]-v1[1])))
    
        # 计算三个子面积
        s1 = np.apply_along_axis(lambda p: sub_area(p), 1, points)
        s2 = np.apply_along_axis(lambda p: abs((v0[0]*(p[1]-v2[1]) + p[0]*(v2[1]-v0[1]) + v2[0]*(v0[1]-p[1]))), 1, points)
        s3 = np.apply_along_axis(lambda p: abs((p[0]*(v1[1]-v2[1]) + v1[0]*(v2[1]-p[1]) + v2[0]*(p[1]-v1[1]))), 1, points)
        
        # 判断点是否在三角形内（考虑浮点数误差）
        is_inside = np.isclose(s1 + s2 + s3, area, atol=1e-8) | ((s1 < 1e-8) & (s2 < 1e-8) | (s1 < 1e-8) & (s3 < 1e-8) | (s2 < 1e-8) & (s3 < 1e-8))
        
        # 重塑为图像形状
        mask = is_inside.reshape(w, h).astype(bool)
        return mask
    
    def _distance_to_triangle_boundary(self, points, triangle) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算点到三角形边界的最短距离（像素坐标版）"""
        A, B, C = triangle
    
        # 计算三角形三边的向量
        AB = B - A
        BC = C - B
        CA = A - C
        
        # 计算点到各边的距离及投影点
        def distance_projection_and_type(p, seg_start, seg_vec, edge_idx, vertex1, vertex2):
            """计算点p到线段的距离、投影点及投影类型"""
            seg_len_sq = np.sum(seg_vec**2, axis=-1)
            
            # 处理零长度线段的情况（退化为顶点）
            if np.isscalar(seg_len_sq):
                if seg_len_sq == 0:
                    dist = np.linalg.norm(p - seg_start, axis=-1)
                    return dist, seg_start, np.full_like(dist, vertex1, dtype=np.uint8)
            
            # 计算投影比例t
            t = np.sum((p - seg_start) * seg_vec, axis=-1) / seg_len_sq
            
            # 限制t在[0,1]范围内
            t_clamped = np.clip(t, 0, 1)
            
            # 计算投影点
            projection = seg_start + t_clamped[..., np.newaxis] * seg_vec
            
            # 计算距离
            distance = np.linalg.norm(p - projection, axis=-1)
            
            # 判断投影点类型：0-2为边，3-5为顶点
            is_vertex1 = np.isclose(t_clamped, 0)  # 投影在顶点1
            is_vertex2 = np.isclose(t_clamped, 1)  # 投影在顶点2
            # is_edge = ~is_vertex1 & ~is_vertex2    # 投影在边上
            
            # 初始化类型数组
            proj_type = np.full_like(distance, edge_idx, dtype=np.uint8)
            proj_type[is_vertex1] = vertex1
            proj_type[is_vertex2] = vertex2
            
            return distance, projection, proj_type
        
        # 计算点到三条边的距离、投影点及类型
        d_ab, proj_ab, type_ab = distance_projection_and_type(points, A, AB, 0, 3, 4)
        d_bc, proj_bc, type_bc = distance_projection_and_type(points, B, BC, 1, 4, 5)
        d_ca, proj_ca, type_ca = distance_projection_and_type(points, C, CA, 2, 5, 3)
        
        # 创建距离数组和索引数组
        distances = np.stack([d_ab, d_bc, d_ca], axis=-1)
        min_distances = np.min(distances, axis=-1)

        min_indices = np.argmin(distances, axis=-1)
        all_projections = np.stack([proj_ab, proj_bc, proj_ca], axis=-2)
        a, b = min_indices.shape
        i_indices, j_indices = np.ogrid[:a, :b]
        min_projections = all_projections[i_indices, j_indices, min_indices, :]

        all_type = np.stack([type_ab, type_bc, type_ca], axis=-1)
        min_type = all_type[i_indices, j_indices, min_indices]
        
        return min_distances, min_type, min_projections

    def _gen_g(self, association_map, size) -> np.ndarray:
        """计算距离场的梯度（像素坐标版）"""
        def __update_u(assoc, point):
            projection = assoc['projection']
            return projection - point
            
        u = np.vectorize(__update_u, signature='(),(2)->(n)')(association_map, self._points)

        def __update_theta(assoc):
            dir = assoc['dir'][0]
            return np.arctan2(dir[0], dir[1])
        theta = np.vectorize(__update_theta)(association_map)

        def __update_theta_omega(assoc):
            dirs = assoc['dir']
            theta = np.arctan2(dirs[0][1], dirs[0][0])
            l = len(dirs)
            if l > 2:
                dirs = dirs[:3]
                theta2 = np.arctan2(dirs[1][1], dirs[1][0])
                theta3 = np.arctan2(dirs[2][1], dirs[2][0])
                omega = np.array([theta2 - theta, theta3 - theta2, theta - theta3])
            else:
                omega = np.array([0.5, 0.5, 0])

            return theta, omega
        
        theta, omega = np.vectorize(__update_theta_omega, otypes=(float, np.ndarray),
                                    signature='()->(),(3)')(association_map)
        omega = np.abs(omega) / (np.pi * 2)
        return u, theta, omega
    
    def generate_dataset(self, output_dir='synthetic_dataset', target_size=None, offset=0):
        """
        生成完整数据集并保存（支持指定目标尺寸）
        
        参数:
            output_dir: 输出目录
            target_size: 目标图像尺寸，None使用初始化尺寸
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
        
        target_size = target_size or self.image_size
        print(f"开始生成{self.num_samples}个样本（尺寸{target_size}x{target_size}）...")
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, target_size, target_size)
        self.ctx = cairo.Context(self.surface)
        self.ctx.set_antialias(cairo.Antialias.BEST)
        
        for i in tqdm(range(self.num_samples)):
            self.ctx.set_source_rgb(1, 1, 1)
            self.ctx.paint()

            num_shapes = np.random.randint(max(1, self.max_shapes_per_image // 2), self.max_shapes_per_image + 1)
            # 生成图像和标签（含归一化参数）
            # image, labels = self.generate_image_with_shapes(num_shapes, target_size)
            labels = self.generate_image_with_shapes_simple(num_shapes, target_size)
            
            # 保存图像（BGR转RGB）
            img_path = os.path.join(output_dir, 'images', f'{i+offset:05d}.png')
            self.surface.write_to_png(img_path)
            
            # 保存标签（包含归一化参数，与图像尺寸无关）
            label_path = os.path.join(output_dir, 'labels', f'{i+offset:05d}.npz')
            np.savez(label_path, label=labels)
        
        print(f"数据集已生成并保存到{output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=21)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--max_shapes_per_image', type=int, default=5)
    parser.add_argument('--noise_level', type=float, default=0.00)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--offset', type=int, default=0)
    args = parser.parse_args()

    dataset_gen = DatasetGenerator(
        image_size=args.size,
        num_samples=args.num_samples,
        max_shapes_per_image=args.max_shapes_per_image,
        noise_level=args.noise_level,
    )
    
    output_dir = args.output_dir if args.output_dir else f'./data/{args.size}pix'
    dataset_gen.generate_dataset(target_size=args.size, output_dir=output_dir, offset=args.offset)


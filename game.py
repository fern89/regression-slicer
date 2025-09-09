import pygame
import random
import math

# --- Constants ---
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 50, 50)
BLUE = (50, 100, 220)
GREEN = (50, 220, 100)
PURPLE = (150, 50, 220)
GREY = (150, 150, 150)
DARK_GREY = (40, 40, 40)
YELLOW = (255, 220, 50)

# Game settings
OBJECT_RADIUS_MIN = 30
OBJECT_RADIUS_MAX = 50
INITIAL_OBJECT_SPEED = 150
INITIAL_SPAWN_RATE_MS = 1500
MIN_SPAWN_RATE_MS = 400
MAX_MISSES = 5
BONUS_SCORE_MULTIPLIER = 500

# Time-based difficulty constants
SPEED_INCREASE_PER_SECOND = 2.5
SPAWN_RATE_DECREASE_PER_SECOND = 20

# --- Helper Functions ---
def line_segment_circle_collision(line_start, line_end, circle_center, circle_radius):
    p1, p2, c = pygame.math.Vector2(line_start), pygame.math.Vector2(line_end), pygame.math.Vector2(circle_center)
    line_vec = p2 - p1
    if line_vec.length_squared() == 0: return p1.distance_to(c) < circle_radius
    t = max(0, min(1, (c - p1).dot(line_vec) / line_vec.length_squared()))
    return (p1 + t * line_vec).distance_to(c) < circle_radius

def calculate_fit_score(points, line_start, line_end):
    n = len(points)
    if n < 2: return 0.0
    centroid_x = sum(p[0] for p in points) / n
    centroid_y = sum(p[1] for p in points) / n
    total_spread = sum(math.hypot(p[0] - centroid_x, p[1] - centroid_y)**2 for p in points)
    if total_spread == 0: return 1.0
    x1, y1 = line_start; x2, y2 = line_end
    A, B, C = y1 - y2, x2 - x1, x1 * y2 - x2 * y1
    line_norm = math.sqrt(A**2 + B**2)
    if line_norm == 0: return 0.0
    sum_sq_dist_from_line = sum((abs(A * px + B * py + C) / line_norm)**2 for px, py in points)
    fit_score = 1 - (sum_sq_dist_from_line / total_spread)
    return fit_score

def point_in_polygon(x, y, polygon_vertices):
    n = len(polygon_vertices)
    inside = False
    p1x, p1y = polygon_vertices[0]
    for i in range(n + 1):
        p2x, p2y = polygon_vertices[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y: xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters: inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def generate_graph_data(player_average_r2, width, height, num_points=50):
    noise_factor = (1.0 - max(0, player_average_r2)) * (height / 3)
    m, c = random.uniform(-0.8, 0.8), random.uniform(-height / 8, height / 8)
    line_start, line_end = (0, m * (-width/2) + c), (width, m * (width/2) + c)
    points = [(random.uniform(0, width), (m * (px - width/2) + c) + random.uniform(-noise_factor, noise_factor)) for px in [random.uniform(0, width) for _ in range(num_points)]]
    return {"points": points, "line_start": line_start, "line_end": line_end}

def draw_end_graph(surface, graph_data, average_r2, pos_rect, fonts):
    pygame.draw.rect(surface, DARK_GREY, pos_rect)
    pygame.draw.rect(surface, GREY, pos_rect, 2)
    title_surf = fonts['small'].render("Your Performance Graph", True, WHITE)
    title_rect = title_surf.get_rect(midtop=(pos_rect.centerx, pos_rect.top + 10))
    surface.blit(title_surf, title_rect)
    r2_color = GREEN if average_r2 > 0.8 else (YELLOW if average_r2 > 0.5 else RED)
    r2_text = f"Average R²: {average_r2:.3f}"
    r2_surf = fonts['medium'].render(r2_text, True, r2_color)
    r2_rect = r2_surf.get_rect(midbottom=(pos_rect.centerx, pos_rect.bottom - 10))
    surface.blit(r2_surf, r2_rect)
    graph_area_rect = pos_rect.inflate(-40, -80)
    start_unclipped = (graph_area_rect.left + graph_data['line_start'][0], graph_area_rect.centery + graph_data['line_start'][1])
    end_unclipped = (graph_area_rect.left + graph_data['line_end'][0], graph_area_rect.centery + graph_data['line_end'][1])
    clipped_line = graph_area_rect.clipline(start_unclipped, end_unclipped)
    if clipped_line:
        start_clipped, end_clipped = clipped_line
        pygame.draw.line(surface, BLUE, start_clipped, end_clipped, 2)
    for p in graph_data['points']:
        px_screen, py_screen = graph_area_rect.left + p[0], graph_area_rect.centery + p[1]
        if graph_area_rect.collidepoint(px_screen, py_screen):
            pygame.draw.circle(surface, WHITE, (int(px_screen), int(py_screen)), 2)

# --- Game Classes ---
class Debris:
    def __init__(self, surface, pos, velocity):
        self.surface = surface
        self.pos = pygame.math.Vector2(pos)
        self.velocity = pygame.math.Vector2(velocity)
        self.angle = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-200, 200)
        self.initial_lifetime = 0.8
        self.lifetime = self.initial_lifetime

    def update(self, dt):
        self.pos += self.velocity * dt
        self.angle += self.rotation_speed * dt
        self.lifetime -= dt
        self.velocity.y += 200 * dt

    def draw(self, surface):
        if self.lifetime > 0:
            rotated_surf = pygame.transform.rotozoom(self.surface, self.angle, 1)
            alpha = max(0, 255 * (self.lifetime / self.initial_lifetime))
            rotated_surf.set_alpha(alpha)
            draw_rect = rotated_surf.get_rect(center=self.pos)
            surface.blit(rotated_surf, draw_rect)

class FallingObject:
    def __init__(self):
        self.x = random.randint(self.avg_radius, SCREEN_WIDTH - self.avg_radius)
        self.y = -self.avg_radius
        self.color = random.choice([BLUE, RED, GREEN, PURPLE])
        self.internal_points = []
        self._generate_internal_points()
    def update(self, dt, current_speed): self.y += current_speed * dt
    def _generate_internal_points(self, num_points=12, noise_factor=0.4): raise NotImplementedError
    def draw(self, surface, offset=(0,0)): raise NotImplementedError
    def is_sliced(self, line_start, line_end): raise NotImplementedError

    def create_debris(self, slice_start, slice_end):
        size = int(self.avg_radius * 2.6)
        snapshot_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        self.draw(snapshot_surf, offset=(-self.x + size/2, -self.y + size/2))
        slice_vec = pygame.math.Vector2(slice_end) - pygame.math.Vector2(slice_start)
        if slice_vec.length() == 0: slice_vec = pygame.math.Vector2(0, -1)
        perp_vec = slice_vec.rotate(90).normalize()
        debris_speed = random.uniform(150, 250)
        x1, y1 = slice_start[0]-self.x+size/2, slice_start[1]-self.y+size/2
        x2, y2 = slice_end[0]-self.x+size/2, slice_end[1]-self.y+size/2
        A, B, C = y1-y2, x2-x1, x1*y2-x2*y1
        debris1_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        debris2_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        for x in range(size):
            for y in range(size):
                color = snapshot_surf.get_at((x, y))
                if color.a > 0:
                    if A*x + B*y + C > 0: debris1_surf.set_at((x, y), color)
                    else: debris2_surf.set_at((x, y), color)
        piece1 = Debris(debris1_surf, (self.x, self.y), perp_vec * debris_speed)
        piece2 = Debris(debris2_surf, (self.x, self.y), -perp_vec * debris_speed)
        return [piece1, piece2]

class CircleObject(FallingObject):
    def __init__(self):
        self.avg_radius = random.randint(OBJECT_RADIUS_MIN, OBJECT_RADIUS_MAX)
        self.radius = self.avg_radius
        super().__init__()
    def _generate_internal_points(self, num_points=12, noise_factor=0.4):
        true_m, true_c = random.uniform(-1.5, 1.5), random.uniform(-self.radius*0.2, self.radius*0.2)
        while len(self.internal_points) < num_points:
            px = random.uniform(-self.radius*0.8, self.radius*0.8)
            py_noised = (true_m*px+true_c)+random.uniform(-self.radius*noise_factor, self.radius*noise_factor)
            if math.hypot(px, py_noised) < self.radius*0.9: self.internal_points.append((px, py_noised))
    def draw(self, surface, offset=(0,0)):
        center_pos = (int(self.x+offset[0]), int(self.y+offset[1]))
        pygame.draw.circle(surface, self.color, center_pos, self.radius)
        pygame.draw.circle(surface, WHITE, center_pos, self.radius, 2)
        for px, py in self.internal_points: pygame.draw.circle(surface, WHITE, (int(self.x+px+offset[0]), int(self.y+py+offset[1])), 2)
    def is_sliced(self, line_start, line_end): return line_segment_circle_collision(line_start, line_end, (self.x, self.y), self.radius)

class SquareObject(FallingObject):
    def __init__(self):
        self.avg_radius = random.randint(OBJECT_RADIUS_MIN, OBJECT_RADIUS_MAX)
        self.size = self.avg_radius * 1.8
        super().__init__()
        self.rect = pygame.Rect(self.x-self.size/2, self.y-self.size/2, self.size, self.size)
    def _generate_internal_points(self, num_points=12, noise_factor=0.4):
        bound = self.size/2*0.9
        true_m, true_c = random.uniform(-1.5, 1.5), random.uniform(-bound*0.2, bound*0.2)
        for _ in range(num_points):
            px, py = random.uniform(-bound, bound), (true_m*random.uniform(-bound, bound)+true_c)+random.uniform(-bound*noise_factor, bound*noise_factor)
            if abs(py) < bound: self.internal_points.append((px, py))
    def update(self, dt, current_speed):
        super().update(dt, current_speed)
        self.rect.center = (self.x, self.y)
    def draw(self, surface, offset=(0,0)):
        draw_rect = self.rect.copy()
        draw_rect.center = (self.x+offset[0], self.y+offset[1])
        pygame.draw.rect(surface, self.color, draw_rect)
        pygame.draw.rect(surface, WHITE, draw_rect, 2)
        for px, py in self.internal_points: pygame.draw.circle(surface, WHITE, (int(self.x+px+offset[0]), int(self.y+py+offset[1])), 2)
    def is_sliced(self, line_start, line_end): return self.rect.clipline(line_start, line_end)

class IrregularObject(FallingObject):
    def __init__(self):
        self.avg_radius = random.randint(OBJECT_RADIUS_MIN, OBJECT_RADIUS_MAX)
        self.relative_vertices = self._generate_polygon_shape()
        super().__init__()
        self.screen_vertices = [(self.x+v[0], self.y+v[1]) for v in self.relative_vertices]
    def _generate_polygon_shape(self, num_verts=7):
        angles = sorted([random.uniform(0, 2*math.pi) for _ in range(num_verts)])
        return [(random.uniform(self.avg_radius*0.9, self.avg_radius*1.3)*math.cos(a), random.uniform(self.avg_radius*0.9, self.avg_radius*1.3)*math.sin(a)) for a in angles]
    def _generate_internal_points(self, num_points=12, noise_factor=0.4):
        bound = self.avg_radius
        true_m, true_c = random.uniform(-1.5, 1.5), 0
        while len(self.internal_points) < num_points:
            px, py = random.uniform(-bound, bound), (true_m*random.uniform(-bound, bound)+true_c)+random.uniform(-bound*noise_factor, bound*noise_factor)
            if point_in_polygon(px, py, self.relative_vertices): self.internal_points.append((px, py))
    def update(self, dt, current_speed):
        super().update(dt, current_speed)
        self.screen_vertices = [(self.x+v[0], self.y+v[1]) for v in self.relative_vertices]
    def draw(self, surface, offset=(0,0)):
        draw_vertices = [(v[0]+offset[0], v[1]+offset[1]) for v in self.screen_vertices]
        pygame.draw.polygon(surface, self.color, draw_vertices)
        pygame.draw.polygon(surface, WHITE, draw_vertices, 2)
        for px, py in self.internal_points: pygame.draw.circle(surface, WHITE, (int(self.x+px+offset[0]), int(self.y+py+offset[1])), 2)
    def is_sliced(self, line_start, line_end): return line_segment_circle_collision(line_start, line_end, (self.x, self.y), self.avg_radius*1.3)

class ScorePopup:
    def __init__(self, x, y, score, fit_score_value=None, is_bonus=False):
        self.x, self.y, self.lifetime = x, y, 1.0
        if is_bonus:
            self.score_text, self.color, self.font = f"COMBO! +{score}", YELLOW, pygame.font.Font(None, 36)
        else:
            self.score_text = f"{score:+}"
            self.font = pygame.font.Font(None, 28)
            if score < 0: self.color = RED
            elif fit_score_value > 0.9: self.color, self.score_text = GREEN, self.score_text + " Perfect!"
            elif fit_score_value > 0.8: self.color = WHITE
            else: self.color = GREY
    def update(self, dt): self.lifetime -= dt; self.y -= 50*dt
    def draw(self, surface):
        text_surf = self.font.render(self.score_text, True, self.color)
        text_rect = text_surf.get_rect(center=(self.x, self.y))
        surface.blit(text_surf, text_rect)

def title_screen(screen, fonts):
    clock = pygame.time.Clock()
    title_surf = fonts['large'].render("Regression Slicer", True, GREEN)
    title_rect = title_surf.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT*0.3))
    instructions1_surf = fonts['medium'].render("Slice the falling data orbs as accurately as possible.", True, WHITE)
    instructions1_rect = instructions1_surf.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT*0.5))
    instructions2_surf = fonts['small'].render("More accurate slices and combos grant more points!", True, GREY)
    instructions2_rect = instructions2_surf.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT*0.58))
    start_text_surf = fonts['medium'].render("Press any key to begin", True, YELLOW)
    start_text_rect = start_text_surf.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT*0.8))
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return False
                return True
        screen.fill(BLACK)
        screen.blit(title_surf, title_rect)
        screen.blit(instructions1_surf, instructions1_rect)
        screen.blit(instructions2_surf, instructions2_rect)
        alpha = (math.sin(pygame.time.get_ticks()*0.002) + 1)/2 * 255
        start_text_surf.set_alpha(alpha)
        screen.blit(start_text_surf, start_text_rect)
        pygame.display.flip()
        clock.tick(FPS)

def game(screen, fonts):
    pygame.display.set_caption("Regression Slicer")
    clock = pygame.time.Clock()
    score, misses, game_over = 0, 0, False
    spawn_timer = INITIAL_SPAWN_RATE_MS / 1000.0
    falling_objects, popups, debris_pieces = [], [], []
    is_slicing, slice_start, slice_end = False, (0,0), (0,0)
    fit_scores_history, final_graph_data = [], None
    shape_classes = [CircleObject, SquareObject, IrregularObject]
    
    running = True
    start_time = pygame.time.get_ticks()
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return # Exit the game function and return to the main menu
            if game_over:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r: game(screen, fonts); return
            else:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: is_slicing, slice_start, slice_end = True, event.pos, event.pos
                if event.type == pygame.MOUSEMOTION and is_slicing: slice_end = event.pos
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and is_slicing:
                    is_slicing = False
                    sliced_this_turn, last_sliced_pos = 0, None
                    for obj in reversed(falling_objects):
                        if obj.is_sliced(slice_start, slice_end):
                            ls_rel, le_rel = (slice_start[0]-obj.x, slice_start[1]-obj.y), (slice_end[0]-obj.x, slice_end[1]-obj.y)
                            fit = calculate_fit_score(obj.internal_points, ls_rel, le_rel)
                            fit_scores_history.append(fit)
                            original_fit = fit
                            fit = max(fit, -1)
                            if original_fit > 0.9: fit += 1
                            points = int(fit * 1000)
                            score += points
                            popups.append(ScorePopup(obj.x, obj.y, points, fit_score_value=original_fit))
                            debris_pieces.extend(obj.create_debris(slice_start, slice_end))
                            falling_objects.remove(obj)
                            sliced_this_turn += 1
                            last_sliced_pos = (obj.x, obj.y)
                    if sliced_this_turn > 1:
                        bonus = BONUS_SCORE_MULTIPLIER * (sliced_this_turn - 1)
                        score += bonus
                        if last_sliced_pos: popups.append(ScorePopup(last_sliced_pos[0], last_sliced_pos[1], bonus, is_bonus=True))

        if not game_over:
            elapsed_time_seconds = (pygame.time.get_ticks() - start_time) / 1000.0
            current_speed = INITIAL_OBJECT_SPEED + (elapsed_time_seconds * SPEED_INCREASE_PER_SECOND)
            spawn_timer -= dt
            if spawn_timer <= 0:
                falling_objects.append(random.choice(shape_classes)())
                next_spawn_delay_ms = max(MIN_SPAWN_RATE_MS, INITIAL_SPAWN_RATE_MS - (elapsed_time_seconds * SPAWN_RATE_DECREASE_PER_SECOND))
                spawn_timer = next_spawn_delay_ms / 1000.0
            for obj in falling_objects: obj.update(dt, current_speed)
            for piece in reversed(debris_pieces):
                piece.update(dt)
                if piece.lifetime <= 0: debris_pieces.remove(piece)
            for popup in reversed(popups):
                popup.update(dt)
                if popup.lifetime <= 0: popups.remove(popup)
            for obj in reversed(falling_objects):
                if obj.y - obj.avg_radius > SCREEN_HEIGHT: falling_objects.remove(obj); misses += 1
            if misses >= MAX_MISSES: game_over = True

        screen.fill(BLACK)
        for obj in falling_objects: obj.draw(screen)
        for piece in debris_pieces: piece.draw(screen)
        if is_slicing: pygame.draw.line(screen, WHITE, slice_start, slice_end, 3)
        for popup in popups: popup.draw(screen)
        score_text = fonts['medium'].render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        misses_text = fonts['medium'].render(f"Misses: {misses}/{MAX_MISSES}", True, RED)
        misses_rect = misses_text.get_rect(right=SCREEN_WIDTH - 10, top=10)
        screen.blit(misses_text, misses_rect)
        
        if game_over:
            if final_graph_data is None:
                average_r2 = sum(fit_scores_history) / len(fit_scores_history) if fit_scores_history else 0.0
                graph_rect = pygame.Rect(0, 0, 400, 250)
                graph_rect.center = (SCREEN_WIDTH/2, SCREEN_HEIGHT/2 + 80)
                final_graph_data = {"data": generate_graph_data(average_r2, graph_rect.width-40, graph_rect.height-80), "rect": graph_rect, "avg_r2": average_r2}
            
            go_text_surf = fonts['large'].render("GAME OVER", True, RED)
            go_text_rect = go_text_surf.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT*0.20))
            screen.blit(go_text_surf, go_text_rect)
            final_score_surf = fonts['large'].render(f"Final Score: {score}", True, GREEN)
            final_score_rect = final_score_surf.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT*0.35))
            screen.blit(final_score_surf, final_score_rect)
            draw_end_graph(screen, final_graph_data['data'], final_graph_data['avg_r2'], final_graph_data['rect'], fonts)
            restart_text = fonts['medium'].render("Press 'R' to Restart", True, WHITE)
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT-50))
            screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()
    
    # Don't quit pygame here, let the main loop handle it.

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()
    fonts = {'large': pygame.font.Font(None, 74), 'medium': pygame.font.Font(None, 36), 'small': pygame.font.Font(None, 28)}

    while True:
        should_start_game = title_screen(screen, fonts)
        if should_start_game:
            game(screen, fonts)
        else:
            break
            
    pygame.quit()
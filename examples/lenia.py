import cv2
import numpy as np
import scipy.signal as signal
import guibbon as gbn

WIDTH = 100
HEIGHT = 100

def add_orbitum(grid, x, y, flip_x=False, flip_y=False):
    orbitum = np.array([
        [0, 0, 0, 0, 0, 0, 0.1, 0.14, 0.1, 0, 0, 0.03, 0.03, 0, 0, 0.3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0.08, 0.24, 0.3, 0.3, 0.18, 0.14, 0.15, 0.16, 0.15, 0.09, 0.2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0.15, 0.34, 0.44, 0.46, 0.38, 0.18, 0.14, 0.11, 0.13, 0.19, 0.18, 0.45, 0, 0, 0],
        [0, 0, 0, 0, 0.06, 0.13, 0.39, 0.5, 0.5, 0.37, 0.06, 0, 0, 0, 0.02, 0.16, 0.68, 0, 0, 0],
        [0, 0, 0, 0.11, 0.17, 0.17, 0.33, 0.4, 0.38, 0.28, 0.14, 0, 0, 0, 0, 0, 0.18, 0.42, 0, 0],
        [0, 0, 0.09, 0.18, 0.13, 0.06, 0.08, 0.26, 0.32, 0.32, 0.27, 0, 0, 0, 0, 0, 0, 0.82, 0, 0],
        [0.27, 0, 0.16, 0.12, 0, 0, 0, 0.25, 0.38, 0.44, 0.45, 0.34, 0, 0, 0, 0, 0, 0.22, 0.17, 0],
        [0, 0.07, 0.2, 0.02, 0, 0, 0, 0.31, 0.48, 0.57, 0.6, 0.57, 0, 0, 0, 0, 0, 0, 0.49, 0],
        [0, 0.59, 0.19, 0, 0, 0, 0, 0.2, 0.57, 0.69, 0.76, 0.76, 0.49, 0, 0, 0, 0, 0, 0.36, 0],
        [0, 0.58, 0.19, 0, 0, 0, 0, 0, 0.67, 0.83, 0.9, 0.92, 0.87, 0.12, 0, 0, 0, 0, 0.22, 0.07],
        [0, 0, 0.46, 0, 0, 0, 0, 0, 0.7, 0.93, 1, 1, 1, 0.61, 0, 0, 0, 0, 0.18, 0.11],
        [0, 0, 0.82, 0, 0, 0, 0, 0, 0.47, 1, 1, 0.98, 1, 0.96, 0.27, 0, 0, 0, 0.19, 0.1],
        [0, 0, 0.46, 0, 0, 0, 0, 0, 0.25, 1, 1, 0.84, 0.92, 0.97, 0.54, 0.14, 0.04, 0.1, 0.21, 0.05],
        [0, 0, 0, 0.4, 0, 0, 0, 0, 0.09, 0.8, 1, 0.82, 0.8, 0.85, 0.63, 0.31, 0.18, 0.19, 0.2, 0.01],
        [0, 0, 0, 0.36, 0.1, 0, 0, 0, 0.05, 0.54, 0.86, 0.79, 0.74, 0.72, 0.6, 0.39, 0.28, 0.24, 0.13, 0],
        [0, 0, 0, 0.01, 0.3, 0.07, 0, 0, 0.08, 0.36, 0.64, 0.7, 0.64, 0.6, 0.51, 0.39, 0.29, 0.19, 0.04, 0],
        [0, 0, 0, 0, 0.1, 0.24, 0.14, 0.1, 0.15, 0.29, 0.45, 0.53, 0.52, 0.46, 0.4, 0.31, 0.21, 0.08, 0, 0],
        [0, 0, 0, 0, 0, 0.08, 0.21, 0.21, 0.22, 0.29, 0.36, 0.39, 0.37, 0.33, 0.26, 0.18, 0.09, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.03, 0.13, 0.19, 0.22, 0.24, 0.24, 0.23, 0.18, 0.13, 0.05, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.06, 0.08, 0.09, 0.07, 0.05, 0.01, 0, 0, 0, 0, 0]
    ])
    if flip_x:
        orbitum = orbitum[:, ::-1]

    if flip_y:
        orbitum = orbitum[::-1, :]

    H, W = grid.shape
    h, w = orbitum.shape
    left = max(0, x - w // 2)
    top = max(0, y - h // 2)
    right = min(left + w, W)
    bottom = min(top + h, H)
    grid[top:bottom, left:right] = orbitum


def generate_colorbar(color_list, N):
    def linear_light_space(arr):
        return arr ** 2 / 255

    def linear_light_space_inverse(arr):
        return (255 * arr) ** 0.5

    n = len(color_list)
    col_n13 = linear_light_space(np.array(color_list)).astype(np.uint8)
    rgn = np.arange(n)
    rgN = np.linspace(0, n - 1, N)
    c0 = np.interp(rgN, rgn, col_n13[:, 0]).reshape(-1, 1, 1)
    c1 = np.interp(rgN, rgn, col_n13[:, 1]).reshape(-1, 1, 1)
    c2 = np.interp(rgN, rgn, col_n13[:, 2]).reshape(-1, 1, 1)
    res = np.round(linear_light_space_inverse(np.concatenate([c0, c1, c2], axis=2))).astype(np.uint8)
    return res


def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def create_K_lenia():
    mu = 0.5
    sigma = 0.15

    R = 13
    y, x = np.ogrid[-R:R, -R:R]
    distance = np.sqrt((1 + x) ** 2 + (1 + y) ** 2) / R

    K_lenia = gauss(distance, mu, sigma)
    K_lenia[distance > 1] = 0
    K_lenia = K_lenia / np.sum(K_lenia)

    return K_lenia


def applyRules(grid: np.ndarray, K_lenia):
    def growth(X):
        mu = 0.15
        sigma = 0.015
        return -1 + 2 * gauss(X, mu, sigma)

    U = signal.convolve2d(grid, K_lenia, mode='same', boundary='wrap')

    grid = grid + 0.1 * growth(U)
    return np.clip(grid, 0, 1)


def drawGrid(grid, colorbar_lut):
    N = len(colorbar_lut)
    grid_int = (grid * (N - 1)).astype(np.float32)
    prout = cv2.remap(colorbar_lut, np.zeros_like(grid_int), grid_int, interpolation=cv2.INTER_NEAREST)

    gbn.imshow("lenia", prout.astype(np.uint8), cv2_interpolation=cv2.INTER_NEAREST)


class LeniaView:
    def __init__(self):
        self.N = 100
        self.colors = [
            (0, 0, 0),
            (106, 97, 224),
            (215, 255, 253),
            (224, 97, 106),
        ]

        n = len(self.colors)

        win = gbn.create_window("lenia")

        self.callback_list = [lambda rgb, ind_=ind: self.on_change_color(ind_, rgb) for ind in range(n)]
        for ind in range(n):
            init_col = tuple(list(self.colors[ind])[::-1])
            win.create_color_picker(f"Color {ind}", on_change=self.callback_list[ind], initial_color_rgb=init_col)

        win.setMouseCallback(onMouse=self.on_mouse)

    def run(self):
        self.grid = np.zeros((HEIGHT, WIDTH))
        add_orbitum(self.grid, WIDTH//2, HEIGHT//2)
        K_lenia = create_K_lenia()
        self.colorbar_lut = generate_colorbar(self.colors, self.N)

        while True:
            self.grid = applyRules(self.grid, K_lenia)
            drawGrid(self.grid, self.colorbar_lut)
            if gbn.waitKeyEx(1) == 27:
                break

    def on_change_color(self, ind, rgb):
        r, g, b = rgb
        print(f"({b}, {g}, {r})")
        self.colors[ind] = b, g, r
        self.colorbar_lut = generate_colorbar(self.colors, self.N)

    def on_mouse(self, cvevent, x, y, flag, param):
        print(cvevent, x, y, flag, param)
        if flag == 1:
            add_orbitum(self.grid, int(x), int(y))

def main():
    lv = LeniaView()
    lv.run()


if __name__ == "__main__":
    main()

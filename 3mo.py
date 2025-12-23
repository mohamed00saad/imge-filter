import cv2
import numpy as np

cap = cv2.VideoCapture(1)

mode = 'o'
k_size = 5
t1, t2 = 50, 150
Q = 1.5
alpha = 2  # must satisfy 2*alpha < k_size*k_size

def ensure_odd(n, min_n=3):
    n = max(min_n, n)
    return n if n % 2 == 1 else n + 1

def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img

def arithmetic_mean(gray, k):
    return cv2.blur(gray, (k, k))

def geometric_mean(gray, k):
    g = gray.astype(np.float32) + 1.0
    logg = np.log(g)
    gm = np.exp(cv2.blur(logg, (k, k)))
    return np.clip(gm, 0, 255).astype(np.uint8)

def harmonic_mean(gray, k):
    g = gray.astype(np.float32) + 1.0
    hm = (k * k) / (cv2.blur(1.0 / g, (k, k)) + 1e-8)
    return np.clip(hm, 0, 255).astype(np.uint8)

def contraharmonic_mean(gray, k, Q):
    g = gray.astype(np.float32) + 1e-6
    num = cv2.blur(g ** (Q + 1.0), (k, k))
    den = cv2.blur(g ** Q, (k, k)) + 1e-8
    chm = num / den
    return np.clip(chm, 0, 255).astype(np.uint8)

def alpha_trimmed_mean(gray, k, alpha):
    N = k * k
    a = int(alpha)
    if 2 * a >= N:
        a = max(0, (N // 2) - 1)

    pad = k // 2
    g = gray.astype(np.float32)
    gp = cv2.copyMakeBorder(g, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)

    out = np.zeros_like(g)

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            w = gp[i:i+k, j:j+k].reshape(-1)
            w.sort()
            trimmed = w[a:N-a] if a > 0 else w
            out[i, j] = trimmed.mean()

    return np.clip(out, 0, 255).astype(np.uint8)

def max_filter(gray, k):
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(gray, kernel)

def min_filter(gray, k):
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(gray, kernel)

def midpoint_filter(gray, k):
    mx = max_filter(gray, k).astype(np.uint16)
    mn = min_filter(gray, k).astype(np.uint16)
    return ((mx + mn) // 2).astype(np.uint8)

def low_pass(gray, k):
    return cv2.blur(gray, (k, k))

def high_pass_sharpen(gray):
    kernel = np.array([[0, -1, 0],
                       [-1,  5, -1],
                       [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(gray, -1, kernel)

def sobel_magnitude(gray):
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx*sx + sy*sy)
    return cv2.convertScaleAbs(mag)

# Full names for each mode
MODE_NAME = {
    'o': "Original",
    'g': "Gray",
    'b': "Gaussian Blur",
    'c': "Canny (Manual)",
    's': "Sobel (Magnitude)",
    'a': "Arithmetic Mean",
    'e': "Geometric Mean",
    'h': "Harmonic Mean",
    'k': "Contraharmonic Mean",
    'm': "Median Filter",
    'x': "Max Filter",
    'n': "Min Filter",
    'd': "Midpoint Filter",
    't': "Alpha-Trimmed Mean",
    'l': "Low Pass Filter",
    'p': "High Pass / Sharpening",
}

def draw_label(img, title, subtitle=""):
    img = to_bgr(img)

    # Black box behind text for readability
    cv2.rectangle(img, (5, 5), (620, 90), (0, 0, 0), -1)

    cv2.putText(img, title, (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if subtitle:
        cv2.putText(img, subtitle, (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        break

    k_size = ensure_odd(k_size, 3)
    gray = to_gray(frame)

    title = MODE_NAME.get(mode, "Original")
    subtitle = ""

    if mode == 'o':
        output = frame

    elif mode == 'g':
        output = gray

    elif mode == 'b':
        output = cv2.GaussianBlur(frame, (k_size, k_size), 0)
        subtitle = f"k={k_size}"

    elif mode == 'c':
        edges = cv2.Canny(gray, t1, t2)
        output = edges
        subtitle = f"T1={t1}  T2={t2}"

    elif mode == 's':
        output = sobel_magnitude(gray)

    elif mode == 'a':
        output = arithmetic_mean(gray, k_size)
        subtitle = f"k={k_size}"

    elif mode == 'e':
        output = geometric_mean(gray, k_size)
        subtitle = f"k={k_size}"

    elif mode == 'h':
        output = harmonic_mean(gray, k_size)
        subtitle = f"k={k_size}"

    elif mode == 'k':
        output = contraharmonic_mean(gray, k_size, Q)
        subtitle = f"k={k_size}  Q={Q:.2f}"

    elif mode == 'm':
        output = cv2.medianBlur(gray, k_size)
        subtitle = f"k={k_size}"

    elif mode == 'x':
        output = max_filter(gray, k_size)
        subtitle = f"k={k_size}"

    elif mode == 'n':
        output = min_filter(gray, k_size)
        subtitle = f"k={k_size}"

    elif mode == 'd':
        output = midpoint_filter(gray, k_size)
        subtitle = f"k={k_size}"

    elif mode == 't':
        output = alpha_trimmed_mean(gray, k_size, alpha)
        subtitle = f"k={k_size}  alpha={alpha}"

    elif mode == 'l':
        output = low_pass(gray, k_size)
        subtitle = f"k={k_size}"

    elif mode == 'p':
        output = high_pass_sharpen(gray)

    else:
        output = frame
        title = "Original"

    output = draw_label(output, title, subtitle)

    cv2.imshow("Image Processing Project", output)

    k = cv2.waitKey(1) & 0xFF

    # Switch mode by keys
    if k in [ord(ch) for ch in "ogbcsaehkmxndtlp"]:
        mode = chr(k)

    # Kernel control
    if k == ord('+'):
        k_size += 2
    elif k == ord('-') and k_size > 3:
        k_size -= 2

    # Canny thresholds control
    if k == ord('1'): t1 = max(0, t1 - 5)
    if k == ord('2'): t1 = min(255, t1 + 5)
    if k == ord('3'): t2 = max(0, t2 - 5)
    if k == ord('4'): t2 = min(255, t2 + 5)

    # Contraharmonic Q control
    if k == ord('['): Q -= 0.1
    if k == ord(']'): Q += 0.1

    # Alpha control
    if k == ord(','): alpha = max(0, alpha - 1)
    if k == ord('.'): alpha = min((k_size*k_size)//2 - 1, alpha + 1)

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

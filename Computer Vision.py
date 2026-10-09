import cv2
import matplotlib.pyplot as plt

gambar = cv2.imread("apple.jpg")

if gambar is None:
    print("Gambar apple tidak ditemukan!")
    print("Pastikan file bernama 'apple.jpg' berada di folder yang sama.")
    exit()

print("Gambar berhasil dibaca!")

tinggi, lebar, channel = gambar.shape

print("Lebar gambar :", lebar, "pixel")
print("Tinggi gambar :", tinggi, "pixel")
print("Jumlah channel :", channel)

gambar_rgb = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)

grayscale = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(8, 6))

plt.imshow(gambar_rgb)
plt.title("Gambar Apple")
plt.axis('off')

plt.show()

plt.figure(figsize=(8, 6))

plt.imshow(grayscale, cmap="gray")
plt.title("Gambar Apple - Grayscale")
plt.axis("off")

plt.show()

cv2.imwrite("apple_grayscale.jpg", grayscale)

print("Gambar grascale berhasil disimpan sebagai apple_grayscale.jpg")

gambar_resize = cv2.resize(gambar_rgb, (300, 300))

plt.figure(figsize=(8, 6))
plt.imshow(gambar_resize)
plt.title("Apple setelah Resize")
plt.axis("off")
plt.show()

blur = cv2.GaussianBlur(grayscale, (5,5), 0)

plt.figure(figsize=(8, 6))
plt.imshow(blur, cmap="gray")
plt.title("Apple Setelah Gaussian Blur")
plt.axis("off")
plt.show()

ret, threshold = cv2.threshold(
    blur,
    120,
    255,
    cv2.THRESH_BINARY
)

plt.figure(figsize=(8, 6))
plt.imshow(threshold, cmap="gray")
plt.title("Hasil Thresholding")
plt.axis("off")
plt.show()

edges = cv2.Canny(blur, 50, 150)

plt.figure(figsize=(8, 6))
plt.imshow(edges, cmap="gray")
plt.title("Deteksi Tepi Apple - Canny")
plt.axis("off")
plt.show()

contours, hierarchy = cv2.findContours(
    threshold,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Jumlah objek/contour yang ditemukan:", len(contours))

gambar_contour = gambar_rgb.copy()

cv2.drawContours(
    gambar_contour,
    contours,
    -1,
    (0, 255, 0),
    3
)

plt.figure(figsize=(8, 6))
plt.imshow(gambar_contour)
plt.title("Contour Apple")
plt.axis("off")
plt.show()

gambar_box = gambar_rgb.copy()

for contour in contours:

    luas = cv2.contourArea(contour)

    if luas > 500:

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            gambar_box,
             (x, y),
             (x + w, y + h),
             (255, 0, 0),
            3
        )

        print("Luas objek:", luas)
        print("Posisi X:", x)
        print("Posisi Y:", y)
        print("Lebar:", w)
        print("Tinggi:", h)

plt.figure(figsize=(8, 6))
plt.imshow(gambar_box)
plt.title("Bounding Box Apple")
plt.axis("off")
plt.show()

gambar_label = gambar_rgb.copy()

for i, contour in enumerate(contours):

    luas = cv2.contourArea(contour)

    if luas > 500:

        x, y, w, h = cv2.boundingRect(contour)

        cv2.putText(
            gambar_label,
            "Apel Imut",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

plt.figure(figsize=(8, 6))
plt.imshow(gambar_label)
plt.title("Label Objek Apel")
plt.axis('off')
plt.show()

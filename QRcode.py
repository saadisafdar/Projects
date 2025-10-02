import qrcode
from PIL import Image
import os

data = input("Enter anything to generate QR: ")

# Create QR code
qr = qrcode.QRCode(version=3, box_size=10, border=4)
qr.add_data(data)
qr.make(fit=True)

# Generate image
image = qr.make_image(fill="black", back_color="peru")
image.save("qr_code.png")

# Display confirmation
print(f"QR code saved as 'qr_code.png'")
print(f"File location: {os.path.abspath('qr_code.png')}")

# Open the image
Image.open("qr_code.png").show()
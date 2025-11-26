from rembg import remove
from PIL import Image

input_path = 'input/clgot.jpeg'
output_path = 'output/clgot.png'

inp = Image.open(input_path)
out = remove(inp)
out.save(output_path)

print("Saved:", output_path)

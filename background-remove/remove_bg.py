from rembg import remove
from PIL import Image

input_path = 'input/clgot.jpeg'
output_path = 'output/clgot.png'

inp = Image.open(input_path)
out = remove(inp)
out.save(output_path)

print("Saved:", output_path)




from PIL import Image

img = Image.open("input/clgot.jpeg")
up = img.resize((img.width*2, img.height*2), Image.BICUBIC)
up.save("output/clgot_2x.png")

print("Saved:", "output/clgot_2x.png")





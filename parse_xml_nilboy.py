import sys, os
import xml.etree.ElementTree

def parse_xml(xmlpath, name):
	labels = []
	xmlpath = xmlpath + "/" + name
	e = xml.etree.ElementTree.parse(xmlpath).getroot()
	objects = ["backpack", "person"]
	for child in e.findall("object"):
		objtype = child.find("name").text
		if objtype in objects:
			print(name)
			box = child.find("bndbox")
			print(box)
			xmin = box.find("xmin").text
			ymin = box.find("ymin").text
			xmax = box.find("xmax").text
			ymax = box.find("ymax").text
			labels.append([xmin, ymin, xmax, ymax, objects.index(objtype)])
	return labels

def convert_to_string(image_path, labels):
  out_string = ''
  out_string += image_path
  for label in labels:
    for i in label:
      out_string += ' ' + str(i)
  out_string += '\n'
  return out_string


if __name__ == '__main__':
	xmlpath = sys.argv[1]
	imgpath = sys.argv[2]
	outfile = sys.argv[3]
	cwd = os.getcwd()
	out = open(outfile, 'w')
	for f in os.scandir(xmlpath):
		lab = parse_xml(xmlpath, f.name)
		imgname = f.name.replace("xml", "jpg")
		record = convert_to_string(cwd + "/" + imgpath + "/" + imgname, lab)
		out.write(record)





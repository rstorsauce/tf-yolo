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
			box = child.find("bndbox")
			xmin = box.find("xmin").text
			ymin = box.find("ymin").text
			xmax = box.find("xmax").text
			ymax = box.find("ymax").text
			labels.append([objects.index(objtype), int(xmin), int(ymin), int(xmax), int(ymax)])
	return labels

def convert_to_string(labels):
	out = ""
	for obj in labels:
		width = obj[3] - obj[1]
		height = obj[4] - obj[2]
		x = (width / 2) + obj[1]
		y = (height / 2) + obj[2]
		objline = "{} {} {} {} {}\n".format(obj[0], x, y, width, height)
		out += objline
	return out



if __name__ == '__main__':
	imgdir = sys.argv[1]
	xmldir = sys.argv[2]

	for f in os.scandir(xmldir):
		print(f.name)
		labels = parse_xml(xmldir, f.name)
		outstr = convert_to_string(labels)
		outfile = imgdir + "/" + f.name.replace("xml", "txt")
		out = open(outfile, 'w')
		out.write(outstr)




import os
import xml.etree.ElementTree as ET
yuan_dir = './Annotations'  # 设置原始标签路径为 Annos
new_dir = './na'  # 设置新标签路径 Annotations
for filename in os.listdir(yuan_dir):
    file_path = os.path.join(yuan_dir, filename)
    new_path=os.path.join(new_dir,filename)
    dom = ET.parse(file_path)
    root = dom.getroot()
    for obj in root.iter('object'):  # 获取object节点中的name子节点
        if obj.find('name').text== 'Misc':
            root.remove(obj)
            print(file_path)
            #print("change %s to %s." % (yuan_name, new_name1))
 # 保存到指定文件
#    break
#    dom.write(new_path, xml_declaration=True)


def get_svg_dimensions(file_path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(file_path)
    root = tree.getroot()
    width_  = root.attrib['width']
    height_ = root.attrib['height']
    width  = float(width_ .replace('pt', '').replace('px', ''))
    height = float(height_.replace('pt', '').replace('px', ''))
    return width, height, width_, height_

def stack_svg(svg_files, stack_type='vertical'):
    import xml.etree.ElementTree as ET
    widths, heights, _,_ = zip(*(get_svg_dimensions(f) for f in svg_files))

    y_offset = 0
    x_offset = 0

    print(f'stack_svg: {widths=}, {heights=}')
    if stack_type == 'vertical':
        total_width = max(widths)
        total_height = sum(heights)

        print(f'{total_width=}, {total_height=}')

        # Create a new SVG root element
        new_svg = ET.Element('svg', width=f'{total_width}px', height=f'{total_height}px', xmlns="http://www.w3.org/2000/svg")

        for svg_file, height in zip(svg_files, heights):
            svg_fragment = ET.parse(svg_file).getroot()
            for element in list(svg_fragment):
                #adjust_element_coordinates(element, 0, y_offset)
                #element.attrib['transform'] = f'translate(0,{y_offset})'
                adjust_transform(element, x_offset, y_offset)
                new_svg.append(element)
            y_offset += height
    elif stack_type == 'horizontal':
        total_width = sum(widths)
        total_height = max(heights)

        # Create a new SVG root element
        new_svg = ET.Element('svg', width=f'{total_width}px', height=f'{total_height}px', xmlns="http://www.w3.org/2000/svg")

        for svg_file, width in zip(svg_files, widths):
            svg_fragment = ET.parse(svg_file).getroot()
            for element in list(svg_fragment):
                #adjust_element_coordinates(element, x_offset, 0)
                # this is less reliable
                #element.attrib['transform'] = f'translate({x_offset},0)'
                adjust_transform(element, x_offset, y_offset)
                new_svg.append(element)
            x_offset += width
    else:
        raise ValueError(stack_type)

    # Write the combined SVG to the output file
    tree = ET.ElementTree(new_svg)
    return tree

def adjust_transform(element, x_offset, y_offset):
    if 'transform' in element.attrib:
        transforms = element.attrib['transform'].\
                replace('translate(', '').replace(')', '').split(',')
        if len(transforms) == 2:
            tx, ty = map(float, transforms)
            tx += x_offset
            ty += y_offset
            element.attrib['transform'] = f'translate({tx},{ty})'
        elif len(transforms) == 1:
            tx = float(transforms[0])
            tx += x_offset
            element.attrib['transform'] = f'translate({tx},0)'
    else:
        element.attrib['transform'] = f'translate({x_offset},{y_offset})'

def adjust_element_coordinates(element, x_offset, y_offset):
    if 'x' in element.attrib:
        element.attrib['x'] = str(float(element.attrib['x']) + x_offset)
    if 'y' in element.attrib:
        element.attrib['y'] = str(float(element.attrib['y']) + y_offset)
    for child in element:
        adjust_element_coordinates(child, x_offset, y_offset)

def stackSVGandShowJupy(svg_files, stack_type, fnfout, show=1):
    '''
        svg_files: full paths to svg files
    '''
    assert stack_type in ['vertical','horizontal']
    print(f'Stacking with {stack_type=}',  svg_files)
    restree = stack_svg(svg_files, stack_type=stack_type)

    #fnfout = pjoin(path_fig, 'behav', f'Fig3_stacked_dynESpert_stage.svg')
    restree.write(fnfout)
    print(f"SVG files have been combined and saved as {fnfout}")

    if show:
        from IPython.display import SVG, display, Image
        # Display the stacked SVG file
        print('svg pic:')
        display(SVG(filename=fnfout));

def svg2png(fnfout, dpi=None, inpdpi=72):    
    # cairo def is 96, but matplotlib def is (perhaps) 72
    assert '.svg' in fnfout
    import cairosvg
    from PIL import Image
    
    output_height = None
    output_width  = None

    # Convert SVG to PNG
    fnfout_png = fnfout.replace('.svg','.png')
    if dpi is not None:
        width, height, width_, height_ = get_svg_dimensions(fnfout)
        #calculate output dimensions
        output_width  = width * dpi / inpdpi  # Assuming 96 DPI for original dimensions
        output_height = height * dpi / inpdpi

        fnfout_png = fnfout_png[:-4] + f'_dpi={dpi}.png'
        dpi_ = dpi
    else:
        dpi_ = 72
    with open(fnfout, 'rb') as sf:    
        # scale=
        cairosvg.svg2png(sf.read(), write_to=fnfout_png, dpi=dpi_ ,  output_width = output_width, output_height = output_height)    
        print(f'Saved to {fnfout_png}')

    if dpi is not None:
        # open the PNG file with Pillow
        img = Image.open(fnfout_png)

        # Set the DPI metadata
        img.save(fnfout_png, dpi=(dpi, dpi))

    # Display the PNG file
    #tree.write(output_file)
    return fnfout_png

def svg2pdf(fnfout): #, dpi=None, inpdpi=72):
    assert '.svg' in fnfout
    import cairosvg
    fnfout_pdf = fnfout.replace('.svg','.pdf')
    cairosvg.svg2pdf(url=fnfout, write_to=fnfout_pdf)#, dpi=dpi)
    print(f'Saved to {fnfout_pdf}')
    return fnfout_pdf
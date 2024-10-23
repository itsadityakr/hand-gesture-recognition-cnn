import graphviz

# Create a new Graphviz Digraph object
dot = graphviz.Digraph(format='png', graph_attr={'dpi': '300'})  # Increase DPI for better quality

# Set node styles for better visualization
dot.attr('node', shape='rect', style='filled', fillcolor='#e0e0e0', fontname='Helvetica', fontsize='12')

# Input Layer
dot.node('A', 'Input\n(32x32x1)', shape='rect', fillcolor='#add8e6')
# First Convolution Layer
dot.node('B', 'Conv2D\n(32 filters, 3x3)\nOutput: (30x30x32)\nParameters: 320', fillcolor='#98fb98')
# First Activation Layer
dot.node('C', 'Activation\n(ReLU)', fillcolor='#ffebcd')
# First Max Pooling Layer
dot.node('D', 'MaxPooling2D\n(Pool Size: 2x2)\nOutput: (15x15x32)', fillcolor='#ffe4b5')
# Second Convolution Layer
dot.node('E', 'Conv2D\n(64 filters, 3x3)\nOutput: (13x13x64)\nParameters: 18,496', fillcolor='#98fb98')
# Second Activation Layer
dot.node('F', 'Activation\n(ReLU)', fillcolor='#ffebcd')
# Second Max Pooling Layer
dot.node('G', 'MaxPooling2D\n(Pool Size: 2x2)\nOutput: (6x6x64)', fillcolor='#ffe4b5')
# Flatten Layer
dot.node('H', 'Flatten\nOutput: (2304)', fillcolor='#ffd700')
# First Dense Layer
dot.node('I', 'Dense\n(128 units)\nOutput: (128)\nParameters: 294,400', fillcolor='#ffb6c1')
# Dropout Layer
dot.node('J', 'Dropout\n(50%)', fillcolor='#ffb6c1')
# Output Dense Layer
dot.node('K', 'Dense\n(num_classes)\nOutput: (num_classes)\nActivation: Softmax', fillcolor='#add8e6')
# Output Layer
dot.node('L', 'Output\n(num_classes)', fillcolor='#add8e6')

# Create directed edges between nodes
dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH', 'HI', 'IJ', 'JK', 'KL'])

# Render the diagram with increased quality
dot.render('cnn_architecture_detailed', format='png', cleanup=True)

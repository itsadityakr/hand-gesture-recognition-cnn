import graphviz


def create_dataset_block():
    dot = graphviz.Digraph(format='png', graph_attr={'dpi': '300', 'rankdir': 'TB'})
    dot.attr('node', shape='box', style='filled', fillcolor='#ffcccb', fontname='Helvetica', fontsize='12')
    dot.node('Z', 'Dataset Block', shape='box', fillcolor='#ff6666')
    dot.node('Z0', 'Dataset (Input Images)', fillcolor='#ff9999')
    dot.node('Z1', 'Gesture: B (Backward)', fillcolor='#ff9999')
    dot.node('Z2', 'Gesture: F (Forward)', fillcolor='#ff9999')
    dot.node('Z3', 'Gesture: L (Left)', fillcolor='#ff9999')
    dot.node('Z4', 'Gesture: R (Right)', fillcolor='#ff9999')
    dot.node('Z5', 'Gesture: S (Stop)', fillcolor='#ff9999')
    dot.edge('Z', 'Z0')
    dot.edge('Z0', 'Z1')
    dot.edge('Z1', 'Z2')
    dot.edge('Z2', 'Z3')
    dot.edge('Z3', 'Z4')
    dot.edge('Z4', 'Z5')
    dot.render('dataset_block', format='png', cleanup=True)


def create_cnn_block():
    dot = graphviz.Digraph(format='png', graph_attr={'dpi': '300', 'rankdir': 'TB'})
    dot.attr('node', shape='box', style='filled', fillcolor='#ccffcc', fontname='Helvetica', fontsize='12')
    dot.node('C', 'CNN Block', shape='box', fillcolor='#66ff66')
    dot.node('A', 'Input Layer', fillcolor='#99ff99')
    dot.node('B', 'Conv2D', fillcolor='#66ff99')
    dot.node('D', 'MaxPooling2D', fillcolor='#99ff66')
    dot.node('E', 'Conv2D', fillcolor='#66ff99')
    dot.node('G', 'MaxPooling2D', fillcolor='#99ff66')
    dot.node('H', 'Flatten', fillcolor='#ffff66')
    dot.node('I', 'Dense Layer', fillcolor='#ff99cc')
    dot.edge('C', 'A')
    dot.edge('A', 'B')
    dot.edge('B', 'D')
    dot.edge('D', 'E')
    dot.edge('E', 'G')
    dot.edge('G', 'H')
    dot.edge('H', 'I')
    dot.render('cnn_block', format='png', cleanup=True)


def create_output_block():
    dot = graphviz.Digraph(format='png', graph_attr={'dpi': '300', 'rankdir': 'TB'})
    dot.attr('node', shape='box', style='filled', fillcolor='#ffdab9', fontname='Helvetica', fontsize='12')
    dot.node('O', 'Output Block', shape='box', fillcolor='#ffb366')
    dot.node('L', 'Final Gesture Recognition', fillcolor='#ff9966')
    dot.edge('O', 'L')
    dot.render('output_block', format='png', cleanup=True)


def create_combined_block():
    dot = graphviz.Digraph(format='png', graph_attr={'dpi': '300', 'rankdir': 'TB'})
    dot.attr('node', shape='box', style='filled', fontname='Helvetica', fontsize='12')

    # Dataset Block
    dot.node('Z', 'Dataset Block', fillcolor='#ff6666')
    dot.node('Z0', 'Dataset (Input Images)', fillcolor='#ff9999')
    dot.node('Z1', 'Gesture: B', fillcolor='#ff9999')
    dot.node('Z2', 'Gesture: F', fillcolor='#ff9999')
    dot.node('Z3', 'Gesture: L', fillcolor='#ff9999')
    dot.node('Z4', 'Gesture: R', fillcolor='#ff9999')
    dot.node('Z5', 'Gesture: S', fillcolor='#ff9999')
    dot.edge('Z', 'Z0')
    dot.edge('Z0', 'Z1')
    dot.edge('Z1', 'Z2')
    dot.edge('Z2', 'Z3')
    dot.edge('Z3', 'Z4')
    dot.edge('Z4', 'Z5')

    # CNN Block
    dot.node('C', 'CNN Block', fillcolor='#66ff66')
    dot.node('A', 'Input Layer', fillcolor='#99ff99')
    dot.node('B', 'Conv2D', fillcolor='#66ff99')
    dot.node('D', 'MaxPooling2D', fillcolor='#99ff66')
    dot.node('E', 'Conv2D', fillcolor='#66ff99')
    dot.node('G', 'MaxPooling2D', fillcolor='#99ff66')
    dot.node('H', 'Flatten', fillcolor='#ffff66')
    dot.node('I', 'Dense Layer', fillcolor='#ff99cc')

    # Output Block
    dot.node('O', 'Output Block', fillcolor='#ffb366')
    dot.node('L', 'Final Gesture Recognition', fillcolor='#ff9966')

    # Connections
    dot.edge('Z5', 'C')
    dot.edge('C', 'A')
    dot.edge('A', 'B')
    dot.edge('B', 'D')
    dot.edge('D', 'E')
    dot.edge('E', 'G')
    dot.edge('G', 'H')
    dot.edge('H', 'I')
    dot.edge('I', 'O')
    dot.edge('O', 'L')

    dot.render('combined_architecture', format='png', cleanup=True)


while True:
    print("\nMenu:")
    print("1. Print Dataset Block")
    print("2. Print CNN Block")
    print("3. Print Output Block")
    print("4. Print Combined Architecture")
    print("5. Exit")
    choice = input("Enter your choice: ")

    if choice == '1':
        create_dataset_block()
        print("Dataset block saved as dataset.png")
    elif choice == '2':
        create_cnn_block()
        print("CNN block saved as cnn_block.png")
    elif choice == '3':
        create_output_block()
        print("Output block saved as output_block.png")
    elif choice == '4':
        create_combined_block()
        print("Combined architecture saved as combined_architecture.png")
    elif choice == '5':
        break
    else:
        print("Invalid choice. Please try again.")

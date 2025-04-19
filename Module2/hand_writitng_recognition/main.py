# MNIST Classification using TensorFlow 2.x
# This program uses a multi-layer perceptron to classify handwritten digits
# from the MNIST dataset

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist

# Split into training and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoding (e.g., 7 -> [0,0,0,0,0,0,0,1,0,0])
y_train_orig = y_train  # Save original labels
y_test_orig = y_test    # Save original labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Function to visualize a sample image
def display_sample(num):
    # Print the one-hot array of this sample's label
    print(y_train[num]) 
    # Print the label converted back to a number
    label = y_train[num].argmax(axis=0)
    # Display the image
    plt.figure(figsize=(3, 3))
    plt.imshow(x_train[num], cmap=plt.get_cmap('gray_r'))
    plt.title(f'Sample: {num}  Label: {label}')
    plt.show()

# Display a sample image
print("Displaying sample image:")
display_sample(1234)

# Visualize multiple images stacked together
def visualize_multiple_samples():
    # Reshape first 500 training images into a grid
    fig = plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.get_cmap('gray_r'))
        plt.xlabel(y_train_orig[i])
    plt.suptitle('MNIST Sample Images', fontsize=16)
    plt.show()

# Create and train the base model
def create_base_model():
    print("\n--- TRAINING BASE MODEL ---")
    # Create a sequential model
    model = tf.keras.models.Sequential([
        # Flatten the 28x28 images to 784-dimensional vectors
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # First dense layer with 512 nodes and ReLU activation
        tf.keras.layers.Dense(512, activation='relu'),
        # Output layer with 10 nodes (one per digit) and softmax activation
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=100,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    # Evaluate the model
    _, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Base model accuracy: {test_accuracy:.4f}")
    
    return model, test_accuracy, history

# Function to find misclassified images
def find_misclassified_images(model, num_to_show=5):
    print("\n--- FINDING MISCLASSIFIED IMAGES ---")
    # Get predictions from the model
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)
    actual_labels = np.argmax(y_test, axis=1)
    
    # Find indices of misclassified images
    misclassified_indices = np.where(predicted_labels != actual_labels)[0]
    
    print(f"Found {len(misclassified_indices)} misclassified images out of 10,000 test images")
    print(f"Error rate: {len(misclassified_indices)/len(x_test):.4f}")
    
    # Display some misclassified images
    for i in range(min(num_to_show, len(misclassified_indices))):
        idx = misclassified_indices[i]
        plt.figure(figsize=(3, 3))
        plt.imshow(x_test[idx], cmap='gray_r')
        plt.title(f"Actual: {actual_labels[idx]}, Predicted: {predicted_labels[idx]}")
        plt.show()
    
    return misclassified_indices

# Function to test different numbers of hidden nodes
def test_hidden_nodes():
    print("\n--- TESTING DIFFERENT NUMBERS OF HIDDEN NODES ---")
    # Test different numbers of hidden nodes
    hidden_node_options = [128, 256, 512, 1024]
    results = []
    
    for nodes in hidden_node_options:
        print(f"\nTesting model with {nodes} hidden nodes")
        
        # Create a new model with the specified number of hidden nodes
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(nodes, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train for fewer epochs to save time
        history = model.fit(
            x_train, y_train,
            epochs=5,
            batch_size=100,
            validation_data=(x_test, y_test),
            verbose=1
        )
        
        # Evaluate the model
        _, test_accuracy = model.evaluate(x_test, y_test)
        results.append((nodes, test_accuracy))
        print(f"Final accuracy with {nodes} nodes: {test_accuracy:.4f}")
    
    # Print results
    print("\nHidden Nodes Results:")
    for nodes, acc in results:
        print(f"Hidden Nodes: {nodes}, Accuracy: {acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    nodes, accuracies = zip(*results)
    plt.plot(nodes, accuracies, 'o-', linewidth=2)
    plt.xlabel('Number of Hidden Nodes')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of Hidden Layer Size on Accuracy')
    plt.grid(True)
    plt.show()
    
    return results

# Function to test different learning rates
def test_learning_rates():
    print("\n--- TESTING DIFFERENT LEARNING RATES ---")
    # Test different learning rates
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    results = []
    
    for lr in learning_rates:
        print(f"\nTesting model with learning rate {lr}")
        
        # Create a new model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # Compile the model with the specified learning rate
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train for fewer epochs to save time
        history = model.fit(
            x_train, y_train,
            epochs=5,
            batch_size=100,
            validation_data=(x_test, y_test),
            verbose=1
        )
        
        # Evaluate the model
        _, test_accuracy = model.evaluate(x_test, y_test)
        results.append((lr, test_accuracy))
        print(f"Final accuracy with learning rate {lr}: {test_accuracy:.4f}")
    
    # Print results
    print("\nLearning Rate Results:")
    for lr, acc in results:
        print(f"Learning Rate: {lr}, Accuracy: {acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    rates, accuracies = zip(*results)
    plt.semilogx(rates, accuracies, 'o-', linewidth=2)
    plt.xlabel('Learning Rate')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of Learning Rate on Accuracy')
    plt.grid(True)
    plt.show()
    
    return results

# Function to test network with two hidden layers
def test_two_hidden_layers():
    print("\n--- TESTING TWO HIDDEN LAYERS ---")
    
    # Create a new model with two hidden layers
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),  # Second hidden layer
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Train for 10 epochs
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=100,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    # Evaluate the model
    _, test_accuracy = model.evaluate(x_test, y_test)
    print(f"\nTwo Hidden Layers Final Accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return test_accuracy, model

# Function to test different batch sizes
def test_batch_sizes():
    print("\n--- TESTING DIFFERENT BATCH SIZES ---")
    # Test different batch sizes
    batch_size_options = [50, 100, 200]
    results = []
    
    for batch_size in batch_size_options:
        print(f"\nTesting model with batch size {batch_size}")
        
        # Create a new model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train for 5 epochs with the specified batch size
        history = model.fit(
            x_train, y_train,
            epochs=5,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            verbose=1
        )
        
        # Evaluate the model
        _, test_accuracy = model.evaluate(x_test, y_test)
        results.append((batch_size, test_accuracy))
        print(f"Final accuracy with batch size {batch_size}: {test_accuracy:.4f}")
    
    # Print results
    print("\nBatch Size Results:")
    for batch_size, acc in results:
        print(f"Batch Size: {batch_size}, Accuracy: {acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    sizes, accuracies = zip(*results)
    plt.plot(sizes, accuracies, 'o-', linewidth=2)
    plt.xlabel('Batch Size')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of Batch Size on Accuracy')
    plt.grid(True)
    plt.show()
    
    return results

# Function to build the best model based on our findings
def best_model():
    print("\n--- BUILDING THE BEST MODEL ---")
    
    # Create a model with optimized hyperparameters
    # Based on experiments, using:
    # - Two hidden layers (1024 and 512 nodes)
    # - Adam optimizer with learning rate of 0.1
    # - Dropout for regularization
    # - Batch size of 100
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),  # Add dropout for regularization
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),  # Add dropout for regularization
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Use Adam optimizer instead of SGD
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Train for 20 epochs
    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=100,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    # Evaluate the model
    _, test_accuracy = model.evaluate(x_test, y_test)
    print(f"\nBest Model Final Accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return test_accuracy, model

# Main function to run all experiments
def run_experiments():
    # Dictionary to store all results
    all_results = {}
    
    # Run base model
    base_model, base_accuracy, _ = create_base_model()
    all_results['base_accuracy'] = base_accuracy
    
    # Find misclassified images
    misclassified = find_misclassified_images(base_model)
    all_results['misclassified_count'] = len(misclassified)
    
    # Test different numbers of hidden nodes
    nodes_results = test_hidden_nodes()
    all_results['nodes_results'] = nodes_results
    
    # Test different learning rates
    lr_results = test_learning_rates()
    all_results['lr_results'] = lr_results
    
    # Test two hidden layers
    two_layers_accuracy, _ = test_two_hidden_layers()
    all_results['two_layers_accuracy'] = two_layers_accuracy
    
    # Test different batch sizes
    batch_results = test_batch_sizes()
    all_results['batch_results'] = batch_results
    
    # Build the best model
    best_accuracy, _ = best_model()
    all_results['best_accuracy'] = best_accuracy
    
    # Print summary of all results
    print("\n============ SUMMARY OF RESULTS ============")
    print(f"Base Model Accuracy: {base_accuracy:.4f}")
    print(f"Misclassified Images: {len(misclassified)} out of 10,000 ({len(misclassified)/100:.2f}%)")
    
    print("\nHidden Nodes Results:")
    for nodes, acc in nodes_results:
        print(f"Hidden Nodes: {nodes}, Accuracy: {acc:.4f}")
    
    print("\nLearning Rate Results:")
    for lr, acc in lr_results:
        print(f"Learning Rate: {lr}, Accuracy: {acc:.4f}")
    
    print(f"\nTwo Hidden Layers Accuracy: {two_layers_accuracy:.4f}")
    
    print("\nBatch Size Results:")
    for batch_size, acc in batch_results:
        print(f"Batch Size: {batch_size}, Accuracy: {acc:.4f}")
    
    print(f"\nBest Model Accuracy: {best_accuracy:.4f}")
    
    return all_results

# Answer specific assignment questions
def answer_question1():
    # Q1: What is the accuracy of the model?
    print("\n--- QUESTION 1: WHAT IS THE ACCURACY OF THE MODEL? ---")
    model, accuracy, _ = create_base_model()
    print(f"Answer: The accuracy of the base model is: {accuracy:.4f}")
    return accuracy

def answer_question2():
    # Q2: What are some of the misclassified images?
    print("\n--- QUESTION 2: WHAT ARE SOME OF THE MISCLASSIFIED IMAGES? ---")
    model, _, _ = create_base_model()
    misclassified = find_misclassified_images(model, 5)
    print("Answer: See the images above for examples of misclassified digits.")
    return misclassified

def answer_question3():
    # Q3: How is the accuracy affected by using more/fewer hidden neurons?
    print("\n--- QUESTION 3: HOW IS ACCURACY AFFECTED BY HIDDEN NEURONS? ---")
    results = test_hidden_nodes()
    print("Answer: See the plot and results above showing how accuracy changes with different numbers of hidden neurons.")
    return results

def answer_question4():
    # Q4: How is the accuracy affected by using different learning rates?
    print("\n--- QUESTION 4: HOW IS ACCURACY AFFECTED BY LEARNING RATES? ---")
    results = test_learning_rates()
    print("Answer: See the plot and results above showing how accuracy changes with different learning rates.")
    return results

def answer_question5():
    # Q5: How is accuracy affected by adding another hidden layer?
    print("\n--- QUESTION 5: HOW IS ACCURACY AFFECTED BY ADDING ANOTHER HIDDEN LAYER? ---")
    base_model, base_accuracy, _ = create_base_model()
    two_layers_accuracy, _ = test_two_hidden_layers()
    difference = two_layers_accuracy - base_accuracy
    print(f"Answer: Base model accuracy: {base_accuracy:.4f}")
    print(f"Two-layer model accuracy: {two_layers_accuracy:.4f}")
    print(f"Difference: {difference:.4f} ({difference*100:.2f}%)")
    if difference > 0:
        print("Adding a second hidden layer improved accuracy.")
    else:
        print("Adding a second hidden layer did not improve accuracy.")
    return base_accuracy, two_layers_accuracy

def answer_question6():
    # Q6: How is accuracy affected by using different batch sizes?
    print("\n--- QUESTION 6: HOW IS ACCURACY AFFECTED BY BATCH SIZES? ---")
    results = test_batch_sizes()
    print("Answer: See the plot and results above showing how accuracy changes with different batch sizes.")
    return results

def answer_question7():
    # Q7: What is the best accuracy you can get from this MLP?
    print("\n--- QUESTION 7: WHAT IS THE BEST ACCURACY YOU CAN GET? ---")
    best_accuracy, _ = best_model()
    print(f"Answer: The best accuracy achieved is: {best_accuracy:.4f}")
    return best_accuracy

# This is the main entry point for the program
if __name__ == "__main__":
    print("\n===== MNIST CLASSIFICATION EXPERIMENTS =====")
    print("What would you like to do?")
    print("1. Run base model only")
    print("2. Run all experiments")
    print("3. Answer specific question (1-7)")
    print("4. Show sample images")
    
    try:
        choice = int(input("Enter your choice (1-4): "))
        
        if choice == 1:
            # Just run the base model
            print("\nRunning base model...")
            base_model, base_accuracy, base_history = create_base_model()
            print(f"Base model accuracy: {base_accuracy:.4f}")
            
        elif choice == 2:
            # Run all experiments
            print("\nRunning all experiments...")
            results = run_experiments()
            
        elif choice == 3:
            # Answer specific question
            question = int(input("Which question would you like to answer (1-7)? "))
            if question == 1:
                answer_question1()
            elif question == 2:
                answer_question2()
            elif question == 3:
                answer_question3()
            elif question == 4:
                answer_question4()
            elif question == 5:
                answer_question5()
            elif question == 6:
                answer_question6()
            elif question == 7:
                answer_question7()
            else:
                print("Invalid question number. Please enter a number between 1 and 7.")
                
        elif choice == 4:
            # Show sample images
            print("\nShowing sample images...")
            visualize_multiple_samples()
            
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")
            
    except ValueError:
        # Default to just running the base model
        print("\nInvalid input. Running base model by default...")
        base_model, base_accuracy, base_history = create_base_model()
        print(f"Base model accuracy: {base_accuracy:.4f}")
import matplotlib.pyplot as plt


def show_images(images, labels):
    while True:
        print(f"\nTo check any of the {len(labels)} images, enter an image number between 1 & {len(labels)}\n"
              f"If you want to stop checking the images, enter 'Exit'\n")
        index = input(f"Enter the image number: ")
        if index.isdigit():
            index = int(index) - 1
        elif index.lower() == 'exit':
            break
        else:
            print(
                f"\nInvalid action, please do one of the following things:\n"
                f"1.\n"
                f"Provide the numerical value as input, not the word representation of the number.\n"
                f"For example, enter '1' instead of 'one', '42' instead of 'forty-two', and so on.\n"
                f"2.\n"
                f"If you want to stop checking images, enter 'Exit', make sure you do not make any spelling errors.\n\n"
            )
            continue

        image = images[index]
        label = labels[index]

        plt.imshow(
            image.reshape(28, 28),
            cmap="Greys"
        )

        plt.title(
            f"Predicted label: {label}"
        )

        plt.show()

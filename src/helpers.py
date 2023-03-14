# file for helper functions
import editdistance
import os

# i/o functions


def text_error_rate(label, prediction):
    """
    Computes the text error rate (TER) between two strings.
    Parameters
    ----------
    label : str
        The ground truth label.
    prediction : str
        The predicted label.
    Returns
    -------
    float
        The text error rate between the two strings.
    """
    # Compute the edit distance between the two strings.
    distance = editdistance.eval(label, prediction)

    # Compute the length of the longest string.
    length = max(len(label), len(prediction))

    # Return the edit distance divided by the length of the longest string.
    return distance / length


def get_filenames(directory):
    """
    Gets the filenames in a directory.
    Parameters
    ----------
    directory : str
        The directory to search.
    Returns
    -------
    list
        The filenames.
    """
    # Get the filenames in the given directory.
    filenames = os.listdir(directory)

    # Remove the file extensions.
    filenames = [filename.split('.')[0] for filename in filenames]

    # Return the filenames.
    return filenames


if __name__ == '__main__':
    # Compute the text error rate between two strings.
    label = 'this is a test'
    prediction = 'this is a test!'
    print(text_error_rate(label, prediction))

"""Lab 06 - Face recognition with an SVM.

>>> THIS IS THE ONLY FILE YOU HAVE TO EDIT. <<<

The application is already complete. For every face it
  1. finds the face and its landmarks,
  2. aligns/crops the face and computes a 512-d MagFace embedding
     (a vector of unit length: the "identity" of the face),
  3. calls YOUR predict_svm() to get a name and a similarity score,
  4. shows the name if the score >= the threshold slider, otherwise "Unknown".

You implement the two functions below. Until you do, the application runs but
every face is shown as "Unknown" and the Train button reports that
train_svm() is not implemented yet.

THE IDEA (separate classification from verification)
  * The SVM decides WHO the face most likely is (classification).
  * The cosine similarity to that person's average training embedding tells us
    HOW WELL the face matches that person (verification). Unlike predict_proba,
    this score does not become meaningless when there are 100 classes, and it is
    what the threshold slider is compared with.

Allowed: numpy, sklearn.svm.SVC.
Not allowed: predict_proba / probability=True, decision_function,
GridSearchCV / cross-validation, other classifiers (KNN, LinearSVC, ...).
Keep the code simple: you will explain it line by line in the oral exam.
"""

import numpy as np
from sklearn.svm import SVC


def train_svm(embeddings, labels, kernel, C, gamma):
    """Train the classifier and prepare everything predict_svm() needs.

    Parameters
        embeddings : np.ndarray, shape (N, 512), float32.
                     One row per training photo, every row has unit length.
        labels     : list of N strings. labels[i] is the person (= folder name)
                     in photo i. Several rows share the same name.
        kernel     : "linear" or "rbf"   (from the GUI)
        C          : float, SVM regularisation  (from the GUI)
        gamma      : "scale" or "auto"   (from the GUI; only used by "rbf")

    Returns
        A dict with exactly these three entries (the application saves it to
        assets/models/svm_model.joblib and hands it back to predict_svm()):
            "classifier"  : the trained sklearn SVC object
            "class_names" : list of the unique person names, sorted alphabetically
            "centroids"   : np.ndarray, shape (number_of_classes, 512);
                            centroids[i] is the representative embedding of
                            class_names[i]

    Expected steps (follow them in this order):

      1. class_names = the unique names in `labels`, sorted alphabetically.
         (set() removes duplicates, sorted() orders them.)

      2. Create the classifier:  SVC(kernel=kernel, C=C, gamma=gamma)
         Do not pass any other argument (in particular NOT probability=True).

      3. Train it with classifier.fit(embeddings, labels).
         sklearn accepts the string labels directly.

      4. Build one representative embedding ("centroid") per class:
           - create centroids = np.zeros((len(class_names), 512)), float32
           - loop over the classes: for i, name in enumerate(class_names)
               * collect the rows of `embeddings` whose label equals `name`
                 (use a plain for-loop over the indices of `labels`)
               * take their mean with np.mean(..., axis=0)
               * divide the mean by its length (np.linalg.norm) so that the
                 centroid has unit length again, and store it in centroids[i]
         Why must the centroid be normalised again?  (oral question)

      5. Return {"classifier": ..., "class_names": ..., "centroids": ...}.
    """
    # TODO: implement steps 1-5 above.
    return None   # placeholder: the GUI will report "train_svm() is not implemented"


def predict_svm(model, embedding):
    """Recognise one face.

    Parameters
        model     : the dict returned by train_svm()
        embedding : np.ndarray, shape (512,), float32, unit length

    Returns
        (name, similarity)
            name       : str   - the class predicted by the SVM
            similarity : float - cosine similarity between `embedding` and the
                                 centroid of that predicted class (-1 .. 1)

    Do NOT compare with the threshold here and do NOT return "Unknown":
    the application does that with the slider value.

    Expected steps (follow them in this order):

      1. The SVM expects a 2-D array: x = embedding.reshape(1, -1)

      2. name = model["classifier"].predict(x)[0]

      3. Find the position of `name` in model["class_names"]
         (a list: use its .index() method).

      4. Take that row of model["centroids"] and compute the cosine similarity
         with np.dot(embedding, centroid).
         Both vectors have unit length, so the dot product IS the cosine
         similarity. Convert it to a python float.

      5. return name, similarity
    """
    # TODO: implement steps 1-5 above.
    return "Unknown", 0.0   # placeholder: everything is shown as Unknown
